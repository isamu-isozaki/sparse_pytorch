"""
Author: Isamu Isozaki (isamu.website@gmail.com)
Description: Pytorch implementation of ISTA sparse coding
Created:  2021-07-23T14:34:06.516Z
Modified: 2021-07-23T14:34:36.521Z
Modified By: Isamu Isozaki
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
dtype = torch.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Activity(nn.Module):
    def __init__(self, units, batch_size, alpha, sparsity_coef):
        """
        Initialize activity

        Args:
            units (int): number of dictionary filters/size of activity
            batch_size (int): batch size
            alpha (float): learning rate
            sparsity_coef (float): sparsity weight. How much sparsity is prioritized
        """
        super(Activity, self).__init__()
        self.w_init = torch.randn
        self.batch_size = batch_size
        self.units = units
        self.w = Variable(
            self.w_init(batch_size, units).type(dtype),
            requires_grad=False,
        ).to(device)
        self.rate = alpha
        self.sparsity_coef = sparsity_coef
    def shrink(self):
        """
        Shrinks the weights so that the ones that goes to 0 goes to 0
        """
        a = torch.clone(self.w).type(dtype).to(device)
        b = self.rate * self.sparsity_coef
        prior_shrink = torch.abs(a) - b
        shrink_positive = torch.clip(prior_shrink, 0, np.inf)
        sign_a = torch.sign(a)
        self.w = sign_a * shrink_positive
    def update(self, dictionary, x):
        """
        Updates activity weights

        Args:
            dictionary (torch tensor): dictionary
            x (torch tensor): input
        """
        batch_size = x.shape[0]
        w = torch.clone(self.w[:batch_size]).type(dtype).to(device)
        self.w[:batch_size] = w-self.rate* \
            torch.einsum('ij, ki->kj', dictionary, (torch.einsum('ij,kj->ki', dictionary, w)-x))
        self.shrink()
    def reset(self):
        """
        Resets weights for a new batch
        """
        self.w = self.w_init(self.batch_size, self.units).type(dtype).to(device)
    def forward(self, dictionary, batch_size):
        """
        Forward computation of activity

        Args:
            dictionary (torch tensor): dictionary
            batch_size (int): batch size of data

        Returns:
            torch tensor: reconstruction
        """
        return torch.einsum('ij,kj->ki', dictionary, self.w[:batch_size])

class Dictionary(nn.Module):
    def __init__(self, units, dict_filter_size, beta):
        """
        Initialize dictionary

        Args:
            units (int): number of filters
            dict_filter_size (int): size of each filter
            beta (float): between 0 and 1. The proportion to keep from previous A and B when updating
        """
        super(Dictionary, self).__init__()
        self.w_init = torch.randn
        self.units = units
        self.beta = beta
        self.A = Variable(
            self.w_init(units, units).type(dtype),
            requires_grad=False
        ).to(device)
        self.B = Variable(
            self.w_init(dict_filter_size, units).type(dtype),
            requires_grad=False
        ).to(device)
        self.w = Variable(
            self.w_init(dict_filter_size, units).type(dtype),
            requires_grad=False
        ).to(device)
    def forward(self, inputs, activity):
        """
        Forward computation. Gets reconstruction of inputs given activities

        Args:
            inputs (torch tensor): not used
            activity (torch tensor): activity

        Returns:
            torch tensor: reconstruction
        """
        return torch.matmul(self.w, activity)
    def update_AB(self, activity, x):
        """
        Update A and B

        Args:
            activity (torch tensor): activity
            x (torch tensor): input
        """
        A = torch.clone(self.A).type(dtype).to(device)
        B = torch.clone(self.B).type(dtype).to(device)
        batch_size = x.shape[0]
        self.A = self.beta*A + (1-self.beta)*torch.mean(\
            torch.einsum('ij,ik->ijk', activity[:batch_size], activity[:batch_size]), 0)
        self.B = self.beta*B + (1-self.beta)*torch.mean(torch.einsum('ij,ik->ijk', x, activity[:batch_size]), 0)
    def update(self):
        """
        Update dictionary
        """
        epsilon = 1e-5
        w = torch.clone(self.w).type(dtype).to(device)
        for i in range(self.units):
            w[:, i] = 1/(self.A[i, i]+epsilon) * (self.B[:, i] - torch.einsum('ij,j->i',w,self.A[:, i])+w[:, i]*self.A[i, i])
            w[:, i] = self.w[:, i] / (torch.norm(w[:, i])+epsilon)
        self.w = w
def dictionary_loss(dictionary, activity, x):
    """
    Get dictionary reconstruction loss

    Args:
        dictionary (torch tensor): dictionary
        activity (torch tensor): activity
        x (torch tensor): input tensor

    Returns:
        torch tensor: reconstruction loss
    """
    return torch.mean(0.5*torch.square(torch.einsum('ij,kj->ki', dictionary, activity) - x))
def sparsity_loss(activity, sparsity_coef):
    """
    Get sparsity loss

    Args:
        activity (torch tensor): activity
        sparsity_coef (float): sparsity coefficient

    Returns:
        torch tensor: sparsity loss
    """
    return torch.mean(torch.abs(activity)*sparsity_coef)

class SparseModel(nn.Module):
    def __init__(self, activity, dictionary, activity_epochs, dict_filter_size, \
                data_size, batch_size, num_layers):
        """
        Initialize sparse model

        Args:
            activity (Activity): activity
            dictionary (Dictionary): dictionary
            activity_epochs (int): epochs to wait for 
            dict_filter_size (int): size of each filter
            batch_size (int): batch size
            num_layers (int): number of layers. Currently unimplemented
        """
        super(SparseModel, self).__init__()
        self.activity = activity
        self.dictionary = dictionary
        self.dict_filter_size = dict_filter_size
        self.data_size = data_size
        self.batch_size = batch_size
        self.activity_epochs = activity_epochs
        self.batch_num = data_size // batch_size + (1 if (self.data_size % self.batch_size) else 0)
        self.num_layers = num_layers
    def compile(self, sparsity_loss, dictionary_loss):
        """
        Compile the losses

        Args:
            sparsity_loss (method): sparsity component of loss
            dictionary_loss (method): dictionary component of loss
        """
        self.sparsity_loss = sparsity_loss
        self.dictionary_loss = dictionary_loss
    def train_step_end(self):
        """
        Function to run when epoch ends. Updates dictionary based on A and B
        """
        self.dictionary.update()
    def train_step(self, patches):
        """
        Given batch, get activities and update A and B

        Args:
            patches (pytorch tensor): batch

        Returns:
            dict: dictionary of dictionary loss and sparsity loss
        """
        patches = patches.type(dtype).to(device)
        patches = torch.reshape(patches, [-1, self.dict_filter_size])
        self.activity.reset()
        
        dictionary = self.dictionary.w
        activity = self.activity.w
        for _ in range(self.activity_epochs):
            self.activity.update(dictionary, patches)
        dictionary_loss = self.dictionary_loss(dictionary, activity, patches)
        sparsity_loss = self.sparsity_loss(activity, self.activity.rate*self.activity.sparsity_coef)
        self.dictionary.update_AB(activity, patches)
        return {'dictionary loss': dictionary_loss, 'sparsity loss': sparsity_loss}
    def forward(self, patches):
        """
        Forward computation of batch data

        Args:
            patches (pytorch tensor): batch

        Returns:
            pytorch tensor: dictionary reconstructed version of patches
        """
        patches = patches.type(dtype).to(device)
        patches = torch.reshape(patches, [-1, self.dict_filter_size])
        dictionary = self.dictionary.w
        self.activity.reset()

        for _ in range(self.activity_epochs):
            self.activity.update(dictionary, patches)
        return self.activity.forward(self.dictionary.w, batch_size=patches.shape[0])
    def fit(self, trainset, epochs, batch_size):
        """
        Fit dictionary given training dataset

        Args:
            trainset (pd.DataFrame): training data
            epochs (int): number of epochs
            batch_size (int): number of batches
        """
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        for epoch in range(epochs):
            t = tqdm(enumerate(trainloader, 0), desc='Bar desc', leave=True)
            for i, data in t:
                losses = self.train_step(data)
                loss_text = f'epoch: {epoch}/{epochs} '
                for key in losses:
                    loss_text += key + ': {0:.4f} '.format(losses[key])
                t.set_description(loss_text, refresh=True) 
            self.train_step_end()
            