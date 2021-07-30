"""
Author: Isamu Isozaki (isamu.website@gmail.com)
Description: Run sparse model
Created:  2021-07-23T16:42:02.952Z
Modified: 2021-07-23T16:42:24.046Z
Modified By: Isamu Isozaki
"""
from tqdm import tqdm, trange
from model import Activity, Dictionary, SparseModel, sparsity_loss, dictionary_loss

def run_sparse_model(trainloader, epochs, batch_size):
    """
    Train sparse model and returns it

    Args:
        trainloader (torch.utils.data.DataLoader): Data loader
        epochs (int): number of epochs
        batch_size (int): the size of each batch
    """
    