
from torch.utils.data import DataLoader 
import numpy as np


from pyfiles.dataset import SiameseDataset
from pyfiles.utils import make_pairs

def make_data_loader(data, labels, batch_size, shuffle):
    """ Image pairs (A, P) & (A, N) and returns a data_loader
    """
    # first we need to get the indeces 
    idx = [np.where(labels == l)[0] for l in range(0,10)]
    # now we can get the data 
    pos, neg, y = make_pairs(data, idx)
    # now we can make a dataset 
    dataset = SiameseDataset(pos, neg, y)
    # and we can convert it to a data loader 
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=shuffle, num_workers=2)
    
    return dataloader