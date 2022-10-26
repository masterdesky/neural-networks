import os
import torch
import pandas as pd

class CustomDataset(torch.utils.data.Dataset):
    '''
    TODO
    '''
    def __init__(self,
                 data_path : str=None, labels_path : str=None):
        assert data_path != None, "A valid path to the dataset should be given!"
        assert labels_path != None, "A valid path to the labels should be given!"
        
        self.data = pd.read_csv(data_path)
        self.labels = pd.read_csv(labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx, :]
        label = self.labels[idx]
        
        return data, label