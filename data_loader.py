'''
Author: Qi7
Date: 2022-08-16 17:01:11
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-08-16 17:01:27
Description: 
'''
import random
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


class SiameseNetworkDataset(Dataset):
    def __init__(self, my_dataset):
        self.data = my_dataset
    
    def __getitem__(self, index):
        s0_tuple = random.choice(self.data)

        # we need to approximately 50% of signals to be in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # Look until the same class signal is found
                s1_tuple = random.choice(self.data)
                if int(s0_tuple[-1]) == int(s1_tuple[-1]):
                    break
        else:
            while True:
                s1_tuple = random.choice(self.data)
                if int(s0_tuple[-1]) != int(s1_tuple[-1]):
                    break
        
        return s0_tuple[:-1], s1_tuple[:-1], s0_tuple[-1], s1_tuple[-1], torch.from_numpy(np.array([int(s0_tuple[-1] != s1_tuple[-1])], dtype=np.float32))
    
    def __len__(self):
        return len(self.data)