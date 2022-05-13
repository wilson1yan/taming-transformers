import os.path as osp
import warnings
import glob
import pickle
import math
import h5py

import torch
from torch.utils.data import Dataset


class H5PyDataset(Dataset):
    """ Generic dataset for video files stored in folders
    Returns BCTHW videos in the range [-1, 1] """

    def __init__(self, config, train=True):
        super().__init__()
        self.train = train
        self.config = config

        hf = h5py.File(config.data_path, 'r')
        split = 'train' if train else 'test'
        self._images = hf[f'{split}_data']

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]
        image = torch.FloatTensor(image)
        image = 2 * (image / 255.) - 1
        return dict(image=image)

        
class H5PyDatasetTrain(H5PyDataset):
    def __init__(self, config):
        super().__init__(config, train=True)

class H5PyDatasetTest(H5PyDataset):
    def __init__(self, config):
        super().__init__(config, train=False) 
