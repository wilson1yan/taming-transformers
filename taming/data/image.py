import os.path as osp
import warnings
import glob
import pickle
import math
import json

import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, config, train=True):
        super().__init__()
        self.train = train
        self.config = config
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        split = 'train' if train else 'test'
        self.dataset = ImageFolder(osp.join(config.data_path, split), transform)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        image = image[None].movedim(1, -1)
        return dict(image=image)

        
class ImageDatasetTrain(ImageDataset):
    def __init__(self, config):
        super().__init__(config, train=True)

class ImageDatasetTest(ImageDataset):
    def __init__(self, config):
        super().__init__(config, train=False)
