import os.path as osp
import numpy as np
import warnings
import glob
import pickle
import math
import h5py

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .metadata_helper import load_metadata
from .hdf5_loader import load_camera_imgs


class RobonetDataset(Dataset):
    """ Generic dataset for video files stored in folders
    Returns BCTHW videos in the range [-1, 1] """

    def __init__(self, config, train=True):
        super().__init__()
        self.train = train
        self.config = config
        self.seq_len = 4

        self.metadata = load_metadata('/home/wilson/data/robonet')
        n_train = int(0.99 * len(self.metadata))
        rng = np.random.RandomState(0)
        idxs = rng.permutation(len(self.metadata))
        if train:
            self.idxs = idxs[:n_train]
        else:
            self.idxs = idxs[n_train:]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        fname = self.metadata.files[self.idxs[idx]]
        md = self.metadata.get_file_metadata(fname)

        T = min([md['img_T'], md['action_T'] + 1])
        start_idx = np.random.randint(0, T - self.seq_len + 1)

        f = h5py.File(fname, 'r')
        images = load_camera_imgs(0, f, md, md['frame_dim'], start_idx, self.seq_len)
        images = torch.FloatTensor(images.copy())
        images = preprocess(images, self.config.size)

        return dict(image=images)

def preprocess(video, image_size, sequence_length=None):
    # video: THWC, {0, ..., 255}

    resolution = image_size

    video = video.movedim(-1, 1).float() / 255. # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

    video = 2 * video - 1 # [0, 1] -> [-1, 1]

    return video.movedim(1, -1)

        
class RobonetDatasetTrain(RobonetDataset):
    def __init__(self, config):
        super().__init__(config, train=True)

class RobonetDatasetTest(RobonetDataset):
    def __init__(self, config):
        super().__init__(config, train=False) 
