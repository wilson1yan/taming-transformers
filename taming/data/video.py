import os.path as osp
import warnings
import glob
import pickle
import math

import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets.video_utils import VideoClips


class VideoDataset(Dataset):
    """ Generic dataset for video files stored in folders
    Returns BCTHW videos in the range [-1, 1] """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, config, train=True):
        super().__init__()
        self.train = train
        self.config = config
        self.seq_len = 1

        folder = osp.join(config.data_path, 'train' if train else 'test')
        files = sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
                     for ext in self.exts], [])
        warnings.filterwarnings('ignore')
        cache_file = osp.join(folder, f"metadata_{self.seq_len}.pkl")
        if not osp.exists(cache_file):
            clips = VideoClips(files, self.seq_len, num_workers=48)
        else:
            metadata = pickle.load(open(cache_file, 'rb'))
            clips = VideoClips(files, self.seq_len,
                               _precomputed_metadata=metadata)
        self._clips = clips

    def __len__(self):
        return self._clips.num_clips()

    def __getitem__(self, idx):
        video, _, _, idx = self._clips.get_clip(idx)
        video = preprocess(video, self.config.size)
        image = video[0]
        return dict(image=image)

        
class VideoDatasetTrain(VideoDataset):
    def __init__(self, config):
        super().__init__(config, train=True)

class VideoDatasetTest(VideoDataset):
    def __init__(self, config):
        super().__init__(config, train=False)
        
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
