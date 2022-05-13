import torch.nn.functional as F
import random
import math
import torch
from omegaconf import OmegaConf
import os.path as osp
import argparse
import glob

from torchvision.io import read_video
from taming.models.vqgan import VQModel
from cwvae.utils import save_video_grid


def preprocess(fname, image_size=128, sequence_length=16):
    # video: THWC, {0, ..., 255}

    video = read_video(fname, pts_unit='sec')[0]
    resolution = image_size

    video = video.permute(0, 3, 1, 2).float() / 255. # TCHW
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

    return video


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--ckpt', type=str, required=True)
args = parser.parse_args()


def load_config(config_path):
    config = OmegaConf.load(config_path)
    return config

def load_vqgan(config, ckpt_path):
    model = VQModel(**config.model.params)
    sd = torch.load(ckpt_path, map_location='cpu')['state_dict']
    model.load_state_dict(sd, strict=False)
    return model.eval()

def reconstruct(x, model):
    z, _, [_, _, indices] = model.encode(x)
    xrec = model.decode(z)
    return xrec


device = torch.device('cuda')
torch.set_grad_enabled(False)

config = load_config(osp.join(args.ckpt, 'configs', 'model.yaml'))
model = load_vqgan(config, ckpt_path=osp.join(args.ckpt, 'ckpts', 'last.ckpt')).to(device)

videos = glob.glob('/home/wilson/data/kinetics600/train/*/*.mp4')
random.seed(0)
random.shuffle(videos)
x = torch.stack([preprocess(videos[i]) for i in range(8)], dim=0).to(device)
B, T = x.shape[:2]
x_recon = []
for i in range(B):
    x_recon.append(reconstruct(x[i], model))
x_recon = torch.stack(x_recon, dim=0)
x_recon = torch.clamp(x_recon, -1, 1)

viz = torch.stack([x_recon, x], dim=1).flatten(end_dim=1) * 0.5 + 0.5
print(viz.shape)
save_video_grid(viz, 'recon.gif')

