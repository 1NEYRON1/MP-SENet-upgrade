import sys
sys.path.append("..")
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pesq import pesq
from torch.nn.utils import spectral_norm
from joblib import Parallel, delayed
from utils import *
from dataset import mag_pha_stft


def cal_pesq(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=15)(delayed(cal_pesq)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score)


def metric_loss(metric_ref, metrics_gen):
    loss = 0
    for metric_gen in metrics_gen:
        metric_loss = F.mse_loss(metric_ref, metric_gen.flatten())
        loss += metric_loss

    return loss


class MetricDiscriminator(nn.Module):
    def __init__(self, dim=16, in_channel=2):
        super(MetricDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim*2, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*2, affine=True),
            nn.PReLU(dim*2),
            nn.utils.spectral_norm(nn.Conv2d(dim*2, dim*4, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*4, affine=True),
            nn.PReLU(dim*4),
            nn.utils.spectral_norm(nn.Conv2d(dim*4, dim*8, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*8, affine=True),
            nn.PReLU(dim*8),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(dim*8, dim*4)),
            nn.Dropout(0.3),
            nn.PReLU(dim*4),
            nn.utils.spectral_norm(nn.Linear(dim*4, 1)),
            LearnableSigmoid1d(1)
        )

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=1)
        return self.layers(xy)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, resolutions=[[400, 100, 400], [1024, 120, 600], [256, 50, 120]], compress_factor=0.3):
    
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            MetricDiscriminator() for _ in resolutions
        ])
        self.resolutions = resolutions
        self.compress_factor = compress_factor
        
    def forward(self, clean_wav, gen_wav):
        metric_rs = []
        metric_gs = []
        
        for i, (n_fft, hop, win) in enumerate(self.resolutions):
            clean_mag, _, _ = mag_pha_stft(clean_wav, n_fft, hop, win, compress_factor=self.compress_factor)
            gen_mag, _, _ = mag_pha_stft(gen_wav, n_fft, hop, win, compress_factor=self.compress_factor)
            
            metric_r = self.discriminators[i](clean_mag, clean_mag)
            metric_g = self.discriminators[i](clean_mag, gen_mag)
            
            metric_rs.append(metric_r)
            metric_gs.append(metric_g)
            
        return metric_rs, metric_gs
