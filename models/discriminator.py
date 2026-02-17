import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor
from pesq import pesq
from torch.nn.utils.parametrizations import spectral_norm
from utils import *


def cal_pesq(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


class MetricDiscriminator(nn.Module):
    def __init__(self, dim=16, in_channel=2, dropout=0.3):
        super(MetricDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, dim, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            spectral_norm(nn.Conv2d(dim, dim*2, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*2, affine=True),
            nn.PReLU(dim*2),
            spectral_norm(nn.Conv2d(dim*2, dim*4, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*4, affine=True),
            nn.PReLU(dim*4),
            spectral_norm(nn.Conv2d(dim*4, dim*8, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*8, affine=True),
            nn.PReLU(dim*8),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            spectral_norm(nn.Linear(dim*8, dim*4)),
            nn.Dropout(dropout),
            nn.PReLU(dim*4),
            spectral_norm(nn.Linear(dim*4, 1)),
            LearnableSigmoid1d(1)
        )

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=1)
        return self.layers(xy)


class AsyncPESQ:
    def __init__(self, max_workers=4):
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self._futures = None

    def submit(self, clean_list, noisy_list, sr=16000):
        self._futures = [
            self.executor.submit(cal_pesq, c, n, sr)
            for c, n in zip(clean_list, noisy_list)
        ]

    def collect(self):
        if self._futures is None:
            return None
        scores = np.array([f.result() for f in self._futures])
        self._futures = None
        if -1 in scores:
            return None
        return torch.FloatTensor((scores - 1) / 3.5)

    def shutdown(self):
        self.executor.shutdown(wait=True)
