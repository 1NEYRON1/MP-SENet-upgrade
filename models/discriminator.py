from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
import numpy as np
from pesq import pesq
from joblib import Parallel, delayed
from utils import *


# def pesq_loss(clean, noisy, sr=16000):
#     try:
#         pesq_score = pesq(sr, clean, noisy, 'wb')
#     except:
#         # error can happen due to silent period
#         pesq_score = -1
#     return pesq_score


# def batch_pesq(clean, noisy):
#     pesq_score = Parallel(n_jobs=15)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
#     pesq_score = np.array(pesq_score)
#     if -1 in pesq_score:
#         return None
#     pesq_score = (pesq_score - 1) / 3.5
#     return torch.FloatTensor(pesq_score)

def cal_pesq(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, "wb")
    except Exception:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


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
    
class AsyncPESQ:
    def __init__(self, max_workers=4):
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self._futures = None

    def submit(self, clean_list, noisy_list, sr=16000):
        self._futures = [
            self.executor.submit(cal_pesq, c, n, sr)
            for c, n in zip(clean_list, noisy_list, strict=True)
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