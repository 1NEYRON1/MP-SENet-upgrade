import json
from pathlib import Path
import torch
import torch.nn as nn

from pesq import pesq
import os
import shutil


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    return AttrDict(config_dict)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    return "cpu"


def compute_pesq(ref, est, sr=16000):
    try:
        ref_np = ref.cpu().numpy().squeeze()
        est_np = est.cpu().numpy().squeeze()
        min_len = min(len(ref_np), len(est_np))
        return pesq(sr, ref_np[:min_len], est_np[:min_len], "wb")
    except:
        return 0.0


def compute_sisnr(ref, est):    
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    dot = (ref * est).sum(dim=-1, keepdim=True)
    s_ref = dot * ref / (ref.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
    e_noise = est - s_ref
    return (
        10
        * torch.log10(
            s_ref.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + 1e-8) + 1e-8
        )
        .mean()
        .item()
    )


class LearnableSigmoid1d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
