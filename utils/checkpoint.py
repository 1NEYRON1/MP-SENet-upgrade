"""Checkpoint utilities for MPNet training."""

import os
import glob
import torch


def scan_checkpoint(cp_dir, prefix):
    """Scan for the latest checkpoint file with given prefix."""
    pattern = os.path.join(cp_dir, prefix + "*")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def load_checkpoint(filepath, device):
    """Load checkpoint from file."""
    assert os.path.isfile(filepath)
    checkpoint_dict = torch.load(filepath, map_location=device)
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    """Save checkpoint to file."""
    torch.save(obj, filepath)
    print(f"Saved checkpoint: {filepath}")










