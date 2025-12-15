"""
MinGRU Experiment: Replacing GRU with MinGRU in GTCRN and MPNet models.

Based on paper: "Were RNNs All We Needed?" (arXiv:2410.01201)
https://arxiv.org/pdf/2410.01201

Usage:
    from experiments.min_gru import MinGTCRN, MinMPNet
"""

from .mingru_models import (
    MinGRU,
    MinGTCRN,
    MinMPNet,
    MinTRA,
    MinGRNN,
    MinDPGRNN,
    MinFFN,
    MinTransformerBlock,
)

__all__ = [
    "MinGRU",
    "MinGTCRN",
    "MinMPNet",
    "MinTRA",
    "MinGRNN",
    "MinDPGRNN",
    "MinFFN",
    "MinTransformerBlock",
]
