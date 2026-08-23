"""
CF-JEA Losses — VICReg regularization components
"""

import torch
import torch.nn.functional as F


def variance_reg(z):
    """Variance regularization: encourage std >= 1 per dimension."""
    z_flat = z.reshape(-1, z.shape[-1])
    std = z_flat.std(dim=0)
    return F.relu(1.0 - std).mean()


def covariance_reg(z):
    """Covariance regularization: decorrelate dimensions."""
    z_flat = z.reshape(-1, z.shape[-1])
    z_centered = z_flat - z_flat.mean(dim=0)
    N = z_centered.shape[0]
    cov = (z_centered.T @ z_centered) / (N - 1)
    D = cov.shape[0]
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / D
