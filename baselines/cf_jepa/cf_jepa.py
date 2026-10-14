"""
CF-JEPA — Crop-based Forward Joint-Embedding Predictive Architecture for Time Series.

Paper main:
  1. Random crops → online encoder → 3 zone-conditional LinearPredictors
     forecast future-half latents in 3 zones (L1 on cosine-normalized targets)
  2. Target encoder (EMA of online encoder, stop-grad) — provides prediction
     targets, used at inference for forecasting (use_target=True)
  3. VICReg (variance + covariance) on mean-pooled crop representations
  4. Multi-scale invariance: crops from same sequence match at pool(2, 4, 8)
  5. Horizon annealing: predictor loss weight decays linearly 1 → 0 over training

Evaluation:
  - Classification: online encoder → adaptive_avg_pool(n_bins) → flatten → SVM
  - Forecasting: target encoder → causal encoding → Ridge regression

This module exposes the static building blocks (PARAMS dicts, the zone
predictor, and sample_crop) consumed by the trainer (`trainer.py`). The actual
training loop lives in `CFJEPATrainer.fit`.
"""

import numpy as np
import torch
import torch.nn as nn

# Optuna-tuned hyperparameters (13 hard UCR datasets, n_bins=8).
# NOTE: enc_depth in this dict is overwritten by the trainer's `depth=5`
# default (CFJEPATrainer.__init__) — depth=5 is the value actually used in
# all experiments. The values below reflect the trainer-applied configuration.
PARAMS_128 = dict(
    lr=0.000340, enc_hidden=256, enc_output=128, enc_depth=5, n_crops=4,
    crop_min=0.310, crop_max=0.686, ema_decay=0.992,
    lambda_var=0.038, lambda_cov=0.143, lambda_inv=0.401, n_bins=8,
)

PARAMS_320 = dict(
    lr=0.000225, enc_hidden=256, enc_output=320, enc_depth=5, n_crops=4,
    crop_min=0.367, crop_max=0.418, ema_decay=0.983,
    lambda_var=0.081, lambda_cov=0.076, lambda_inv=1.101, n_bins=8,
)


class LinearPredictor(nn.Module):
    """Near-identity linear predictor for forward latent prediction."""
    def __init__(self, dim=320):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dim) + torch.randn(dim, dim) * 0.01)
            self.linear.bias.zero_()

    def forward(self, z):
        return self.linear(z)


def sample_crop(T, crop_min=0.2, crop_max=0.5):
    """Sample a random crop from a time series of length T."""
    crop_len = max(int(T * np.random.uniform(crop_min, crop_max)), 4)
    max_start = max(T - crop_len - 3, 0)
    crop_start = np.random.randint(0, max_start + 1)
    return crop_start, crop_start + crop_len
