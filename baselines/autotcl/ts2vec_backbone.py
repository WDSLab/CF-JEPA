"""TS2Vec encoder backbone for AutoTCL classification / anomaly mode.

Verbatim copy of TS2Vec's dilated-conv encoder (AAAI 2022) — the
``TSEncoder``, ``DilatedConvEncoder``, ``ConvBlock``, ``SamePadConv``
and mask helpers from ``baselines/ts2vec/ts2vec/models/{encoder.py,
dilated_conv.py}``.

Why a local copy?
-----------------
AutoTCL paper claims a TS2Vec backbone for classification (Sec 4.2;
App D.1: "we followed the experiment setting of TS2Vec"; App D.3.1
Table 7 reports the TS2Vec-backbone ablation). The released
``AutoTCL_CoST.py`` only ships the CoST-no-SFD backbone, so we
restore the paper-stated backbone here.

Importing from ``baselines.ts2vec.ts2vec.models.encoder`` would
create a cross-baseline dependency on the TS2Vec upstream clone
(gitignored). Keeping a local copy under the AutoTCL wrapper directory
matches the CF-JEPA convention (see ``baselines/cf_jepa/encoder_variants.py``)
and ensures the file is committed and survives ``setup_baselines.sh``
re-clones.

Source: https://github.com/zhihanyue/ts2vec (TS2Vec, AAAI 2022).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Mask helpers (from ts2vec/models/encoder.py)
# ---------------------------------------------------------------------------
def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t:t + l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


# ---------------------------------------------------------------------------
# Dilated conv stack (from ts2vec/models/dilated_conv.py)
# ---------------------------------------------------------------------------
class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, :-self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final else None
        )

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1),
            )
            for i in range(len(channels))
        ])

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# TS2Vec encoder (from ts2vec/models/encoder.py)
# ---------------------------------------------------------------------------
class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3,
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):  # x: B x T x input_dims
        # AutoTCL calls forward twice in one step: _net(x) and _net(ax) where
        # ax = weight_s * mask_h * x. Original TS2Vec encoder mutated x in place
        # (`x[~nan_mask] = 0`), which corrupts ax's autograd graph (x version 0
        # captured in ax → version 1 after first call). Clone first to keep
        # the input pristine across multiple forward passes.
        x = x.clone()
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        return x
