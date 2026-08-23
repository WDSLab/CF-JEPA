"""
CF-JEA Encoder — MultiScale Dilated Depthwise Convolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDWConv1d(nn.Module):
    def __init__(self, channels, kernel_size, dilation=1, strict_causal=False):
        super().__init__()
        self.receptive = (kernel_size - 1) * dilation
        self.strict_causal = strict_causal
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              dilation=dilation, groups=channels, bias=False)

    def forward(self, x):
        if self.strict_causal:
            x = F.pad(x, (self.receptive, 0))
        else:
            p = self.receptive // 2
            r = self.receptive - p
            x = F.pad(x, (p, r))
        return self.conv(x)


class MultiScaleDWConvBlock(nn.Module):
    def __init__(self, channels, dilation=1, strict_causal=False):
        super().__init__()
        self.dw_convs = nn.ModuleList([
            CausalDWConv1d(channels, k, dilation=dilation, strict_causal=strict_causal)
            for k in [3, 9, 15]
        ])
        self.bn1 = nn.BatchNorm1d(channels)
        self.pw = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = sum(dw(x) for dw in self.dw_convs)
        out = F.gelu(self.bn1(out))
        out = F.gelu(self.bn2(self.pw(out)))
        return out + residual


class Encoder(nn.Module):
    """MultiScale DWConv encoder for time series.

    Args:
        input_dims: Number of input channels
        hidden: Hidden dimension
        output: Output representation dimension
        depth: Number of DWConv blocks
        n_bins: Number of temporal bins for pooling (used in evaluation)
        strict_causal: If True, use left-only padding (for forecasting)
    """
    def __init__(self, input_dims=1, hidden=64, output=320, depth=4,
                 n_bins=8, strict_causal=False):
        super().__init__()
        self.n_bins = n_bins
        self.input_fc = nn.Linear(input_dims, hidden)
        self.blocks = nn.ModuleList([
            MultiScaleDWConvBlock(hidden, dilation=2 ** min(i, 7),
                                  strict_causal=strict_causal)
            for i in range(depth)
        ])
        self.output_fc = nn.Linear(hidden, output)

    def forward(self, x):
        """x: (B, T, C) -> (B, T, D)"""
        h = self.input_fc(x).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)
        return self.output_fc(h)
