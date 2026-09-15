"""Convolutional Block Attention Module (CBAM).

Based on the channel-then-spatial attention mechanism introduced by Woo et al.,
ECCV 2018. This implementation is written for this repository and follows the
paper equations rather than copying an external implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if reduction <= 0:
            raise ValueError("reduction must be positive")
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in {3, 7}:
            raise ValueError("CBAM spatial kernel must be 3 or 7")
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))


class CBAM(nn.Module):
    """Sequential channel and spatial feature refinement."""

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)
