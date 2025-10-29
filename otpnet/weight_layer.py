"""Weight layer implementation for OTPNet.

This module contains the residual building block that stacks several convolutions,
normalisation layers, and a GELU activation. It follows the structure described
in the OTPNet blueprint and is reused inside the proximal network.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def _make_norm(norm: str, num_features: int) -> nn.Module:
    """Construct a normalisation layer for 2D feature maps."""
    norm = norm.lower()
    if norm == "layer":
        # GroupNorm with a single group behaves like LayerNorm for conv features.
        return nn.GroupNorm(1, num_features)
    if norm == "instance":
        return nn.InstanceNorm2d(num_features, affine=True)
    if norm == "batch":
        return nn.BatchNorm2d(num_features)
    raise ValueError(f"Unsupported norm type '{norm}'. Expected layer|instance|batch.")


class WeightLayer(nn.Module):
    """Residual weight layer used by the proximal network."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        norm: str = "layer",
        activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm1 = _make_norm(norm, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.activation = activation if activation is not None else nn.GELU()
        self.norm2 = _make_norm(norm, channels)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block."""
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.norm2(out)
        out = self.conv3(out)
        return out + residual


__all__ = ["WeightLayer"]
