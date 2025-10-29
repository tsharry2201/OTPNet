"""ODE-inspired proximal network for OTPNet."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Mapping, Tuple

import torch
from torch import nn

from .weight_layer import WeightLayer


class ProximalNetwork(nn.Module):
    """Stack of weight layers that predicts modulation parameters."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_layers: int = 3,
        norm: str = "layer",
        parameter_names: Iterable[str] = ("lambda", "mu", "alpha", "beta"),
    ) -> None:
        super().__init__()
        self.parameter_names = tuple(parameter_names)

        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.layers = nn.ModuleList(
            WeightLayer(hidden_channels, norm=norm) for _ in range(num_layers)
        )
        self.feature_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

        self.heads = nn.ModuleDict(
            {
                name: nn.Conv2d(out_channels, out_channels, kernel_size=1)
                for name in self.parameter_names
            }
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Forward pass returning refined features and parameter maps."""
        features = self.input_proj(x)
        for layer in self.layers:
            features = layer(features)
        features = self.feature_proj(features)

        params = OrderedDict(
            (name, head(features)) for name, head in self.heads.items()
        )
        return features, params


__all__ = ["ProximalNetwork"]
