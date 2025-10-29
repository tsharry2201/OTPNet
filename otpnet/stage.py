"""OTPNet stage definition."""

from __future__ import annotations

import torch
from torch import nn

from .proximal_network import ProximalNetwork


class OTPStage(nn.Module):
    """Single optimization-inspired stage in OTPNet."""

    def __init__(
        self,
        hidden_channels: int,
        *,
        proximal_layers: int = 3,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        in_channels = hidden_channels * 3
        self.proximal = ProximalNetwork(
            in_channels,
            hidden_channels,
            hidden_channels,
            num_layers=proximal_layers,
            norm=norm,
        )
        self.pan_transform = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.ms_transform = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)

    def forward(
        self,
        h_prev: torch.Tensor,
        pan_feat: torch.Tensor,
        ms_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Update latent HR-MS representation."""
        fused = torch.cat([h_prev, pan_feat, ms_feat], dim=1)
        prox_features, params = self.proximal(fused)

        lambda_map = params["lambda"]
        mu_map = params["mu"]
        alpha_map = torch.sigmoid(params["alpha"])
        beta_map = torch.sigmoid(params["beta"])

        pan_guided = self.pan_transform(prox_features)
        ms_guided = self.ms_transform(ms_feat)

        update = alpha_map * (lambda_map * pan_guided) + beta_map * (mu_map * ms_guided)
        return h_prev + update


__all__ = ["OTPStage"]
