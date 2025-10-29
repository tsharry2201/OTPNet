"""Main OTPNet model definition."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .stage import OTPStage


class OTPNet(nn.Module):
    """OTPNet for PAN and MS image fusion."""

    def __init__(
        self,
        *,
        pan_channels: int = 1,
        ms_channels: int = 4,
        hidden_channels: int = 64,
        num_stages: int = 4,
        proximal_layers: int = 3,
        norm: str = "layer",
        upsample_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        self.ms_channels = ms_channels
        self.hidden_channels = hidden_channels
        self.upsample_mode = upsample_mode

        self.pan_encoder = nn.Sequential(
            nn.Conv2d(pan_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.ms_encoder = nn.Sequential(
            nn.Conv2d(ms_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.fusion_head = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.stages = nn.ModuleList(
            OTPStage(
                hidden_channels=hidden_channels,
                proximal_layers=proximal_layers,
                norm=norm,
            )
            for _ in range(num_stages)
        )

        self.reconstruction = nn.Conv2d(hidden_channels, ms_channels, kernel_size=3, padding=1)

    def _upsample_to_pan(self, lr_ms: torch.Tensor, pan_size: Tuple[int, int]) -> torch.Tensor:
        """Upsample LR-MS tensor to match PAN spatial resolution."""
        if lr_ms.shape[-2:] == pan_size:
            return lr_ms
        mode = self.upsample_mode
        align_corners = False if mode in {"bilinear", "bicubic", "trilinear"} else None
        return F.interpolate(lr_ms, size=pan_size, mode=mode, align_corners=align_corners)

    def forward(self, pan: torch.Tensor, lr_ms: torch.Tensor) -> torch.Tensor:
        """Forward pass producing a high-resolution MS prediction."""
        if pan.dim() != 4 or lr_ms.dim() != 4:
            raise ValueError("Expected 4D tensors for both pan and lr_ms inputs (B,C,H,W).")

        pan_size = pan.shape[-2:]
        ms_up = self._upsample_to_pan(lr_ms, pan_size)

        pan_feat = self.pan_encoder(pan)
        ms_feat = self.ms_encoder(ms_up)

        h = self.fusion_head(torch.cat([pan_feat, ms_feat], dim=1))

        for stage in self.stages:
            h = stage(h, pan_feat, ms_feat)

        return self.reconstruction(h)


__all__ = ["OTPNet"]
