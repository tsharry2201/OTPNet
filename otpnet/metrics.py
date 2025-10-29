"""Evaluation metrics for pansharpening."""

from __future__ import annotations

import torch


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio."""
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    return 20 * torch.log10(data_range) - 10 * torch.log10(torch.clamp(mse, min=1e-8))


def sam(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Spectral angle mapper (radians)."""
    b, c, h, w = pred.shape
    pred_v = pred.permute(0, 2, 3, 1).reshape(-1, c)
    target_v = target.permute(0, 2, 3, 1).reshape(-1, c)

    dot = torch.sum(pred_v * target_v, dim=1)
    pred_norm = torch.norm(pred_v, dim=1)
    target_norm = torch.norm(target_v, dim=1)

    cos = dot / torch.clamp(pred_norm * target_norm, min=eps)
    cos = torch.clamp(cos, -1.0, 1.0)
    return torch.acos(cos).reshape(b, h * w).mean(dim=1)


__all__ = ["l1_loss", "l2_loss", "psnr", "sam"]
