"""Evaluation metrics for pansharpening."""

from __future__ import annotations

import torch


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio using the same formulation as evaluate_results.py."""
    pred_detached = pred.detach()
    target_detached = target.detach()

    mse = torch.mean((pred_detached - target_detached) ** 2)
    if mse.item() <= 1e-10:
        return float("inf")

    dtype = pred_detached.dtype if pred_detached.is_floating_point() else torch.float32
    eps = torch.finfo(dtype).eps
    data_range_tensor = torch.as_tensor(data_range, dtype=dtype, device=pred_detached.device)
    psnr_value = 20.0 * torch.log10(data_range_tensor / (torch.sqrt(mse) + eps))
    return float(psnr_value.detach().cpu())


def sam(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Spectral angle mapper matching evaluate_results.py (averaged across pixels)."""
    pred_detached = pred.detach()
    target_detached = target.detach()

    b, c, h, w = pred_detached.shape
    pred_pixels = pred_detached.permute(0, 2, 3, 1).reshape(b * h * w, c)
    target_pixels = target_detached.permute(0, 2, 3, 1).reshape(b * h * w, c)

    inner_product = torch.sum(pred_pixels * target_pixels, dim=1)
    pred_norm = torch.norm(pred_pixels, dim=1)
    target_norm = torch.norm(target_pixels, dim=1)

    cos_theta = inner_product / (pred_norm * target_norm + eps)
    cos_theta = torch.clamp(cos_theta, 0.0, 1.0)

    angles = torch.acos(cos_theta)
    return float(torch.mean(angles).detach().cpu())


__all__ = ["l1_loss", "l2_loss", "psnr", "sam"]
