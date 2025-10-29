"""Evaluation metrics for pansharpening."""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64)


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio using the same formulation as evaluate_results.py."""
    pred_np = _to_numpy(pred)
    target_np = _to_numpy(target)
    mse = np.mean((pred_np - target_np) ** 2)
    if mse <= 1e-10:
        return float("inf")
    eps = np.finfo(np.float64).eps
    return float(20.0 * np.log10(data_range / (np.sqrt(mse) + eps)))


def sam(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Spectral angle mapper matching evaluate_results.py (averaged across pixels)."""
    pred_np = _to_numpy(pred)
    target_np = _to_numpy(target)

    b, c, h, w = pred_np.shape
    pred_pixels = np.transpose(pred_np, (0, 2, 3, 1)).reshape(-1, c)
    target_pixels = np.transpose(target_np, (0, 2, 3, 1)).reshape(-1, c)

    inner_product = np.sum(pred_pixels * target_pixels, axis=1)
    pred_norm = np.sqrt(np.sum(pred_pixels ** 2, axis=1))
    target_norm = np.sqrt(np.sum(target_pixels ** 2, axis=1))

    cos_theta = inner_product / (pred_norm * target_norm + eps)
    cos_theta = np.clip(cos_theta, 0.0, 1.0)

    angles = np.arccos(cos_theta)
    return float(np.mean(angles))


__all__ = ["l1_loss", "l2_loss", "psnr", "sam"]
