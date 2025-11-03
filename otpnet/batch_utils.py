"""Utilities for preparing OTPNet mini-batches."""

from __future__ import annotations

from typing import Dict

import torch


def prepare_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    scale: float,
) -> Dict[str, torch.Tensor]:
    """Move a batch of tensors to the target device and normalise by *scale*."""
    non_blocking = device.type == "cuda"
    return {
        key: tensor.to(device, non_blocking=non_blocking) / scale
        for key, tensor in batch.items()
    }


__all__ = ["prepare_batch"]
