#!/usr/bin/env python3
"""Generate OTPNet predictions on a test set and export for evaluate_results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.io as sio
import torch
from torch.serialization import add_safe_globals

from otpnet import OTPNet
from otpnet.data import create_dataloader
from otpnet.metrics import psnr, sam
from evaluate import build_model  # reuse checkpoint loading helper

try:
    from pathlib import WindowsPath
except ImportError:  # pragma: no cover
    WindowsPath = None

if WindowsPath is not None:
    add_safe_globals({WindowsPath})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OTPNet on a test dataset and export .mat results.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--data-file", type=Path, default=Path("dataset/test_wv3_multiExm1.h5"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scale", type=float, default=2047.0, help="Value range scaling factor.")
    parser.add_argument("--output-file", type=Path, default=Path("results/test_predictions.mat"))
    parser.add_argument("--upsample-mode", type=str, default=None, help="Override checkpoint upsample mode.")
    parser.add_argument("--num-stages", type=int, default=None, help="Override checkpoint number of stages.")
    parser.add_argument("--hidden-channels", type=int, default=None, help="Override checkpoint hidden channels.")
    parser.add_argument("--proximal-layers", type=int, default=None, help="Override checkpoint proximal depth.")
    return parser.parse_args()


def prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device, scale: float) -> Dict[str, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return {k: v.to(device, non_blocking=non_blocking) / scale for k, v in batch.items()}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    loader = create_dataloader(
        args.data_file,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if len(loader.dataset) == 0:
        raise RuntimeError(f"No samples found in {args.data_file}.")

    sample = loader.dataset[0]
    model = build_model(args, sample, device)

    preds: List[torch.Tensor] = []
    gts: List[torch.Tensor] = []
    l1 = torch.nn.L1Loss()
    losses = []
    psnr_scores = []
    sam_scores = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            scaled = prepare_batch(batch, device, args.scale)
            pred = model(scaled["pan"], scaled["lr_ms"])

            preds.append(pred.cpu())
            gts.append(scaled["hr_ms"].cpu())

            losses.append(l1(pred, scaled["hr_ms"]).item())
            pred_rescaled = pred * args.scale
            hr_rescaled = scaled["hr_ms"] * args.scale
            psnr_scores.append(psnr(pred_rescaled, hr_rescaled, data_range=args.scale))
            sam_scores.append(sam(pred_rescaled, hr_rescaled))

    sr_tensor = torch.cat(preds, dim=0)
    gt_tensor = torch.cat(gts, dim=0)

    # Convert to numpy with channel-last ordering and original scale.
    sr_np = sr_tensor.numpy().astype(np.float32)
    gt_np = gt_tensor.numpy().astype(np.float32)

    sr_np = np.transpose(sr_np, (0, 2, 3, 1)) * args.scale
    gt_np = np.transpose(gt_np, (0, 2, 3, 1)) * args.scale

    output_path = args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(
        output_path,
        {
            "sr": sr_np,
            "gt": gt_np,
        },
    )

    avg_loss = float(sum(losses) / len(losses))
    avg_psnr = float(sum(psnr_scores) / len(psnr_scores))
    avg_sam = float(sum(sam_scores) / len(sam_scores))

    print(f"Saved predictions to {output_path}")
    print(f"Average L1: {avg_loss:.6f} | PSNR: {avg_psnr:.2f}dB | SAM: {avg_sam:.4f}")
    print("Use evaluate_results.py --mat_file", output_path, "to compute additional metrics.")


if __name__ == "__main__":
    main()
