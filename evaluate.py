"""Evaluation script for OTPNet checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from otpnet import OTPNet
from otpnet.data import create_dataloader
from otpnet.metrics import psnr, sam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OTPNet checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, default=Path("dataset/valid_wv3.h5"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--upsample-mode", type=str, default=None, help="Override checkpoint upsample mode.")
    parser.add_argument("--num-stages", type=int, default=None, help="Override checkpoint number of stages.")
    parser.add_argument("--hidden-channels", type=int, default=None, help="Override checkpoint hidden channels.")
    parser.add_argument("--proximal-layers", type=int, default=None, help="Override checkpoint proximal depth.")
    return parser.parse_args()


def build_model(args: argparse.Namespace, sample: Dict[str, torch.Tensor], device: torch.device) -> OTPNet:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args: Dict[str, any] = {}
    if isinstance(checkpoint, dict) and "args" in checkpoint:
        stored = checkpoint.get("args")
        if isinstance(stored, dict):
            ckpt_args = stored

    ms_channels = sample["lr_ms"].shape[0]

    config = {
        "pan_channels": 1,
        "ms_channels": ms_channels,
        "hidden_channels": args.hidden_channels or ckpt_args.get("hidden_channels", 64),
        "num_stages": args.num_stages or ckpt_args.get("num_stages", 4),
        "proximal_layers": args.proximal_layers or ckpt_args.get("proximal_layers", 3),
        "norm": ckpt_args.get("norm", "layer"),
        "upsample_mode": args.upsample_mode or ckpt_args.get("upsample_mode", "bicubic"),
    }

    model = OTPNet(**config)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    return model


def evaluate(model: OTPNet, loader: torch.utils.data.DataLoader, device: torch.device, scale: float) -> Dict[str, float]:
    losses = []
    psnr_scores = []
    sam_scores = []
    l1 = torch.nn.L1Loss()

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) / scale for k, v in batch.items()}
            pred = model(batch["pan"], batch["lr_ms"])
            loss = l1(pred, batch["hr_ms"])
            losses.append(loss.item())
            psnr_scores.append(psnr(pred, batch["hr_ms"]).mean().item())
            sam_scores.append(sam(pred, batch["hr_ms"]).mean().item())

    return {
        "loss": float(sum(losses) / len(losses)),
        "psnr": float(sum(psnr_scores) / len(psnr_scores)),
        "sam": float(sum(sam_scores) / len(sam_scores)),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    loader = create_dataloader(
        args.data_file,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    sample = loader.dataset[0]
    model = build_model(args, sample, device)

    metrics = evaluate(model, loader, device, args.scale)
    print(
        f"Evaluation on {args.data_file} | "
        f"loss={metrics['loss']:.4f} psnr={metrics['psnr']:.2f}dB sam={metrics['sam']:.4f}"
    )


if __name__ == "__main__":
    main()
