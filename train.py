"""Training script for OTPNet on WV3 datasets."""

from __future__ import annotations

import argparse
import math
import platform
from pathlib import Path
from typing import Dict

import torch
from torch import nn
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore[assignment]

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore[assignment]

from otpnet import OTPNet
from otpnet.data import create_dataloader
from otpnet.metrics import psnr, sam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OTPNet for pansharpening.")
    parser.add_argument("--train-file", type=Path, default=Path("dataset/train_wv3.h5"))
    parser.add_argument("--valid-file", type=Path, default=Path("dataset/valid_wv3.h5"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.5, help="StepLR decay factor.")
    parser.add_argument("--step-size", type=int, default=60, help="StepLR step size.")
    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--proximal-layers", type=int, default=3)
    parser.add_argument("--norm", type=str, default="layer", choices=["layer", "instance", "batch"])
    parser.add_argument("--upsample-mode", type=str, default="bicubic")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scale", type=float, default=2047.0, help="Value range scaling factor.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases.")
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="tszharry-xi-an-jiaotong-university",
        help="Weights & Biases entity (user or team).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="OTPNet",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional run name for Weights & Biases.",
    )
    return parser.parse_args()


def prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device, scale: float) -> Dict[str, torch.Tensor]:
    scaled = {}
    for key, tensor in batch.items():
        scaled[key] = tensor.to(device) / scale
    return scaled


def main() -> None:
    args = parse_args()
    if platform.system() == "Windows" and args.num_workers > 0:
        print("Detected Windows environment; forcing num_workers=0 to avoid h5py multiprocessing issues.")
        args.num_workers = 0

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_loader = create_dataloader(
        args.train_file,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = create_dataloader(
        args.valid_file,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = OTPNet(
        pan_channels=1,
        ms_channels=train_loader.dataset[0]["lr_ms"].shape[0],
        hidden_channels=args.hidden_channels,
        num_stages=args.num_stages,
        proximal_layers=args.proximal_layers,
        norm=args.norm,
        upsample_mode=args.upsample_mode,
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=args.output_dir / "logs")
    elif not args.no_tensorboard and SummaryWriter is None:
        print("TensorBoard not available; proceeding without logging.")

    wandb_run = None
    if args.wandb:
        if wandb is None:
            print("Weights & Biases not available; proceeding without W&B logging.")
        else:
            wandb_kwargs = {
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "config": vars(args),
            }
            if args.wandb_run_name:
                wandb_kwargs["name"] = args.wandb_run_name
            wandb_run = wandb.init(**wandb_kwargs)

    best_psnr = -math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for iteration, batch in enumerate(train_loader, start=1):
            batch = prepare_batch(batch, device, args.scale)

            pred = model(batch["pan"], batch["lr_ms"])
            loss = criterion(pred, batch["hr_ms"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if iteration % args.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = running_loss / args.log_interval
                print(f"Epoch {epoch:03d} Iter {iteration:05d} | Loss {avg_loss:.4f} | LR {lr:.2e}")
                if writer:
                    global_step = (epoch - 1) * len(train_loader) + iteration
                    writer.add_scalar("train/loss", avg_loss, global_step)
                running_loss = 0.0

        scheduler.step()

        val_metrics = evaluate(model, valid_loader, device, args.scale)
        print(
            f"[Epoch {epoch:03d}] val_loss={val_metrics['loss']:.4f} "
            f"psnr={val_metrics['psnr']:.2f}dB sam={val_metrics['sam']:.4f}"
        )

        if writer:
            writer.add_scalar("valid/loss", val_metrics["loss"], epoch)
            writer.add_scalar("valid/psnr", val_metrics["psnr"], epoch)
            writer.add_scalar("valid/sam", val_metrics["sam"], epoch)

        checkpoint_path = args.output_dir / f"epoch_{epoch:03d}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "args": vars(args),
            },
            checkpoint_path,
        )

        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            best_path = args.output_dir / "best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "args": vars(args),
                },
                best_path,
            )
            print(f"Saved best model to {best_path}")

    if writer:
        writer.close()


def evaluate(model: OTPNet, loader: torch.utils.data.DataLoader, device: torch.device, scale: float) -> Dict[str, float]:
    criterion = nn.L1Loss()
    model.eval()
    losses = []
    psnr_scores = []
    sam_scores = []

    with torch.no_grad():
        for batch in loader:
            batch = prepare_batch(batch, device, scale)
            pred = model(batch["pan"], batch["lr_ms"])
            loss = criterion(pred, batch["hr_ms"])
            losses.append(loss.item())

            psnr_scores.append(psnr(pred, batch["hr_ms"]).mean().item())
            sam_scores.append(sam(pred, batch["hr_ms"]).mean().item())

    return {
        "loss": float(sum(losses) / len(losses)),
        "psnr": float(sum(psnr_scores) / len(psnr_scores)),
        "sam": float(sum(sam_scores) / len(sam_scores)),
    }


if __name__ == "__main__":
    main()
