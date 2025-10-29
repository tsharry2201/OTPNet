"""Dataset utilities for OTPNet training on WV3 HDF5 files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import h5py
import torch
from torch.utils.data import DataLoader, Dataset


class WV3H5Dataset(Dataset):
    """Dataset wrapper for WV3 HDF5 pansharpening data."""

    def __init__(
        self,
        file_path: str | Path,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"H5 dataset not found: {self.file_path}")

        self.dtype = dtype
        with h5py.File(self.file_path, "r") as h5f:
            self.length = h5f["pan"].shape[0]

        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return self.length

    def _require_handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.file_path, "r")
        return self._handle

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        handle = self._require_handle()

        pan = torch.from_numpy(handle["pan"][index]).to(self.dtype)
        lr_ms = torch.from_numpy(handle["ms"][index]).to(self.dtype)
        lms = torch.from_numpy(handle["lms"][index]).to(self.dtype)
        hr_ms = torch.from_numpy(handle["gt"][index]).to(self.dtype)

        return {
            "pan": pan,
            "lr_ms": lr_ms,
            "lms": lms,
            "hr_ms": hr_ms,
        }

    def __del__(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = None


def create_dataloader(
    file_path: str | Path,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Helper to instantiate a DataLoader for an H5 dataset."""
    dataset = WV3H5Dataset(file_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


__all__ = ["WV3H5Dataset", "create_dataloader"]
