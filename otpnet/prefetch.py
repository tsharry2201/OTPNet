"""Asynchronous mini-batch prefetching utilities."""

from __future__ import annotations

from typing import Dict, Iterator, Optional

import torch

from .batch_utils import prepare_batch


class CUDAPrefetcher:
    """Iterates over a DataLoader while overlapping host-to-device transfers."""

    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        scale: float,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CUDAPrefetcher requires a CUDA device.")
        self.loader = loader
        self.device = device
        self.scale = scale
        self.stream = torch.cuda.Stream(device=device)
        self._iterator: Optional[Iterator[Dict[str, torch.Tensor]]] = None
        self._next_batch: Optional[Dict[str, torch.Tensor]] = None

    def __iter__(self) -> "CUDAPrefetcher":
        self._iterator = iter(self.loader)
        self._next_batch = None
        self._preload()
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._next_batch is None:
            raise StopIteration
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self._next_batch
        self._preload()
        return batch

    def _preload(self) -> None:
        assert self._iterator is not None
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self._next_batch = prepare_batch(batch, self.device, self.scale)


__all__ = ["CUDAPrefetcher"]
