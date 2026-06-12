from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticSleepConfig:
    train_size: int
    val_size: int
    signal_length: int
    temp_seq_len: int
    temp_grid_size: int
    num_classes: int
    hr_range: tuple[float, float]
    rr_range: tuple[float, float]
    noise_std: float


class SyntheticSleepDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        size: int,
        signal_length: int,
        temp_seq_len: int,
        temp_grid_size: int,
        num_classes: int,
        hr_range: tuple[float, float],
        rr_range: tuple[float, float],
        noise_std: float,
        seed: int,
        transform: Any | None = None,
    ) -> None:
        self.size = size
        self.signal_length = signal_length
        self.temp_seq_len = temp_seq_len
        self.temp_grid_size = temp_grid_size
        self.num_classes = num_classes
        self.hr_range = hr_range
        self.rr_range = rr_range
        self.noise_std = noise_std
        self.transform = transform
        generator = torch.Generator().manual_seed(seed)
        self.labels = torch.arange(size) % num_classes
        self.bcg = self._build_bcg(generator)
        self.temp = self._build_temp(generator)
        self.hr = self._build_targets(self.labels, hr_range, generator)
        self.rr = self._build_targets(self.labels, rr_range, generator)

    def _build_bcg(self, generator: torch.Generator) -> torch.Tensor:
        time_axis = torch.linspace(0.0, 1.0, self.signal_length)
        signals = []
        for index in range(self.size):
            label = int(self.labels[index].item())
            freq = 1.0 + 0.2 * label
            base = torch.sin(2.0 * torch.pi * freq * time_axis)
            noise = torch.randn(self.signal_length, generator=generator) * self.noise_std
            signals.append(base + noise)
        return torch.stack(signals, dim=0)

    def _build_temp(self, generator: torch.Generator) -> torch.Tensor:
        base_grid = torch.linspace(0.0, 1.0, self.temp_grid_size)
        sequences = []
        for index in range(self.size):
            label = int(self.labels[index].item())
            trend = torch.linspace(0.0, 0.2 * (label + 1), self.temp_seq_len).unsqueeze(-1)
            grid = base_grid.unsqueeze(0).repeat(self.temp_seq_len, 1)
            noise = (
                torch.randn(
                    self.temp_seq_len,
                    self.temp_grid_size,
                    generator=generator,
                )
                * self.noise_std
            )
            sequences.append(grid + trend + noise)
        return torch.stack(sequences, dim=0)

    def _build_targets(
        self,
        labels: torch.Tensor,
        target_range: tuple[float, float],
        generator: torch.Generator,
    ) -> torch.Tensor:
        low, high = target_range
        scale = (labels.float() + 1.0) / max(float(self.num_classes), 1.0)
        base = low + scale * (high - low)
        noise = torch.randn(labels.shape[0], generator=generator) * self.noise_std
        return base + noise

    def label_distribution(self) -> dict[int, int]:
        counts = torch.bincount(self.labels, minlength=self.num_classes)
        return {index: int(count.item()) for index, count in enumerate(counts)}

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "bcg": self.bcg[index].to(dtype=torch.float32),
            "temp": self.temp[index].to(dtype=torch.float32),
            "label": int(self.labels[index].item()),
            "hr": float(self.hr[index].item()),
            "rr": float(self.rr[index].item()),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

