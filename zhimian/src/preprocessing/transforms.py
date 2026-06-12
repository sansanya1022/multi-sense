from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class StandardizeTensor:
    eps: float = 1e-6

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + self.eps)


class SampleTransform:
    def __init__(self, signal_standardize: bool, temp_standardize: bool) -> None:
        self.signal_standardize = signal_standardize
        self.temp_standardize = temp_standardize
        self.standardize = StandardizeTensor()

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        transformed = dict(sample)
        if self.signal_standardize:
            transformed["bcg"] = self.standardize(transformed["bcg"])
        if self.temp_standardize:
            transformed["temp"] = self.standardize(transformed["temp"])
        transformed["label"] = torch.tensor(transformed["label"], dtype=torch.long)
        transformed["hr"] = torch.tensor(transformed["hr"], dtype=torch.float32)
        transformed["rr"] = torch.tensor(transformed["rr"], dtype=torch.float32)
        return transformed

