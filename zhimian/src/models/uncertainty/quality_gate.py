from __future__ import annotations

import torch
from torch import nn


class QualityGate(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Sigmoid())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.gate(features)

