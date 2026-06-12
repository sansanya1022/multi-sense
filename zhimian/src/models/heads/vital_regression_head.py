from __future__ import annotations

import torch
from torch import nn


class VitalRegressionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 2) -> None:
        super().__init__()
        self.regressor = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.regressor(features)

