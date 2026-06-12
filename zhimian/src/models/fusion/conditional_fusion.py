from __future__ import annotations

import torch
from torch import nn


class ConditionalFusion(nn.Module):
    def __init__(
        self,
        bcg_dim: int,
        temp_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or output_dim
        self.fusion = nn.Sequential(
            nn.Linear(bcg_dim + temp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, bcg_features: torch.Tensor, temp_features: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([bcg_features, temp_features], dim=-1))
