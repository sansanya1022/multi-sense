from __future__ import annotations

import torch
from torch import nn


class BCGEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        output_dim: int,
        num_blocks: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_blocks < 2:
            raise ValueError("num_blocks must be at least 2")
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        in_channels = 1
        for block_index in range(num_blocks):
            out_channels = hidden_channels if block_index < num_blocks - 1 else hidden_channels * 2
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels
        layers.extend(
            [
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(in_channels, output_dim),
            ]
        )
        self.network = nn.Sequential(*layers)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.network(signal.unsqueeze(1))
