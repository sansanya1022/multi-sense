from __future__ import annotations

import torch


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.T @ source_centered / max(source.shape[0] - 1, 1)
    target_cov = target_centered.T @ target_centered / max(target.shape[0] - 1, 1)
    return torch.mean((source_cov - target_cov) ** 2)

