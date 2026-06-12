from __future__ import annotations

from typing import Any

import torch


def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == labels).float().mean().item())


def mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(predictions - targets)).item())


def aggregate_metrics(metric_list: list[dict[str, float]]) -> dict[str, float]:
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    return {key: sum(metric[key] for metric in metric_list) / len(metric_list) for key in keys}


def tensor_to_python(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

