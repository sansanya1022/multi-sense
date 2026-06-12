"""Utility functions used across the sleep controller package."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-dependent fallback
    torch = None


def clip(value: float, low: float, high: float) -> float:
    """Clip a float value into the given range."""

    return max(low, min(high, value))


def pct_delta(current: float, baseline: float) -> float:
    """Return relative deviation from baseline."""

    if baseline == 0:
        return 0.0
    return (current - baseline) / baseline


def set_global_seed(seed: int) -> None:
    """Set all random seeds used by the MVP code."""

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: Any, path: str | Path) -> None:
    """Save dataclass or JSON-serializable object as JSON."""

    serializable = asdict(data) if is_dataclass(data) else data
    Path(path).write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
