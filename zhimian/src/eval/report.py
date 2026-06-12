from __future__ import annotations

from pathlib import Path


def write_text_report(metrics: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {value:.6f}" for key, value in sorted(metrics.items())]
    output_path.write_text("\n".join(lines), encoding="utf-8")

