from __future__ import annotations

from pathlib import Path

from src.eval.strategy_report import compute_strategy_metrics


def test_strategy_report_generates_expected_metrics() -> None:
    result = compute_strategy_metrics(
        dataset_root=Path("data/raw/simulated_sleep_night_u001"),
        training_metrics_path=None,
    )
    assert "overall_strategy_score" in result.metrics
    assert "safety_violation_rate" in result.metrics
    assert result.metrics["sleep_latency_min"] == 37.0
    assert "S1" in result.state_distribution
