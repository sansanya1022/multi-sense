from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.eval.report import write_text_report
from src.utils.config import parse_config


@dataclass
class StrategyEvaluationResult:
    metrics: dict[str, float]
    state_distribution: dict[str, int]
    report_markdown: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_optional_text_metrics(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    metrics: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        try:
            metrics[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return metrics


def _safe_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _score_latency(latency_min: float) -> float:
    return max(0.0, min(1.0, 1.0 - latency_min / 60.0))


def _score_depth(depth_value: float) -> float:
    return max(0.0, min(1.0, depth_value))


def _score_continuity(wake_count: int, micro_arousal_count: int) -> float:
    return max(0.0, min(1.0, 1.0 - (wake_count + micro_arousal_count) / 10.0))


def _score_subjective(subjective_morning_score: float) -> float:
    return max(0.0, min(1.0, subjective_morning_score / 5.0))


def _score_safety(safety_violation_rate: float) -> float:
    return max(0.0, min(1.0, 1.0 - safety_violation_rate))


def _score_smoothness(mean_light_delta: float, mean_aroma_delta: float, mean_temp_delta: float) -> float:
    normalized = (
        min(abs(mean_light_delta) / 1.0, 1.0)
        + min(abs(mean_aroma_delta) / 0.15, 1.0)
        + min(abs(mean_temp_delta) / 0.5, 1.0)
    ) / 3.0
    return max(0.0, min(1.0, 1.0 - normalized))


def compute_strategy_metrics(
    dataset_root: Path,
    training_metrics_path: Path | None = None,
) -> StrategyEvaluationResult:
    """
    Compute a strategy quality report from the observed dataset package.

    Inputs are read from the exact file structure observed under
    simulated_sleep_night_u001 and no extra format assumptions are made.
    """

    action_logs = _read_csv(dataset_root / "action_logs.csv")
    reward_logs = _read_csv(dataset_root / "reward_logs.csv")
    state_snapshots = _read_csv(dataset_root / "state_snapshots.csv")
    episode_rows = _read_csv(dataset_root / "episode_outcomes.csv")
    strategy_json = json.loads((dataset_root / "personalized_strategy.json").read_text(encoding="utf-8"))
    training_metrics = _read_optional_text_metrics(training_metrics_path)

    if len(episode_rows) != 1:
        raise ValueError("episode_outcomes.csv must contain exactly one row")
    episode = episode_rows[0]

    latency_min = _safe_float(episode["sleep_latency_min"])
    depth_value = _safe_float(
        episode["deep_sleep_ratio"],
        default=_safe_float(episode["deep_sleep_proxy"], default=0.5),
    )
    wake_count = int(float(episode["wake_count"]))
    micro_arousal_count = int(float(episode["micro_arousal_count"]))
    subjective_morning_score = _safe_float(episode["subjective_morning_score"])

    safety_violations = sum(1 for row in action_logs if row["safety_violated"].lower() == "true")
    safety_violation_rate = safety_violations / max(len(action_logs), 1)
    mean_light_delta = sum(abs(_safe_float(row["model_light_delta"])) for row in action_logs) / max(len(action_logs), 1)
    mean_aroma_delta = sum(abs(_safe_float(row["model_aroma_delta"])) for row in action_logs) / max(len(action_logs), 1)
    mean_temp_delta = sum(abs(_safe_float(row["model_temp_delta_c"])) for row in action_logs) / max(len(action_logs), 1)
    mean_dense_reward = sum(_safe_float(row["dense_reward"]) for row in reward_logs) / max(len(reward_logs), 1)
    final_terminal_reward = _safe_float(reward_logs[-1].get("terminal_reward"), default=0.0) if reward_logs else 0.0
    final_episode_reward = _safe_float(reward_logs[-1].get("episode_total_reward"), default=0.0) if reward_logs else 0.0

    state_distribution: dict[str, int] = {}
    for row in state_snapshots:
        state_distribution[row["state"]] = state_distribution.get(row["state"], 0) + 1

    latency_score = _score_latency(latency_min)
    depth_score = _score_depth(depth_value)
    continuity_score = _score_continuity(wake_count, micro_arousal_count)
    subjective_score = _score_subjective(subjective_morning_score)
    safety_score = _score_safety(safety_violation_rate)
    smoothness_score = _score_smoothness(mean_light_delta, mean_aroma_delta, mean_temp_delta)

    overall_strategy_score = (
        0.35 * latency_score
        + 0.20 * depth_score
        + 0.15 * continuity_score
        + 0.10 * subjective_score
        + 0.10 * safety_score
        + 0.10 * smoothness_score
    )

    metrics: dict[str, float] = {
        "sleep_latency_min": latency_min,
        "latency_score": latency_score,
        "deep_sleep_value": depth_value,
        "depth_score": depth_score,
        "wake_count": float(wake_count),
        "micro_arousal_count": float(micro_arousal_count),
        "continuity_score": continuity_score,
        "subjective_morning_score": subjective_morning_score,
        "subjective_score": subjective_score,
        "safety_violation_rate": safety_violation_rate,
        "safety_score": safety_score,
        "mean_model_light_delta_abs": mean_light_delta,
        "mean_model_aroma_delta_abs": mean_aroma_delta,
        "mean_model_temp_delta_abs": mean_temp_delta,
        "smoothness_score": smoothness_score,
        "mean_dense_reward": mean_dense_reward,
        "final_terminal_reward": final_terminal_reward,
        "final_episode_reward": final_episode_reward,
        "overall_strategy_score": overall_strategy_score,
    }

    for key, value in training_metrics.items():
        metrics[f"training/{key}"] = value

    markdown_lines = [
        "# Strategy Evaluation Report",
        "",
        f"- dataset_root: `{dataset_root}`",
        f"- age_band: `{strategy_json.get('age_band', 'unknown')}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    summary_keys = [
        "sleep_latency_min",
        "deep_sleep_value",
        "wake_count",
        "micro_arousal_count",
        "subjective_morning_score",
        "safety_violation_rate",
        "mean_dense_reward",
        "overall_strategy_score",
    ]
    for key in summary_keys:
        markdown_lines.append(f"| {key} | {metrics[key]:.4f} |")

    markdown_lines.extend(
        [
            "",
            "## State Distribution",
            "",
            "| State | Count |",
            "|---|---:|",
        ]
    )
    for state_name, count in sorted(state_distribution.items()):
        markdown_lines.append(f"| {state_name} | {count} |")

    markdown_lines.extend(
        [
            "",
            "## Action Smoothness",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| mean_model_light_delta_abs | {mean_light_delta:.4f} |",
            f"| mean_model_aroma_delta_abs | {mean_aroma_delta:.4f} |",
            f"| mean_model_temp_delta_abs | {mean_temp_delta:.4f} |",
            f"| smoothness_score | {smoothness_score:.4f} |",
        ]
    )
    if training_metrics:
        markdown_lines.extend(["", "## Training Metrics", "", "| Metric | Value |", "|---|---:|"])
        for key, value in sorted(training_metrics.items()):
            markdown_lines.append(f"| {key} | {value:.4f} |")

    return StrategyEvaluationResult(
        metrics=metrics,
        state_distribution=state_distribution,
        report_markdown="\n".join(markdown_lines),
    )


def write_strategy_report(result: StrategyEvaluationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text_report(result.metrics, output_dir / "strategy_metrics.txt")
    (output_dir / "strategy_report.md").write_text(result.report_markdown, encoding="utf-8")
    (output_dir / "strategy_metrics.json").write_text(
        json.dumps(
            {
                "metrics": result.metrics,
                "state_distribution": result.state_distribution,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    config = parse_config()
    dataset_root = Path(str(config.eval.dataset_root))
    training_metrics_path = (
        Path(str(config.eval.training_metrics_path))
        if config.eval.get("training_metrics_path")
        else None
    )
    result = compute_strategy_metrics(
        dataset_root=dataset_root,
        training_metrics_path=training_metrics_path,
    )
    output_dir = Path(str(config.eval.output_dir))
    write_strategy_report(result, output_dir)
    print(result.report_markdown)


if __name__ == "__main__":
    main()

