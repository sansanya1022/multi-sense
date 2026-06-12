from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.eval.metrics import aggregate_metrics, classification_accuracy, mean_absolute_error
from src.eval.report import write_text_report
from src.models.network import SleepNet
from src.utils.checkpoint import save_checkpoint
from src.utils.config import save_config
from src.utils.seed import set_global_seed


@dataclass
class RunArtifacts:
    run_dir: Path
    writer: SummaryWriter
    best_checkpoint_path: Path
    best_config_path: Path
    best_metric: float


def prepare_run(config: DictConfig) -> RunArtifacts:
    set_global_seed(int(config.run.seed), deterministic=bool(config.run.deterministic))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(str(config.run.output_root))
        / str(config.run.stage)
        / f"{config.run.experiment_name}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    best_checkpoint_path = run_dir / str(config.checkpoint.save_best_filename)
    best_config_path = run_dir / str(config.checkpoint.save_config_filename)
    save_config(config, best_config_path)
    mode = str(config.checkpoint.mode)
    best_metric = float("inf") if mode == "min" else float("-inf")
    return RunArtifacts(
        run_dir=run_dir,
        writer=writer,
        best_checkpoint_path=best_checkpoint_path,
        best_config_path=best_config_path,
        best_metric=best_metric,
    )


def build_model_and_optimizer(config: DictConfig) -> tuple[SleepNet, AdamW, torch.device]:
    device = torch.device(str(config.run.device))
    model = SleepNet(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.optimization.learning_rate),
        weight_decay=float(config.optimization.weight_decay),
    )
    return model, optimizer, device


def compute_supervised_losses(
    config: DictConfig,
    logits: torch.Tensor,
    vital_pred: torch.Tensor,
    labels: torch.Tensor,
    hr_targets: torch.Tensor,
    rr_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    cls_loss = nn.CrossEntropyLoss()(logits, labels)
    hr_loss = nn.MSELoss()(vital_pred[:, 0], hr_targets)
    rr_loss = nn.MSELoss()(vital_pred[:, 1], rr_targets)
    total_loss = (
        float(config.optimization.cls_loss_weight) * cls_loss
        + float(config.optimization.hr_loss_weight) * hr_loss
        + float(config.optimization.rr_loss_weight) * rr_loss
    )
    return total_loss, {
        "cls_loss": float(cls_loss.item()),
        "hr_loss": float(hr_loss.item()),
        "rr_loss": float(rr_loss.item()),
        "total_loss": float(total_loss.item()),
    }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_epoch(
    config: DictConfig,
    model: SleepNet,
    loader: DataLoader[Any],
    optimizer: AdamW | None,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics: list[dict[str, float]] = []
    for batch in loader:
        batch_on_device = move_batch_to_device(batch, device)
        output = model(batch_on_device["bcg"], batch_on_device["temp"])
        loss, loss_metrics = compute_supervised_losses(
            config,
            logits=output.logits,
            vital_pred=output.vital_pred,
            labels=batch_on_device["label"],
            hr_targets=batch_on_device["hr"],
            rr_targets=batch_on_device["rr"],
        )
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(config.optimization.grad_clip_norm),
            )
            optimizer.step()
        metrics.append(
            {
                **loss_metrics,
                "accuracy": classification_accuracy(output.logits, batch_on_device["label"]),
                "hr_mae": mean_absolute_error(output.vital_pred[:, 0], batch_on_device["hr"]),
                "rr_mae": mean_absolute_error(output.vital_pred[:, 1], batch_on_device["rr"]),
            }
        )
    return aggregate_metrics(metrics)


def log_metrics(writer: SummaryWriter, split: str, epoch: int, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        writer.add_scalar(f"{split}/{key}", value, epoch)


def maybe_save_best(
    config: DictConfig,
    artifacts: RunArtifacts,
    model: SleepNet,
    optimizer: AdamW,
    epoch: int,
    metrics: dict[str, float],
) -> RunArtifacts:
    monitor_key = str(config.checkpoint.monitor).split("/", maxsplit=1)[-1]
    current_metric = metrics[monitor_key]
    is_better = (
        current_metric < artifacts.best_metric
        if str(config.checkpoint.mode) == "min"
        else current_metric > artifacts.best_metric
    )
    if is_better:
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": config,
            },
            artifacts.best_checkpoint_path,
        )
        save_config(config, artifacts.best_config_path)
        artifacts.best_metric = current_metric
    return artifacts


def finalize_run(artifacts: RunArtifacts, final_metrics: dict[str, float]) -> None:
    write_text_report(final_metrics, artifacts.run_dir / "metrics.txt")
    artifacts.writer.close()

