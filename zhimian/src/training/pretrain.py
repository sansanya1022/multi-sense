from __future__ import annotations

from src.datasets.registry import build_dataloaders
from src.training.common import (
    build_model_and_optimizer,
    finalize_run,
    log_metrics,
    maybe_save_best,
    prepare_run,
    run_epoch,
)
from src.utils.config import parse_config


def main() -> None:
    config = parse_config()
    artifacts = prepare_run(config)
    train_loader, val_loader = build_dataloaders(config)
    model, optimizer, device = build_model_and_optimizer(config)
    final_metrics: dict[str, float] = {}
    for epoch in range(1, int(config.optimization.epochs) + 1):
        train_metrics = run_epoch(config, model, train_loader, optimizer, device)
        val_metrics = run_epoch(config, model, val_loader, None, device)
        log_metrics(artifacts.writer, "train", epoch, train_metrics)
        log_metrics(artifacts.writer, "val", epoch, val_metrics)
        artifacts = maybe_save_best(config, artifacts, model, optimizer, epoch, val_metrics)
        final_metrics = {f"train/{k}": v for k, v in train_metrics.items()} | {
            f"val/{k}": v for k, v in val_metrics.items()
        }
    finalize_run(artifacts, final_metrics)


if __name__ == "__main__":
    main()

