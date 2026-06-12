from __future__ import annotations

from itertools import cycle
from typing import Any

import torch
from torch import nn

from src.datasets.registry import build_adaptation_dataloaders
from src.models.adaptation.coral import coral_loss
from src.models.adaptation.dann import DomainClassifier
from src.training.common import (
    build_model_and_optimizer,
    finalize_run,
    log_metrics,
    maybe_save_best,
    move_batch_to_device,
    prepare_run,
    run_epoch,
)
from src.utils.config import parse_config


def run_adaptation_epoch(
    config: Any,
    model: Any,
    domain_classifier: DomainClassifier,
    source_loader: Any,
    target_loader: Any,
    optimizer: Any,
    device: torch.device,
) -> dict[str, float]:
    model.train(True)
    domain_classifier.train(True)
    ce_loss = nn.CrossEntropyLoss()
    source_iter = cycle(source_loader)
    target_iter = cycle(target_loader)
    metric_total = {
        "cls_loss": 0.0,
        "hr_loss": 0.0,
        "rr_loss": 0.0,
        "domain_loss": 0.0,
        "coral_loss": 0.0,
        "total_loss": 0.0,
    }
    num_steps = min(len(source_loader), len(target_loader))
    for _ in range(num_steps):
        source_batch = move_batch_to_device(next(source_iter), device)
        target_batch = move_batch_to_device(next(target_iter), device)
        source_output = model(source_batch["bcg"], source_batch["temp"])
        target_output = model(target_batch["bcg"], target_batch["temp"])

        cls_loss = ce_loss(source_output.logits, source_batch["label"])
        hr_loss = nn.MSELoss()(source_output.vital_pred[:, 0], source_batch["hr"])
        rr_loss = nn.MSELoss()(source_output.vital_pred[:, 1], source_batch["rr"])
        source_domain_logits = domain_classifier(
            source_output.features,
            lambda_=float(config.model.adaptation.grl_lambda),
        )
        target_domain_logits = domain_classifier(
            target_output.features,
            lambda_=float(config.model.adaptation.grl_lambda),
        )
        domain_labels_source = torch.zeros(source_domain_logits.shape[0], dtype=torch.long, device=device)
        domain_labels_target = torch.ones(target_domain_logits.shape[0], dtype=torch.long, device=device)
        domain_loss = ce_loss(source_domain_logits, domain_labels_source) + ce_loss(
            target_domain_logits,
            domain_labels_target,
        )
        feature_coral = coral_loss(source_output.features, target_output.features)
        total_loss = (
            float(config.optimization.cls_loss_weight) * cls_loss
            + float(config.optimization.hr_loss_weight) * hr_loss
            + float(config.optimization.rr_loss_weight) * rr_loss
            + float(config.optimization.domain_loss_weight) * domain_loss
            + 0.1 * feature_coral
        )
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(domain_classifier.parameters()),
            max_norm=float(config.optimization.grad_clip_norm),
        )
        optimizer.step()
        metric_total["cls_loss"] += float(cls_loss.item())
        metric_total["hr_loss"] += float(hr_loss.item())
        metric_total["rr_loss"] += float(rr_loss.item())
        metric_total["domain_loss"] += float(domain_loss.item())
        metric_total["coral_loss"] += float(feature_coral.item())
        metric_total["total_loss"] += float(total_loss.item())
    return {key: value / num_steps for key, value in metric_total.items()}


def main() -> None:
    config = parse_config()
    artifacts = prepare_run(config)
    source_loader, target_loader, val_loader = build_adaptation_dataloaders(config)
    model, optimizer, device = build_model_and_optimizer(config)
    domain_classifier = DomainClassifier(
        input_dim=int(config.model.fusion.output_dim),
        hidden_dim=int(config.model.adaptation.domain_hidden_dim),
    ).to(device)
    optimizer.add_param_group({"params": domain_classifier.parameters()})
    final_metrics: dict[str, float] = {}
    for epoch in range(1, int(config.optimization.epochs) + 1):
        train_metrics = run_adaptation_epoch(
            config=config,
            model=model,
            domain_classifier=domain_classifier,
            source_loader=source_loader,
            target_loader=target_loader,
            optimizer=optimizer,
            device=device,
        )
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

