from __future__ import annotations

from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import nn

from src.models.encoders.bcg_encoder import BCGEncoder
from src.models.encoders.temp_encoder import TempEncoder
from src.models.fusion.conditional_fusion import ConditionalFusion
from src.models.heads.sleep_stage_head import SleepStageHead
from src.models.heads.vital_regression_head import VitalRegressionHead
from src.models.uncertainty.quality_gate import QualityGate


@dataclass
class ModelOutput:
    logits: torch.Tensor
    vital_pred: torch.Tensor
    features: torch.Tensor


class SleepNet(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        dataset_cfg = config.dataset.source if "source" in config.dataset else config.dataset
        temp_input_dim = int(dataset_cfg.temp_seq_len) * int(dataset_cfg.temp_grid_size)
        bcg_output_dim = int(config.model.bcg_encoder.output_dim)
        temp_output_dim = int(config.model.temp_encoder.output_dim)
        fusion_output_dim = int(config.model.fusion.output_dim)
        self.bcg_encoder = BCGEncoder(
            hidden_channels=int(config.model.bcg_encoder.hidden_channels),
            output_dim=bcg_output_dim,
            num_blocks=int(config.model.bcg_encoder.get("num_blocks", 3)),
            kernel_size=int(config.model.bcg_encoder.get("kernel_size", 5)),
            dropout=float(config.model.bcg_encoder.get("dropout", 0.1)),
        )
        self.temp_encoder = TempEncoder(
            input_dim=temp_input_dim,
            hidden_dim=int(config.model.temp_encoder.hidden_dim),
            output_dim=temp_output_dim,
            dropout=float(config.model.temp_encoder.get("dropout", 0.1)),
        )
        self.fusion = ConditionalFusion(
            bcg_dim=bcg_output_dim,
            temp_dim=temp_output_dim,
            output_dim=fusion_output_dim,
            hidden_dim=int(config.model.fusion.get("hidden_dim", fusion_output_dim)),
            dropout=float(config.model.fusion.get("dropout", 0.1)),
        )
        self.quality_gate = (
            QualityGate(fusion_output_dim)
            if bool(config.model.uncertainty.enabled)
            else nn.Identity()
        )
        self.stage_head = SleepStageHead(
            input_dim=fusion_output_dim,
            num_classes=int(config.model.heads.num_classes),
        )
        self.vital_head = VitalRegressionHead(input_dim=fusion_output_dim, output_dim=2)

    def forward(self, bcg: torch.Tensor, temp: torch.Tensor) -> ModelOutput:
        bcg_features = self.bcg_encoder(bcg)
        temp_features = self.temp_encoder(temp)
        fused = self.fusion(bcg_features, temp_features)
        gated = self.quality_gate(fused)
        return ModelOutput(
            logits=self.stage_head(gated),
            vital_pred=self.vital_head(gated),
            features=gated,
        )
