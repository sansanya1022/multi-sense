"""Observation vector builder for the tiny edge policy."""

from __future__ import annotations

from typing import Final

import numpy as np

from sleep_controller.schemas import (
    PersonalizedStrategy,
    PhysiologySample,
    StateSnapshot,
    UserBaseline,
    UserProfile,
)
from sleep_controller.utils import clip, pct_delta


class ObservationBuilder:
    """Build a fixed-length normalized observation for the tiny policy."""

    FEATURE_NAMES: Final[list[str]] = [
        "current_hr_norm",
        "current_br_norm",
        "current_temp_norm",
        "hr_vs_baseline_pct",
        "br_vs_baseline_pct",
        "temp_vs_baseline_delta",
        "hr_vs_target_pct",
        "br_vs_target_pct",
        "temp_vs_target_delta",
        "br_irregularity",
        "stillness",
        "state_one_hot_S1",
        "state_one_hot_S2",
        "state_one_hot_S3",
        "state_one_hot_S4",
        "state_one_hot_S5",
        "state_one_hot_S6",
        "state_confidence",
        "elapsed_min_norm",
        "age_norm",
    ]

    @property
    def feature_names(self) -> list[str]:
        """Return observation feature names."""

        return list(self.FEATURE_NAMES)

    def build(
        self,
        profile: UserProfile,
        baseline: UserBaseline,
        strategy: PersonalizedStrategy,
        sample: PhysiologySample,
        state_snapshot: StateSnapshot,
        elapsed_min: float,
    ) -> np.ndarray:
        """Build the 20-dim observation vector."""

        stillness = (
            sample.stillness
            if sample.stillness is not None
            else baseline.stillness
            if baseline.stillness is not None
            else 0.5
        )
        state_order = ["S1", "S2", "S3", "S4", "S5", "S6"]
        one_hot = [1.0 if state_snapshot.state == state_name else 0.0 for state_name in state_order]

        observation = np.array(
            [
                clip(sample.hr_bpm / 100.0, 0.0, 1.5),
                clip(sample.br_bpm / 30.0, 0.0, 1.5),
                clip((sample.skin_temp_c - 34.0) / 4.0, 0.0, 1.5),
                clip(pct_delta(sample.hr_bpm, baseline.hr_bpm), -1.0, 1.0),
                clip(pct_delta(sample.br_bpm, baseline.br_bpm), -1.0, 1.0),
                clip(sample.skin_temp_c - baseline.skin_temp_c, -3.0, 3.0),
                clip(
                    pct_delta(
                        sample.hr_bpm,
                        strategy.targets["expected_sleep_hr_bpm"],
                    ),
                    -1.0,
                    1.0,
                ),
                clip(
                    pct_delta(
                        sample.br_bpm,
                        strategy.targets["expected_sleep_br_bpm"],
                    ),
                    -1.0,
                    1.0,
                ),
                clip(
                    sample.skin_temp_c - strategy.targets["preferred_temp_c"],
                    -3.0,
                    3.0,
                ),
                clip(sample.br_irregularity or 0.0, 0.0, 1.0),
                clip(stillness, 0.0, 1.0),
                *one_hot,
                clip(state_snapshot.confidence, 0.0, 1.0),
                clip(elapsed_min / 120.0, 0.0, 1.0),
                clip(profile.age / 100.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        if observation.shape != (20,):
            raise ValueError(f"Observation must have shape (20,), got {observation.shape}")
        return observation

