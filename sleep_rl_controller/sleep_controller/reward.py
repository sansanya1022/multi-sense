"""Dense and terminal reward definitions for simulator-side learning."""

from __future__ import annotations

from sleep_controller.schemas import Action, EpisodeOutcome, PersonalizedStrategy, PhysiologySample
from sleep_controller.utils import clip


class RewardCalculator:
    """Compute dense reward and terminal outcome reward."""

    def __init__(
        self,
        w1: float = 0.30,
        w2: float = 0.25,
        w3: float = 0.15,
        w4: float = 0.10,
        w5: float = 0.10,
        w6: float = 0.10,
        k1: float = 0.40,
        k2: float = 0.30,
        k3: float = 0.20,
        k4: float = 0.10,
    ) -> None:
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4

    def compute_dense_reward(
        self,
        prev_sample: PhysiologySample,
        current_sample: PhysiologySample,
        previous_action: Action,
        current_action: Action,
        strategy: PersonalizedStrategy,
        safety_violated: bool,
    ) -> float:
        """Compute step-wise dense reward from physiology improvement and action smoothness."""

        target_hr = strategy.targets["expected_sleep_hr_bpm"]
        target_br = strategy.targets["expected_sleep_br_bpm"]
        target_temp = strategy.targets["preferred_temp_c"]

        prev_hr_error = abs(prev_sample.hr_bpm - target_hr)
        cur_hr_error = abs(current_sample.hr_bpm - target_hr)
        prev_br_error = abs(prev_sample.br_bpm - target_br)
        cur_br_error = abs(current_sample.br_bpm - target_br)
        prev_temp_error = abs(prev_sample.skin_temp_c - target_temp)
        cur_temp_error = abs(current_sample.skin_temp_c - target_temp)

        hr_relax_gain = clip((prev_hr_error - cur_hr_error) / max(target_hr, 1.0), -1.0, 1.0)
        br_regular_gain = clip((prev_br_error - cur_br_error) / max(target_br, 1.0), -1.0, 1.0)
        temp_comfort_gain = clip((prev_temp_error - cur_temp_error) / 3.0, -1.0, 1.0)

        prev_stillness = prev_sample.stillness if prev_sample.stillness is not None else 0.5
        cur_stillness = current_sample.stillness if current_sample.stillness is not None else 0.5
        stillness_gain = clip(cur_stillness - prev_stillness, -1.0, 1.0)

        bounds = strategy.safety_bounds
        action_jitter_cost = (
            abs(current_action.light_lux - previous_action.light_lux)
            / max(bounds["max_light_step_lux"], 1e-6)
            + abs(current_action.aroma_level - previous_action.aroma_level)
            / max(bounds["max_aroma_step"], 1e-6)
            + abs(current_action.temp_delta_c - previous_action.temp_delta_c)
            / max(bounds["max_temp_step_c"], 1e-6)
        ) / 3.0

        safety_violation_cost = 1.0 if safety_violated else 0.0

        reward = (
            self.w1 * hr_relax_gain
            + self.w2 * br_regular_gain
            + self.w3 * temp_comfort_gain
            + self.w4 * stillness_gain
            - self.w5 * action_jitter_cost
            - self.w6 * safety_violation_cost
        )
        return float(reward)

    def compute_terminal_reward(self, outcome: EpisodeOutcome) -> float:
        """Compute terminal reward from episode outcome summary."""

        latency_score = clip(1.0 - outcome.sleep_latency_min / 60.0, 0.0, 1.0)
        depth_value = outcome.deep_sleep_ratio
        if depth_value is None:
            depth_value = outcome.deep_sleep_proxy if outcome.deep_sleep_proxy is not None else 0.5
        depth_score = clip(depth_value, 0.0, 1.0)
        continuity_penalty = (outcome.wake_count + outcome.micro_arousal_count) / 10.0
        continuity_score = clip(1.0 - continuity_penalty, 0.0, 1.0)
        subjective_score = clip(outcome.subjective_morning_score / 5.0, 0.0, 1.0)
        reward = (
            self.k1 * latency_score
            + self.k2 * depth_score
            + self.k3 * continuity_score
            + self.k4 * subjective_score
        )
        return float(reward)

