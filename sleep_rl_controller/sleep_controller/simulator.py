"""Simple simulator for validating the personalized sleep control loop."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from sleep_controller.controller import SleepController
from sleep_controller.reward import RewardCalculator
from sleep_controller.schemas import (
    Action,
    EpisodeOutcome,
    PhysiologySample,
    SimulationResult,
    UserBaseline,
    UserProfile,
)
from sleep_controller.utils import clip, set_global_seed


class SleepSimulator:
    """Simulate a pre-sleep episode with controller-in-the-loop."""

    def __init__(self, seed: int = 42, reward_calculator: RewardCalculator | None = None) -> None:
        self.seed = seed
        set_global_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.reward_calculator = reward_calculator or RewardCalculator()

    def run_episode(
        self,
        profile: UserProfile,
        baseline: UserBaseline,
        controller: SleepController | None = None,
        duration_min: int = 45,
        step_seconds: int = 60,
    ) -> SimulationResult:
        """Run one simple simulated episode."""

        controller = controller or SleepController()
        steps = max(duration_min * 60 // step_seconds, 1)
        current_time = datetime.now().replace(microsecond=0)
        phase = "pre_sleep"
        previous_action = Action(light_lux=2.5, aroma_level=0.0, temp_delta_c=0.0)
        state = {
            "hr": baseline.hr_bpm * (1.08 + 0.05 * profile.stress_trait),
            "br": baseline.br_bpm * (1.06 + 0.04 * profile.stress_trait),
            "temp": baseline.skin_temp_c + 0.15,
            "stillness": 0.45 + 0.15 * (1.0 - profile.stress_trait),
            "br_irregularity": 0.12,
        }
        result = SimulationResult()
        prev_sample: PhysiologySample | None = None
        stable_counter = 0
        sleep_latency_min = float(duration_min)

        for step_index in range(int(steps)):
            elapsed_min = float(step_index * step_seconds / 60.0)
            if elapsed_min >= 15.0:
                phase = "induction"
            sample = PhysiologySample(
                timestamp=current_time + timedelta(seconds=step_index * step_seconds),
                user_id=profile.user_id,
                hr_bpm=float(state["hr"]),
                br_bpm=float(state["br"]),
                skin_temp_c=float(state["temp"]),
                rmssd_ms=baseline.rmssd_ms,
                br_irregularity=float(state["br_irregularity"]),
                stillness=float(state["stillness"]),
            )
            decision = controller.step(
                profile=profile,
                baseline=baseline,
                sample=sample,
                phase=phase,
                previous_action=previous_action,
                elapsed_min=elapsed_min,
            )
            result.decisions.append(decision)
            result.state_snapshots.append(decision.state_snapshot)
            result.action_logs.append(decision.action_log)

            if prev_sample is not None:
                dense_reward = self.reward_calculator.compute_dense_reward(
                    prev_sample=prev_sample,
                    current_sample=sample,
                    previous_action=previous_action,
                    current_action=decision.final_safe_action,
                    strategy=decision.strategy,
                    safety_violated=decision.safety_violated,
                )
                result.rewards.append(dense_reward)
            else:
                result.rewards.append(0.0)

            state = self._transition_state(
                state=state,
                action=decision.final_safe_action,
                strategy=decision.strategy,
            )

            within_hr = abs(state["hr"] - decision.strategy.targets["expected_sleep_hr_bpm"]) <= 2.0
            within_br = abs(state["br"] - decision.strategy.targets["expected_sleep_br_bpm"]) <= 1.0
            calm_stillness = state["stillness"] >= 0.72
            if within_hr and within_br and calm_stillness:
                stable_counter += 1
                if stable_counter >= 3 and sleep_latency_min == float(duration_min):
                    sleep_latency_min = elapsed_min
            else:
                stable_counter = 0

            previous_action = decision.final_safe_action
            prev_sample = sample

        deep_sleep_proxy = clip(
            0.4
            + 0.25 * (1.0 - abs(state["hr"] - baseline.hr_bpm * 0.92) / max(baseline.hr_bpm, 1.0))
            + 0.20 * state["stillness"],
            0.0,
            1.0,
        )
        wake_count = int(self.rng.integers(0, 2 if deep_sleep_proxy > 0.65 else 3))
        micro_arousal_count = int(self.rng.integers(1, 3 if deep_sleep_proxy > 0.65 else 5))
        subjective = clip(2.5 + 2.0 * deep_sleep_proxy - 0.03 * sleep_latency_min, 0.0, 5.0)
        outcome = EpisodeOutcome(
            date=current_time.date(),
            user_id=profile.user_id,
            sleep_latency_min=float(sleep_latency_min),
            deep_sleep_ratio=None,
            deep_sleep_proxy=float(deep_sleep_proxy),
            wake_count=wake_count,
            micro_arousal_count=micro_arousal_count,
            subjective_morning_score=float(subjective),
        )
        result.episode_outcome = outcome
        return result

    def _transition_state(
        self,
        state: dict[str, float],
        action: Action,
        strategy: object,
    ) -> dict[str, float]:
        """Apply a simple interpretable transition based on safe action."""

        light_calming = clip((3.0 - action.light_lux) / 3.0, 0.0, 1.0)
        aroma_calming = clip(action.aroma_level, 0.0, 1.0)
        cooling = clip(-action.temp_delta_c / 2.0, 0.0, 1.0)
        relaxation = 0.45 * light_calming + 0.35 * aroma_calming + 0.20 * cooling

        next_hr = state["hr"] - 0.6 * relaxation + self.rng.normal(0.0, 0.15)
        next_br = state["br"] - 0.25 * relaxation + self.rng.normal(0.0, 0.08)
        next_temp = state["temp"] + 0.1 * action.temp_delta_c + self.rng.normal(0.0, 0.03)
        next_stillness = clip(state["stillness"] + 0.05 * relaxation + self.rng.normal(0.0, 0.02), 0.0, 1.0)
        next_irregularity = clip(state["br_irregularity"] - 0.03 * relaxation + self.rng.normal(0.0, 0.01), 0.0, 1.0)
        return {
            "hr": float(max(next_hr, 40.0)),
            "br": float(max(next_br, 6.0)),
            "temp": float(next_temp),
            "stillness": float(next_stillness),
            "br_irregularity": float(next_irregularity),
        }

