"""Run a one-step controller demo and a simple one-night simulation."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sleep_controller.controller import SleepController
from sleep_controller.schemas import PhysiologySample, UserBaseline, UserProfile
from sleep_controller.simulator import SleepSimulator
from sleep_controller.utils import load_json


def main() -> None:
    root = Path(__file__).resolve().parent
    profile_data = load_json(root / "sample_user_profile.json")
    baseline_data = load_json(root / "sample_user_baseline.json")

    profile = UserProfile(**profile_data)
    baseline = UserBaseline(**baseline_data)
    sample = PhysiologySample(
        timestamp=datetime.now().replace(microsecond=0),
        user_id=profile.user_id,
        hr_bpm=76.0,
        br_bpm=17.0,
        skin_temp_c=36.5,
        rmssd_ms=40.0,
        br_irregularity=0.12,
        stillness=0.55,
    )

    controller = SleepController()
    decision = controller.step(
        profile=profile,
        baseline=baseline,
        sample=sample,
        phase="pre_sleep",
        previous_action=None,
        elapsed_min=5.0,
    )

    print("=== Single Step Decision ===")
    print("state:", decision.state_snapshot.state)
    print("confidence:", round(decision.state_snapshot.confidence, 3))
    print("rationale:", decision.state_snapshot.rationale)
    print("anchor action:", decision.anchor_action)
    print("model delta:", decision.model_delta)
    print("final safe action:", decision.final_safe_action)
    print("safety_violated:", decision.safety_violated)

    simulator = SleepSimulator(seed=123)
    result = simulator.run_episode(
        profile=profile,
        baseline=baseline,
        controller=controller,
        duration_min=30,
        step_seconds=60,
    )

    print("\n=== Simulated Episode Outcome ===")
    print(result.episode_outcome)
    print("states:", [snapshot.state for snapshot in result.state_snapshots[:5]], "...")
    print("num_actions:", len(result.action_logs))
    print("avg_reward:", round(sum(result.rewards) / max(len(result.rewards), 1), 4))


if __name__ == "__main__":
    main()
