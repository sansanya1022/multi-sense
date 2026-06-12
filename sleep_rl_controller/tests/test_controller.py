from datetime import datetime

from sleep_controller.controller import SleepController
from sleep_controller.schemas import PhysiologySample, UserBaseline, UserProfile


def make_profile() -> UserProfile:
    return UserProfile(
        user_id="u_001",
        age=29,
        sex="female",
        stress_trait=0.7,
        temperature_sensitivity=0.5,
        aroma_sensitivity=0.4,
        sleep_schedule_type="regular",
    )


def make_baseline(latency: float = 18.0, nights: int = 0) -> UserBaseline:
    return UserBaseline(
        user_id="u_001",
        hr_bpm=68.0,
        br_bpm=15.0,
        skin_temp_c=36.4,
        rmssd_ms=42.0,
        stillness=0.78,
        avg_sleep_latency_min_7d=latency,
        pathological_insomnia_nights_7d=nights,
    )


def test_controller_step_returns_action_log() -> None:
    controller = SleepController()
    profile = make_profile()
    baseline = make_baseline()
    sample = PhysiologySample(
        timestamp=datetime.now(),
        user_id="u_001",
        hr_bpm=76.0,
        br_bpm=17.0,
        skin_temp_c=36.5,
        rmssd_ms=40.0,
        br_irregularity=0.12,
        stillness=0.55,
    )
    decision = controller.step(profile, baseline, sample, "pre_sleep")
    assert decision.action_log.user_id == "u_001"


def test_s6_forces_zero_model_delta() -> None:
    controller = SleepController()
    profile = make_profile()
    baseline = make_baseline(latency=45.0, nights=5)
    sample = PhysiologySample(
        timestamp=datetime.now(),
        user_id="u_001",
        hr_bpm=80.0,
        br_bpm=18.0,
        skin_temp_c=36.5,
        rmssd_ms=40.0,
        br_irregularity=0.12,
        stillness=0.55,
    )
    decision = controller.step(profile, baseline, sample, "pre_sleep")
    assert decision.state_snapshot.state == "S6"
    assert decision.model_delta.light_lux == 0.0
    assert decision.model_delta.aroma_level == 0.0
    assert decision.model_delta.temp_delta_c == 0.0

