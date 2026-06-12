from datetime import datetime

from sleep_controller.observation_builder import ObservationBuilder
from sleep_controller.personalized_engine import PersonalizedEngine
from sleep_controller.schemas import PhysiologySample, StateSnapshot, UserBaseline, UserProfile


def test_observation_shape_and_missing_stillness() -> None:
    profile = UserProfile(
        user_id="u_001",
        age=29,
        sex="female",
        stress_trait=0.7,
        temperature_sensitivity=0.5,
        aroma_sensitivity=0.4,
        sleep_schedule_type="regular",
    )
    baseline = UserBaseline(
        user_id="u_001",
        hr_bpm=68.0,
        br_bpm=15.0,
        skin_temp_c=36.4,
        rmssd_ms=42.0,
        stillness=0.78,
        avg_sleep_latency_min_7d=18.0,
        pathological_insomnia_nights_7d=0,
    )
    strategy = PersonalizedEngine().generate(profile, baseline)
    sample = PhysiologySample(
        timestamp=datetime.now(),
        user_id="u_001",
        hr_bpm=76.0,
        br_bpm=17.0,
        skin_temp_c=36.5,
        rmssd_ms=40.0,
        br_irregularity=None,
        stillness=None,
    )
    snapshot = StateSnapshot(
        timestamp=datetime.now(),
        user_id="u_001",
        phase="pre_sleep",
        state="S3",
        confidence=0.7,
        rationale="test",
    )
    observation = ObservationBuilder().build(
        profile=profile,
        baseline=baseline,
        strategy=strategy,
        sample=sample,
        state_snapshot=snapshot,
        elapsed_min=10.0,
    )
    assert observation.shape == (20,)

