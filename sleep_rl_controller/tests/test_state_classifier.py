from datetime import datetime

from sleep_controller.personalized_engine import PersonalizedEngine
from sleep_controller.schemas import PhysiologySample, UserBaseline, UserProfile
from sleep_controller.state_classifier import StateClassifier


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


def test_s6_has_highest_priority() -> None:
    profile = make_profile()
    baseline = make_baseline(latency=45.0, nights=5)
    strategy = PersonalizedEngine().generate(profile, baseline)
    sample = PhysiologySample(
        timestamp=datetime.now(),
        user_id="u_001",
        hr_bpm=80.0,
        br_bpm=18.0,
        skin_temp_c=36.5,
        rmssd_ms=40.0,
        br_irregularity=0.20,
        stillness=0.40,
    )
    snapshot = StateClassifier().classify(profile, baseline, strategy, sample, "pre_sleep")
    assert snapshot.state == "S6"


def test_high_hr_br_classifies_to_s1_or_s2() -> None:
    profile = make_profile()
    baseline = make_baseline()
    strategy = PersonalizedEngine().generate(profile, baseline)
    sample = PhysiologySample(
        timestamp=datetime.now(),
        user_id="u_001",
        hr_bpm=78.0,
        br_bpm=18.0,
        skin_temp_c=36.5,
        rmssd_ms=40.0,
        br_irregularity=0.10,
        stillness=0.50,
    )
    snapshot = StateClassifier().classify(profile, baseline, strategy, sample, "pre_sleep")
    assert snapshot.state in {"S1", "S2"}

