from datetime import date, datetime, timedelta

from sleep_controller.personalized_engine import PersonalizedEngine
from sleep_controller.reward import RewardCalculator
from sleep_controller.schemas import Action, EpisodeOutcome, PhysiologySample, UserBaseline, UserProfile


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


def make_baseline() -> UserBaseline:
    return UserBaseline(
        user_id="u_001",
        hr_bpm=68.0,
        br_bpm=15.0,
        skin_temp_c=36.4,
        rmssd_ms=42.0,
        stillness=0.78,
        avg_sleep_latency_min_7d=18.0,
        pathological_insomnia_nights_7d=0,
    )


def test_reward_functions_return_float() -> None:
    profile = make_profile()
    baseline = make_baseline()
    strategy = PersonalizedEngine().generate(profile, baseline)
    reward_calculator = RewardCalculator()
    prev_sample = PhysiologySample(
        timestamp=datetime.now(),
        user_id="u_001",
        hr_bpm=76.0,
        br_bpm=17.0,
        skin_temp_c=36.6,
        rmssd_ms=40.0,
        br_irregularity=0.12,
        stillness=0.55,
    )
    current_sample = PhysiologySample(
        timestamp=datetime.now() + timedelta(minutes=1),
        user_id="u_001",
        hr_bpm=73.0,
        br_bpm=16.0,
        skin_temp_c=36.4,
        rmssd_ms=40.0,
        br_irregularity=0.10,
        stillness=0.65,
    )
    dense_reward = reward_calculator.compute_dense_reward(
        prev_sample=prev_sample,
        current_sample=current_sample,
        previous_action=Action(light_lux=2.0, aroma_level=0.1, temp_delta_c=0.0),
        current_action=Action(light_lux=1.5, aroma_level=0.2, temp_delta_c=-0.5),
        strategy=strategy,
        safety_violated=False,
    )
    terminal_reward = reward_calculator.compute_terminal_reward(
        EpisodeOutcome(
            date=date.today(),
            user_id="u_001",
            sleep_latency_min=18.0,
            deep_sleep_ratio=None,
            deep_sleep_proxy=0.72,
            wake_count=1,
            micro_arousal_count=2,
            subjective_morning_score=4.0,
        )
    )
    assert isinstance(dense_reward, float)
    assert isinstance(terminal_reward, float)

