from sleep_controller.personalized_engine import PersonalizedEngine
from sleep_controller.schemas import UserBaseline, UserProfile


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


def test_personalized_strategy_contains_required_groups() -> None:
    engine = PersonalizedEngine()
    strategy = engine.generate(make_profile(), make_baseline())
    assert "s1_hr_pct" in strategy.thresholds
    assert "expected_sleep_hr_bpm" in strategy.targets
    assert "light_scale" in strategy.anchor_scales
    assert "light_lux_max" in strategy.safety_bounds

