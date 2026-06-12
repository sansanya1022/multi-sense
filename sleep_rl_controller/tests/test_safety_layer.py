from sleep_controller.personalized_engine import PersonalizedEngine
from sleep_controller.safety_layer import SafetyLayer
from sleep_controller.schemas import Action, UserBaseline, UserProfile


def make_strategy():
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
    return PersonalizedEngine().generate(profile, baseline)


def test_safety_layer_clamps_out_of_bound_action() -> None:
    strategy = make_strategy()
    candidate = Action(light_lux=10.0, aroma_level=2.0, temp_delta_c=-8.0)
    safe_action, violated = SafetyLayer().apply(candidate, None, strategy, "S1")
    assert safe_action.light_lux <= strategy.safety_bounds["light_lux_max"]
    assert safe_action.aroma_level <= strategy.safety_bounds["aroma_level_max"]
    assert safe_action.temp_delta_c >= strategy.safety_bounds["temp_delta_min_c"]
    assert violated is True


def test_safety_layer_limits_step_change() -> None:
    strategy = make_strategy()
    previous = Action(light_lux=1.0, aroma_level=0.1, temp_delta_c=0.0)
    candidate = Action(light_lux=4.0, aroma_level=0.8, temp_delta_c=-2.0)
    safe_action, violated = SafetyLayer().apply(candidate, previous, strategy, "S1")
    assert abs(safe_action.light_lux - previous.light_lux) <= strategy.safety_bounds["max_light_step_lux"]
    assert abs(safe_action.aroma_level - previous.aroma_level) <= strategy.safety_bounds["max_aroma_step"]
    assert abs(safe_action.temp_delta_c - previous.temp_delta_c) <= strategy.safety_bounds["max_temp_step_c"]
    assert violated is True

