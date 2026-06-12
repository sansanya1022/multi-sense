"""Safety layer with absolute clamp, rate limiting, and S6 conservative mode."""

from __future__ import annotations

from sleep_controller.schemas import Action, PersonalizedStrategy
from sleep_controller.utils import clip


class SafetyLayer:
    """Enforce action limits before sending actions to actuators."""

    def apply(
        self,
        candidate_action: Action,
        previous_action: Action | None,
        strategy: PersonalizedStrategy,
        state: str,
    ) -> tuple[Action, bool]:
        """Return a safe action and whether any safety rule was triggered."""

        bounds = strategy.safety_bounds
        violated = False

        safe_action = Action(
            light_lux=clip(
                candidate_action.light_lux,
                bounds["light_lux_min"],
                bounds["light_lux_max"],
            ),
            aroma_level=clip(
                candidate_action.aroma_level,
                bounds["aroma_level_min"],
                bounds["aroma_level_max"],
            ),
            temp_delta_c=clip(
                candidate_action.temp_delta_c,
                bounds["temp_delta_min_c"],
                bounds["temp_delta_max_c"],
            ),
        )

        if safe_action != candidate_action:
            violated = True

        if previous_action is not None:
            rate_limited = Action(
                light_lux=clip(
                    safe_action.light_lux,
                    previous_action.light_lux - bounds["max_light_step_lux"],
                    previous_action.light_lux + bounds["max_light_step_lux"],
                ),
                aroma_level=clip(
                    safe_action.aroma_level,
                    previous_action.aroma_level - bounds["max_aroma_step"],
                    previous_action.aroma_level + bounds["max_aroma_step"],
                ),
                temp_delta_c=clip(
                    safe_action.temp_delta_c,
                    previous_action.temp_delta_c - bounds["max_temp_step_c"],
                    previous_action.temp_delta_c + bounds["max_temp_step_c"],
                ),
            )
            if rate_limited != safe_action:
                violated = True
            safe_action = rate_limited

        if state == "S6":
            s6_action = Action(
                light_lux=min(safe_action.light_lux, 1.0),
                aroma_level=min(safe_action.aroma_level, min(bounds["aroma_level_max"], 0.2)),
                temp_delta_c=max(safe_action.temp_delta_c, -0.5),
            )
            if s6_action != safe_action:
                violated = True
            safe_action = s6_action

        return safe_action, violated

