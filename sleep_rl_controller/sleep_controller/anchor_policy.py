"""Personalized anchor policy for state- and phase-dependent baseline actions."""

from __future__ import annotations

from sleep_controller.schemas import Action, PersonalizedStrategy
from sleep_controller.utils import clip


class AnchorPolicy:
    """Return a deterministic anchor action before model delta adjustment."""

    def get_anchor(
        self,
        state: str,
        phase: str,
        strategy: PersonalizedStrategy,
    ) -> Action:
        """Generate an anchor action for the given state and phase."""

        base_map = {
            "S1": Action(light_lux=1.6, aroma_level=0.55, temp_delta_c=-1.6),
            "S2": Action(light_lux=1.0, aroma_level=0.45, temp_delta_c=-1.3),
            "S3": Action(light_lux=1.8, aroma_level=0.35, temp_delta_c=-0.8),
            "S4": Action(light_lux=2.2, aroma_level=0.20, temp_delta_c=-0.5),
            "S5": Action(light_lux=1.0, aroma_level=0.10, temp_delta_c=-0.2),
            "S6": Action(light_lux=0.6, aroma_level=0.10, temp_delta_c=-0.2),
        }
        action = base_map[state]

        if phase.lower() == "induction":
            action = Action(
                light_lux=action.light_lux * 0.75,
                aroma_level=action.aroma_level * 1.10,
                temp_delta_c=action.temp_delta_c * 1.10,
            )

        light_scale = strategy.anchor_scales["light_scale"]
        aroma_scale = strategy.anchor_scales["aroma_scale"]
        temp_cooling_scale = strategy.anchor_scales["temp_cooling_scale"]
        relaxation_boost = strategy.anchor_scales.get("relaxation_boost", 1.0)

        if state in {"S1", "S3"}:
            aroma_scale *= relaxation_boost
            temp_cooling_scale *= min(relaxation_boost, 1.2)

        scaled = Action(
            light_lux=action.light_lux * light_scale,
            aroma_level=action.aroma_level * aroma_scale,
            temp_delta_c=action.temp_delta_c * temp_cooling_scale,
        )

        bounds = strategy.safety_bounds
        return Action(
            light_lux=clip(
                scaled.light_lux,
                bounds["light_lux_min"],
                bounds["light_lux_max"],
            ),
            aroma_level=clip(
                scaled.aroma_level,
                bounds["aroma_level_min"],
                bounds["aroma_level_max"],
            ),
            temp_delta_c=clip(
                scaled.temp_delta_c,
                bounds["temp_delta_min_c"],
                bounds["temp_delta_max_c"],
            ),
        )

