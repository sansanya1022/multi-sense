"""Generate personalized thresholds, targets, anchor scales, and safety bounds."""

from __future__ import annotations

from sleep_controller.schemas import PersonalizedStrategy, UserBaseline, UserProfile
from sleep_controller.utils import clip


class PersonalizedEngine:
    """Build user-specific strategy parameters from profile and baseline."""

    def resolve_age_band(self, age: int) -> str:
        """Map age to a coarse age band."""

        if age < 13:
            return "child"
        if age < 18:
            return "teen"
        if age < 30:
            return "young_adult"
        if age < 50:
            return "adult"
        if age < 65:
            return "older_adult"
        return "elderly"

    def generate(
        self,
        profile: UserProfile,
        baseline: UserBaseline,
    ) -> PersonalizedStrategy:
        """Generate a deterministic personalized strategy."""

        age_band = self.resolve_age_band(profile.age)

        age_hr_shift = {
            "child": 0.02,
            "teen": 0.01,
            "young_adult": 0.00,
            "adult": -0.005,
            "older_adult": -0.01,
            "elderly": -0.015,
        }[age_band]

        thresholds = {
            "s1_hr_pct": 0.08 + age_hr_shift,
            "s1_br_pct": 0.05,
            "s2_br_pct": 0.12 + max(0.0, profile.stress_trait - 0.5) * 0.08,
            "s5_hr_pct": -0.05 - age_hr_shift,
            "s5_br_pct": -0.08 - age_hr_shift,
            "mild_hr_pct": 0.04,
            "mild_br_pct": 0.05,
            "s6_latency_min": 35.0 if age_band in {"older_adult", "elderly"} else 30.0,
            "s6_pathological_nights_7d": 4.0,
        }

        hr_scale = 0.90 + 0.05 * (1.0 - profile.stress_trait)
        br_scale = 0.85 + 0.10 * (1.0 - profile.stress_trait)
        temp_drop = 0.10 + 0.20 * profile.temperature_sensitivity
        targets = {
            "expected_sleep_hr_bpm": baseline.hr_bpm * hr_scale,
            "expected_sleep_br_bpm": baseline.br_bpm * br_scale,
            "preferred_temp_c": baseline.skin_temp_c - temp_drop,
        }

        anchor_scales = {
            "light_scale": 0.95 if age_band in {"older_adult", "elderly"} else 1.0,
            "aroma_scale": clip(1.0 - 0.5 * profile.aroma_sensitivity, 0.5, 1.0),
            "temp_cooling_scale": clip(
                1.0 - 0.25 * profile.temperature_sensitivity - (0.10 if age_band == "elderly" else 0.0),
                0.6,
                1.0,
            ),
            "relaxation_boost": 1.0 + 0.25 * profile.stress_trait,
        }

        safety_bounds = {
            "light_lux_min": 0.0,
            "light_lux_max": 5.0,
            "aroma_level_min": 0.0,
            "aroma_level_max": clip(1.0 - 0.4 * profile.aroma_sensitivity, 0.4, 1.0),
            "temp_delta_min_c": -4.0 if age_band in {"older_adult", "elderly"} else -5.0,
            "temp_delta_max_c": 2.0,
            "max_light_step_lux": 1.0,
            "max_aroma_step": 0.08 if age_band == "elderly" else 0.10,
            "max_temp_step_c": 0.35 if age_band == "elderly" else 0.50,
        }

        return PersonalizedStrategy(
            age_band=age_band,
            thresholds=thresholds,
            targets=targets,
            anchor_scales=anchor_scales,
            safety_bounds=safety_bounds,
        )

