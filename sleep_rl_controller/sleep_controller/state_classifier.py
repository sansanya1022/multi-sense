"""Rule-based state classifier with reduced-feature fallback."""

from __future__ import annotations

from datetime import datetime

from sleep_controller.schemas import (
    PersonalizedStrategy,
    PhysiologySample,
    StateSnapshot,
    UserBaseline,
    UserProfile,
)
from sleep_controller.utils import clip, pct_delta


class StateClassifier:
    """Classify the current pre-sleep state into S1-S6."""

    def classify(
        self,
        profile: UserProfile,
        baseline: UserBaseline,
        strategy: PersonalizedStrategy,
        sample: PhysiologySample,
        phase: str,
    ) -> StateSnapshot:
        """Return one readable state snapshot."""

        thresholds = strategy.thresholds
        hr_pct = pct_delta(sample.hr_bpm, baseline.hr_bpm)
        br_pct = pct_delta(sample.br_bpm, baseline.br_bpm)
        temp_delta = sample.skin_temp_c - baseline.skin_temp_c
        stillness = (
            sample.stillness
            if sample.stillness is not None
            else baseline.stillness
            if baseline.stillness is not None
            else 0.5
        )
        reduced_feature = sample.br_irregularity is None or sample.stillness is None

        if (
            (baseline.avg_sleep_latency_min_7d or 0.0) >= thresholds["s6_latency_min"]
            or baseline.pathological_insomnia_nights_7d
            >= int(thresholds["s6_pathological_nights_7d"])
        ):
            return StateSnapshot(
                timestamp=sample.timestamp,
                user_id=sample.user_id,
                phase=phase,
                state="S6",
                confidence=0.95,
                rationale=(
                    "7-day insomnia risk rule triggered: "
                    f"latency={baseline.avg_sleep_latency_min_7d}, "
                    f"nights={baseline.pathological_insomnia_nights_7d}"
                ),
            )

        state = "S4"
        confidence = 0.60
        rationale = (
            f"HR is {hr_pct * 100:.1f}% vs baseline, "
            f"BR is {br_pct * 100:.1f}% vs baseline, "
            f"Temp delta is {temp_delta:.2f}C"
        )

        if hr_pct >= thresholds["s1_hr_pct"] and br_pct >= thresholds["s1_br_pct"]:
            state = "S1"
            confidence = clip(0.70 + 1.5 * max(hr_pct - thresholds["s1_hr_pct"], 0.0), 0.55, 0.95)
            rationale = (
                f"HR is {hr_pct * 100:.1f}% above baseline and "
                f"BR is {br_pct * 100:.1f}% above baseline"
            )
        elif br_pct >= thresholds["s2_br_pct"] or stillness <= 0.45:
            state = "S2"
            confidence = clip(0.65 + 1.2 * max(br_pct - thresholds["s2_br_pct"], 0.0), 0.55, 0.92)
            rationale = (
                f"BR is {br_pct * 100:.1f}% above baseline and "
                f"stillness is {stillness:.2f}"
            )
        elif hr_pct <= thresholds["s5_hr_pct"] and br_pct <= thresholds["s5_br_pct"]:
            state = "S5"
            confidence = clip(0.65 + 1.2 * abs(min(hr_pct - thresholds["s5_hr_pct"], 0.0)), 0.55, 0.92)
            rationale = (
                f"HR is {hr_pct * 100:.1f}% below baseline and "
                f"BR is {br_pct * 100:.1f}% below baseline"
            )
        elif (
            abs(hr_pct) >= thresholds["mild_hr_pct"]
            or abs(br_pct) >= thresholds["mild_br_pct"]
            or (sample.br_irregularity or 0.0) > 0.08
            or stillness < 0.60
        ):
            state = "S3"
            confidence = 0.58 if reduced_feature else 0.70
            rationale = (
                f"Mild anomaly detected: HR {hr_pct * 100:.1f}%, "
                f"BR {br_pct * 100:.1f}%, stillness {stillness:.2f}"
            )

        if reduced_feature and state not in {"S6", "S4"}:
            confidence = min(confidence, 0.60)
            rationale += " (reduced-feature mode)"

        if confidence < 0.55:
            state = "S4"
            confidence = 0.55
            rationale = f"Fallback to S4 due to low-confidence classification. {rationale}"

        return StateSnapshot(
            timestamp=sample.timestamp if isinstance(sample.timestamp, datetime) else datetime.now(),
            user_id=sample.user_id,
            phase=phase,
            state=state,
            confidence=confidence,
            rationale=rationale,
        )

