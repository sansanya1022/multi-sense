"""Core schema definitions for the personalized sleep controller MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Optional

import numpy as np

SleepState = Literal["S1", "S2", "S3", "S4", "S5", "S6"]


def _validate_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


@dataclass(slots=True)
class UserProfile:
    """User profile used for personalization."""

    user_id: str
    age: int
    sex: Optional[str]
    stress_trait: float
    temperature_sensitivity: float
    aroma_sensitivity: float
    sleep_schedule_type: str

    def __post_init__(self) -> None:
        if self.age < 0:
            raise ValueError("age must be non-negative")
        _validate_unit_interval("stress_trait", self.stress_trait)
        _validate_unit_interval(
            "temperature_sensitivity",
            self.temperature_sensitivity,
        )
        _validate_unit_interval("aroma_sensitivity", self.aroma_sensitivity)


@dataclass(slots=True)
class UserBaseline:
    """Normal-state baseline physiology for a user."""

    user_id: str
    hr_bpm: float
    br_bpm: float
    skin_temp_c: float
    rmssd_ms: Optional[float]
    stillness: Optional[float]
    avg_sleep_latency_min_7d: Optional[float]
    pathological_insomnia_nights_7d: int

    def __post_init__(self) -> None:
        if self.hr_bpm <= 0 or self.br_bpm <= 0:
            raise ValueError("hr_bpm and br_bpm must be positive")
        if self.pathological_insomnia_nights_7d < 0:
            raise ValueError("pathological_insomnia_nights_7d must be non-negative")
        if self.stillness is not None:
            _validate_unit_interval("stillness", self.stillness)


@dataclass(slots=True)
class PhysiologySample:
    """One real-time physiology observation."""

    timestamp: datetime
    user_id: str
    hr_bpm: float
    br_bpm: float
    skin_temp_c: float
    rmssd_ms: Optional[float]
    br_irregularity: Optional[float]
    stillness: Optional[float]

    def __post_init__(self) -> None:
        if self.hr_bpm <= 0 or self.br_bpm <= 0:
            raise ValueError("hr_bpm and br_bpm must be positive")
        if self.stillness is not None:
            _validate_unit_interval("stillness", self.stillness)


@dataclass(slots=True)
class StateSnapshot:
    """Result of the rule-based state classification."""

    timestamp: datetime
    user_id: str
    phase: str
    state: SleepState
    confidence: float
    rationale: str

    def __post_init__(self) -> None:
        _validate_unit_interval("confidence", self.confidence)


@dataclass(slots=True)
class PersonalizedStrategy:
    """Personalized thresholds, targets, scales, and safety bounds."""

    age_band: str
    thresholds: dict[str, float]
    targets: dict[str, float]
    anchor_scales: dict[str, float]
    safety_bounds: dict[str, float]


@dataclass(slots=True)
class Action:
    """Continuous action for light, aroma, and local temperature control."""

    light_lux: float
    aroma_level: float
    temp_delta_c: float

    def __add__(self, other: "Action") -> "Action":
        return Action(
            light_lux=self.light_lux + other.light_lux,
            aroma_level=self.aroma_level + other.aroma_level,
            temp_delta_c=self.temp_delta_c + other.temp_delta_c,
        )

    def __sub__(self, other: "Action") -> "Action":
        return Action(
            light_lux=self.light_lux - other.light_lux,
            aroma_level=self.aroma_level - other.aroma_level,
            temp_delta_c=self.temp_delta_c - other.temp_delta_c,
        )

    @staticmethod
    def zero() -> "Action":
        return Action(light_lux=0.0, aroma_level=0.0, temp_delta_c=0.0)

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.light_lux, self.aroma_level, self.temp_delta_c],
            dtype=np.float32,
        )


@dataclass(slots=True)
class ActionLog:
    """Structured log for each controller decision."""

    timestamp: datetime
    user_id: str
    phase: str
    state: str
    anchor_light_lux: float
    anchor_aroma_level: float
    anchor_temp_delta_c: float
    model_light_delta: float
    model_aroma_delta: float
    model_temp_delta_c: float
    final_light_lux: float
    final_aroma_level: float
    final_temp_delta_c: float
    safety_violated: bool


@dataclass(slots=True)
class EpisodeOutcome:
    """Single-night terminal outcome summary."""

    date: date
    user_id: str
    sleep_latency_min: float
    deep_sleep_ratio: Optional[float]
    deep_sleep_proxy: Optional[float]
    wake_count: int
    micro_arousal_count: int
    subjective_morning_score: float

    def __post_init__(self) -> None:
        if self.sleep_latency_min < 0:
            raise ValueError("sleep_latency_min must be non-negative")
        if self.wake_count < 0 or self.micro_arousal_count < 0:
            raise ValueError("wake counts must be non-negative")
        if not 0.0 <= self.subjective_morning_score <= 5.0:
            raise ValueError("subjective_morning_score must be within [0, 5]")


@dataclass(slots=True)
class ControlDecision:
    """Detailed result returned by one controller step."""

    strategy: PersonalizedStrategy
    state_snapshot: StateSnapshot
    anchor_action: Action
    model_delta: Action
    final_safe_action: Action
    safety_violated: bool
    observation: np.ndarray
    action_log: ActionLog


@dataclass(slots=True)
class SimulationResult:
    """Result bundle returned by the simulator."""

    state_snapshots: list[StateSnapshot] = field(default_factory=list)
    action_logs: list[ActionLog] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    decisions: list[ControlDecision] = field(default_factory=list)
    episode_outcome: Optional[EpisodeOutcome] = None

