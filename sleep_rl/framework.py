from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class UserState(str, Enum):
    S1 = "high_sympathetic_anxiety"
    S2 = "high_arousal_excitement"
    S3 = "moderate_rumination"
    S4 = "calm_baseline"
    S5 = "low_arousal_fatigue"
    S6 = "pathological_insomnia_risk"


class SleepPhase(str, Enum):
    PRE_SLEEP = "pre_sleep"
    INDUCTION = "induction"
    N1_N2 = "n1_n2"
    N3 = "n3"
    REM = "rem"
    WAKE = "wake"


class AgeBand(str, Enum):
    ADOLESCENT = "adolescent"
    YOUNG_ADULT = "young_adult"
    ADULT = "adult"
    MIDDLE_AGED = "middle_aged"
    SENIOR = "senior"


@dataclass
class PhysiologyFrame:
    hr_bpm: float
    br_bpm: float
    skin_temp_c: float
    rmssd_ms: Optional[float] = None
    br_irregularity: Optional[float] = None
    stillness: Optional[float] = None


@dataclass
class UserBaseline:
    hr_bpm: float
    br_bpm: float
    skin_temp_c: float
    rmssd_ms: Optional[float] = None
    stillness: Optional[float] = None
    expected_sleep_hr_bpm: Optional[float] = None
    expected_sleep_br_bpm: Optional[float] = None
    preferred_temp_c: Optional[float] = None
    pathological_insomnia_nights_7d: int = 0
    avg_sleep_latency_min_7d: float = 20.0


@dataclass
class UserProfile:
    age: int
    baseline: UserBaseline
    sex: Optional[str] = None
    stress_trait: float = 0.5
    temperature_sensitivity: float = 0.5
    aroma_sensitivity: float = 0.5
    sleep_schedule_type: str = "regular"


@dataclass
class PersonalizedThresholds:
    s1_hr_pct: float = 0.08
    s1_rmssd_pct: float = -0.15
    s1_br_irregularity: float = 0.20
    s2_hr_pct: float = 0.05
    s2_br_pct: float = 0.15
    s2_stillness: float = 0.70
    s3_br_irregularity_low: float = 0.10
    s3_br_irregularity_high: float = 0.20
    s5_hr_pct: float = -0.05
    s5_rmssd_pct: float = -0.05
    s5_br_pct: float = -0.10
    s5_stillness: float = 0.85
    pathological_latency_min: float = 30.0
    pathological_nights_7d: int = 4


@dataclass
class PersonalizedTargets:
    expected_sleep_hr_bpm: float
    expected_sleep_br_bpm: float
    preferred_temp_c: float


@dataclass
class PersonalizedStrategy:
    age_band: AgeBand
    thresholds: PersonalizedThresholds
    targets: PersonalizedTargets
    anchor_scales: Dict[str, float]
    safety_bounds: Dict[str, float]


@dataclass
class SleepObservation:
    phase: SleepPhase
    state: UserState
    confidence: float
    physiology: PhysiologyFrame
    last_light_lux: float
    last_aroma_level: float
    last_temp_delta_c: float
    elapsed_min: float
    user_profile: Dict[str, float] = field(default_factory=dict)


@dataclass
class SleepEpisodeOutcome:
    sleep_latency_min: float
    deep_sleep_ratio: float
    wake_count: int
    micro_arousal_count: int = 0
    subjective_morning_score: Optional[float] = None


@dataclass
class ActionBounds:
    light_lux_min: float = 0.0
    light_lux_max: float = 5.0
    aroma_level_min: float = 0.0
    aroma_level_max: float = 1.0
    temp_delta_min_c: float = -5.0
    temp_delta_max_c: float = 2.0
    max_temp_step_c: float = 0.5
    max_light_step_lux: float = 1.0
    max_aroma_step: float = 0.2


@dataclass
class RewardWeights:
    hr_relax: float = 0.25
    br_regular: float = 0.20
    temp_comfort: float = 0.15
    stillness: float = 0.10
    action_jitter: float = 0.10
    safety_violation: float = 0.20
    terminal_latency: float = 0.40
    terminal_depth: float = 0.35
    terminal_continuity: float = 0.25


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    rollout_horizon: int = 128
    minibatch_size: int = 32
    ppo_epochs: int = 10
    lagrangian_cost_coef: float = 0.5


@dataclass
class TinyPolicySpec:
    input_dim: int
    hidden_dims: Tuple[int, ...]
    output_dim: int = 3
    quantization_bits: int = 8
    include_bias: bool = True

    def parameter_count(self) -> int:
        dims = (self.input_dim,) + self.hidden_dims + (self.output_dim,)
        total = 0
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            total += in_dim * out_dim
            if self.include_bias:
                total += out_dim
        return total

    def weight_bytes(self) -> int:
        bytes_per_param = 4 if self.quantization_bits == 32 else 2 if self.quantization_bits == 16 else 1
        return self.parameter_count() * bytes_per_param

    def max_activation_bytes(self) -> int:
        max_width = max((self.input_dim,) + self.hidden_dims + (self.output_dim,))
        bytes_per_activation = 4 if self.quantization_bits == 32 else 2 if self.quantization_bits == 16 else 1
        # Double buffer plus small scratch factor for embedded kernels.
        return max_width * bytes_per_activation * 4

    def recommended_runtime_bytes(self) -> int:
        return self.weight_bytes() + self.max_activation_bytes() + 16 * 1024

    def estimate_report(self) -> Dict[str, int]:
        return {
            "parameter_count": self.parameter_count(),
            "weight_bytes": self.weight_bytes(),
            "max_activation_bytes": self.max_activation_bytes(),
            "recommended_runtime_bytes": self.recommended_runtime_bytes(),
        }


def _pct_delta(value: Optional[float], baseline: Optional[float]) -> float:
    if value is None or baseline in (None, 0):
        return 0.0
    return (value - baseline) / baseline


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _age_band(age: int) -> AgeBand:
    if age < 18:
        return AgeBand.ADOLESCENT
    if age < 30:
        return AgeBand.YOUNG_ADULT
    if age < 45:
        return AgeBand.ADULT
    if age < 60:
        return AgeBand.MIDDLE_AGED
    return AgeBand.SENIOR


class PersonalizationEngine:
    """
    Builds a deployable per-user strategy from age and normal physiology.
    This layer is intended to be deterministic and portable to ESP32 C/C++.
    """

    def build(self, profile: UserProfile) -> PersonalizedStrategy:
        band = _age_band(profile.age)
        baseline = profile.baseline

        threshold_shift = {
            AgeBand.ADOLESCENT: 0.01,
            AgeBand.YOUNG_ADULT: 0.00,
            AgeBand.ADULT: -0.005,
            AgeBand.MIDDLE_AGED: -0.01,
            AgeBand.SENIOR: -0.015,
        }[band]

        temp_step_limit = {
            AgeBand.ADOLESCENT: 0.5,
            AgeBand.YOUNG_ADULT: 0.5,
            AgeBand.ADULT: 0.45,
            AgeBand.MIDDLE_AGED: 0.40,
            AgeBand.SENIOR: 0.35,
        }[band]

        aroma_cap = 1.0 - 0.3 * profile.aroma_sensitivity
        temp_cooling_scale = 1.0 - 0.25 * profile.temperature_sensitivity

        thresholds = PersonalizedThresholds(
            s1_hr_pct=0.08 + threshold_shift,
            s1_rmssd_pct=-0.15,
            s1_br_irregularity=0.20,
            s2_hr_pct=0.05 + threshold_shift,
            s2_br_pct=0.15 + threshold_shift,
            s2_stillness=0.70,
            s3_br_irregularity_low=0.10,
            s3_br_irregularity_high=0.20,
            s5_hr_pct=-0.05 - threshold_shift,
            s5_rmssd_pct=-0.05,
            s5_br_pct=-0.10 - threshold_shift,
            s5_stillness=0.85,
            pathological_latency_min=32.0 if band in (AgeBand.MIDDLE_AGED, AgeBand.SENIOR) else 30.0,
            pathological_nights_7d=4,
        )

        targets = PersonalizedTargets(
            expected_sleep_hr_bpm=baseline.expected_sleep_hr_bpm or max(baseline.hr_bpm - 6.0, 45.0),
            expected_sleep_br_bpm=baseline.expected_sleep_br_bpm or max(baseline.br_bpm - 2.0, 10.0),
            preferred_temp_c=baseline.preferred_temp_c or baseline.skin_temp_c,
        )

        anchor_scales = {
            "light_scale": 0.95 if band == AgeBand.SENIOR else 1.0,
            "aroma_scale": aroma_cap,
            "temp_cooling_scale": temp_cooling_scale,
        }

        safety_bounds = {
            "light_lux_max": 5.0,
            "aroma_level_max": aroma_cap,
            "temp_delta_min_c": -4.0 if band in (AgeBand.MIDDLE_AGED, AgeBand.SENIOR) else -5.0,
            "temp_delta_max_c": 2.0,
            "max_temp_step_c": temp_step_limit,
            "max_light_step_lux": 1.0,
            "max_aroma_step": 0.15 if band == AgeBand.SENIOR else 0.20,
        }
        return PersonalizedStrategy(
            age_band=band,
            thresholds=thresholds,
            targets=targets,
            anchor_scales=anchor_scales,
            safety_bounds=safety_bounds,
        )


class RuleBasedStateClassifier:
    """
    Cold-start state classifier adapted from the user's PDF definition.
    It now supports user-personalized thresholds derived from age + baseline.
    """

    def classify(
        self,
        physiology: PhysiologyFrame,
        baseline: UserBaseline,
        strategy: Optional[PersonalizedStrategy] = None,
    ) -> Tuple[UserState, float, str]:
        thresholds = strategy.thresholds if strategy else PersonalizedThresholds()
        hr_delta = _pct_delta(physiology.hr_bpm, baseline.hr_bpm)
        br_delta = _pct_delta(physiology.br_bpm, baseline.br_bpm)
        rmssd_delta = _pct_delta(physiology.rmssd_ms, baseline.rmssd_ms)
        has_rmssd = physiology.rmssd_ms is not None and baseline.rmssd_ms is not None
        has_br_irregularity = physiology.br_irregularity is not None
        br_irreg = physiology.br_irregularity or 0.0
        stillness = physiology.stillness if physiology.stillness is not None else baseline.stillness or 0.5

        if (
            baseline.pathological_insomnia_nights_7d >= thresholds.pathological_nights_7d
            and baseline.avg_sleep_latency_min_7d > thresholds.pathological_latency_min
        ):
            return UserState.S6, 0.90, "7-day trend matches pathological insomnia risk"

        if (
            has_rmssd
            and has_br_irregularity
            and hr_delta > thresholds.s1_hr_pct
            and rmssd_delta < thresholds.s1_rmssd_pct
            and br_irreg > thresholds.s1_br_irregularity
        ):
            return UserState.S1, 0.85, "personalized sympathetic over-activation pattern"

        if (
            has_rmssd
            and hr_delta > thresholds.s2_hr_pct
            and abs(rmssd_delta) <= 0.10
            and br_delta > thresholds.s2_br_pct
            and stillness > thresholds.s2_stillness
        ):
            return UserState.S2, 0.80, "personalized high arousal pattern"

        if (
            has_rmssd
            and has_br_irregularity
            and -0.05 <= hr_delta <= 0.05
            and abs(rmssd_delta) <= 0.10
            and thresholds.s3_br_irregularity_low <= br_irreg <= thresholds.s3_br_irregularity_high
        ):
            return UserState.S3, 0.70, "personalized rumination pattern"

        if (
            has_rmssd
            and hr_delta < thresholds.s5_hr_pct
            and rmssd_delta < thresholds.s5_rmssd_pct
            and br_delta < thresholds.s5_br_pct
            and stillness > thresholds.s5_stillness
        ):
            return UserState.S5, 0.78, "personalized fatigue pattern"

        if not has_rmssd or not has_br_irregularity:
            if hr_delta > thresholds.s1_hr_pct and br_delta > 0.10:
                return UserState.S1, 0.55, "reduced-feature anxious/aroused fallback"
            if hr_delta > thresholds.s2_hr_pct and br_delta > thresholds.s2_br_pct:
                return UserState.S2, 0.55, "reduced-feature excitement fallback"
            if hr_delta < thresholds.s5_hr_pct and br_delta < thresholds.s5_br_pct and stillness > 0.80:
                return UserState.S5, 0.55, "reduced-feature fatigue fallback"
            return UserState.S4, 0.45, "reduced-feature calm fallback"

        return UserState.S4, 0.60, "default calm baseline fallback"


class AnchorActionTable:
    """
    Expert priors distilled from the PDF. Values are normalized targets:
    - light_lux: 0-5 lux
    - aroma_level: 0-1
    - temp_delta_c: relative cooling/heating target
    """

    def get(
        self,
        state: UserState,
        phase: SleepPhase,
        strategy: Optional[PersonalizedStrategy] = None,
    ) -> Dict[str, float]:
        base = {
            SleepPhase.PRE_SLEEP: {"light_lux": 3.0, "aroma_level": 0.45, "temp_delta_c": 0.0},
            SleepPhase.INDUCTION: {"light_lux": 1.5, "aroma_level": 0.55, "temp_delta_c": -1.0},
            SleepPhase.N1_N2: {"light_lux": 0.0, "aroma_level": 0.35, "temp_delta_c": -0.8},
            SleepPhase.N3: {"light_lux": 0.0, "aroma_level": 0.20, "temp_delta_c": -0.5},
            SleepPhase.REM: {"light_lux": 0.0, "aroma_level": 0.15, "temp_delta_c": -0.3},
            SleepPhase.WAKE: {"light_lux": 2.0, "aroma_level": 0.10, "temp_delta_c": 0.0},
        }[phase].copy()

        adjustments = {
            UserState.S1: {"light_lux": -0.8, "aroma_level": 0.15, "temp_delta_c": -1.0},
            UserState.S2: {"light_lux": -0.5, "aroma_level": 0.10, "temp_delta_c": -1.5},
            UserState.S3: {"light_lux": 0.0, "aroma_level": 0.05, "temp_delta_c": 0.0},
            UserState.S4: {"light_lux": 0.0, "aroma_level": 0.0, "temp_delta_c": 0.0},
            UserState.S5: {"light_lux": -1.5, "aroma_level": -0.15, "temp_delta_c": 0.0},
            UserState.S6: {"light_lux": 0.0, "aroma_level": 0.0, "temp_delta_c": 0.0},
        }[state]

        for key, delta in adjustments.items():
            base[key] += delta

        if strategy:
            base["light_lux"] *= strategy.anchor_scales["light_scale"]
            base["aroma_level"] *= strategy.anchor_scales["aroma_scale"]
            if base["temp_delta_c"] < 0:
                base["temp_delta_c"] *= strategy.anchor_scales["temp_cooling_scale"]
        return base


class RewardCalculator:
    def __init__(self, weights: Optional[RewardWeights] = None) -> None:
        self.weights = weights or RewardWeights()

    def dense_reward(
        self,
        current: PhysiologyFrame,
        baseline: UserBaseline,
        action_delta: Dict[str, float],
        safety_violated: bool,
        prev_action: Optional[Dict[str, float]] = None,
        strategy: Optional[PersonalizedStrategy] = None,
    ) -> float:
        if strategy:
            expected_hr = strategy.targets.expected_sleep_hr_bpm
            expected_br = strategy.targets.expected_sleep_br_bpm
            preferred_temp = strategy.targets.preferred_temp_c
        else:
            expected_hr = baseline.expected_sleep_hr_bpm or max(baseline.hr_bpm - 6.0, 45.0)
            expected_br = baseline.expected_sleep_br_bpm or max(baseline.br_bpm - 2.0, 10.0)
            preferred_temp = baseline.preferred_temp_c or baseline.skin_temp_c

        hr_gain = 1.0 - min(abs(current.hr_bpm - expected_hr) / max(expected_hr, 1.0), 1.0)
        br_gain = 1.0 - min(abs(current.br_bpm - expected_br) / max(expected_br, 1.0), 1.0)
        temp_gain = 1.0 - min(abs(current.skin_temp_c - preferred_temp) / 3.0, 1.0)
        stillness_gain = current.stillness if current.stillness is not None else baseline.stillness or 0.5

        jitter_cost = 0.0
        if prev_action:
            jitter_cost = (
                abs(action_delta.get("light_lux", 0.0) - prev_action.get("light_lux", 0.0))
                + abs(action_delta.get("aroma_level", 0.0) - prev_action.get("aroma_level", 0.0))
                + abs(action_delta.get("temp_delta_c", 0.0) - prev_action.get("temp_delta_c", 0.0))
            ) / 3.0

        safety_cost = 1.0 if safety_violated else 0.0
        return (
            self.weights.hr_relax * hr_gain
            + self.weights.br_regular * br_gain
            + self.weights.temp_comfort * temp_gain
            + self.weights.stillness * stillness_gain
            - self.weights.action_jitter * jitter_cost
            - self.weights.safety_violation * safety_cost
        )

    def terminal_reward(self, outcome: SleepEpisodeOutcome) -> float:
        latency_score = 1.0 - min(outcome.sleep_latency_min / 60.0, 1.0)
        depth_score = _clip(outcome.deep_sleep_ratio, 0.0, 1.0)
        continuity_score = 1.0 - min((outcome.wake_count + outcome.micro_arousal_count) / 10.0, 1.0)
        return (
            self.weights.terminal_latency * latency_score
            + self.weights.terminal_depth * depth_score
            + self.weights.terminal_continuity * continuity_score
        )


class SafetyLayer:
    def __init__(self, bounds: Optional[ActionBounds] = None) -> None:
        self.bounds = bounds or ActionBounds()

    def apply(
        self,
        proposed_action: Dict[str, float],
        previous_action: Optional[Dict[str, float]] = None,
        state: Optional[UserState] = None,
        strategy: Optional[PersonalizedStrategy] = None,
    ) -> Tuple[Dict[str, float], bool]:
        action = proposed_action.copy()
        violated = False
        bounds = self._merge_bounds(strategy)

        if state == UserState.S6:
            action["light_lux"] = min(action.get("light_lux", 0.0), 2.0)
            action["aroma_level"] = min(action.get("aroma_level", 0.0), 0.3)
            action["temp_delta_c"] = _clip(action.get("temp_delta_c", 0.0), -1.0, 0.5)

        action["light_lux"] = _clip(action.get("light_lux", 0.0), bounds.light_lux_min, bounds.light_lux_max)
        action["aroma_level"] = _clip(action.get("aroma_level", 0.0), bounds.aroma_level_min, bounds.aroma_level_max)
        action["temp_delta_c"] = _clip(action.get("temp_delta_c", 0.0), bounds.temp_delta_min_c, bounds.temp_delta_max_c)

        if previous_action:
            action["light_lux"] = _clip(
                action["light_lux"],
                previous_action["light_lux"] - bounds.max_light_step_lux,
                previous_action["light_lux"] + bounds.max_light_step_lux,
            )
            action["aroma_level"] = _clip(
                action["aroma_level"],
                previous_action["aroma_level"] - bounds.max_aroma_step,
                previous_action["aroma_level"] + bounds.max_aroma_step,
            )
            action["temp_delta_c"] = _clip(
                action["temp_delta_c"],
                previous_action["temp_delta_c"] - bounds.max_temp_step_c,
                previous_action["temp_delta_c"] + bounds.max_temp_step_c,
            )

        for key, bounded in action.items():
            if proposed_action.get(key) != bounded:
                violated = True
        return action, violated

    def _merge_bounds(self, strategy: Optional[PersonalizedStrategy]) -> ActionBounds:
        if not strategy:
            return self.bounds
        return ActionBounds(
            light_lux_min=self.bounds.light_lux_min,
            light_lux_max=strategy.safety_bounds["light_lux_max"],
            aroma_level_min=self.bounds.aroma_level_min,
            aroma_level_max=strategy.safety_bounds["aroma_level_max"],
            temp_delta_min_c=strategy.safety_bounds["temp_delta_min_c"],
            temp_delta_max_c=strategy.safety_bounds["temp_delta_max_c"],
            max_temp_step_c=strategy.safety_bounds["max_temp_step_c"],
            max_light_step_lux=strategy.safety_bounds["max_light_step_lux"],
            max_aroma_step=strategy.safety_bounds["max_aroma_step"],
        )


class TinyPolicyInterface:
    """
    Edge-side distilled policy placeholder.
    In production this should be backed by a tiny MLP exported to ESP-DL.
    """

    def act(self, observation: Sequence[float]) -> Dict[str, float]:
        return {"light_lux": 0.0, "aroma_level": 0.0, "temp_delta_c": 0.0}


class ObservationBuilder:
    def build_vector(
        self,
        observation: SleepObservation,
        baseline: UserBaseline,
        strategy: Optional[PersonalizedStrategy] = None,
    ) -> List[float]:
        physiology = observation.physiology
        state_one_hot = [1.0 if observation.state == state else 0.0 for state in UserState]
        target_hr = strategy.targets.expected_sleep_hr_bpm if strategy else baseline.expected_sleep_hr_bpm or baseline.hr_bpm
        target_br = strategy.targets.expected_sleep_br_bpm if strategy else baseline.expected_sleep_br_bpm or baseline.br_bpm
        target_temp = strategy.targets.preferred_temp_c if strategy else baseline.preferred_temp_c or baseline.skin_temp_c
        profile_values = list(observation.user_profile.values())
        return [
            physiology.hr_bpm,
            physiology.br_bpm,
            physiology.skin_temp_c,
            _pct_delta(physiology.hr_bpm, baseline.hr_bpm),
            _pct_delta(physiology.br_bpm, baseline.br_bpm),
            _pct_delta(physiology.skin_temp_c, baseline.skin_temp_c),
            _pct_delta(physiology.hr_bpm, target_hr),
            _pct_delta(physiology.br_bpm, target_br),
            _pct_delta(physiology.skin_temp_c, target_temp),
            physiology.br_irregularity or 0.0,
            physiology.stillness if physiology.stillness is not None else baseline.stillness or 0.5,
            observation.confidence,
            observation.last_light_lux,
            observation.last_aroma_level,
            observation.last_temp_delta_c,
            observation.elapsed_min,
        ] + state_one_hot + profile_values


class SafeSleepController:
    """
    A deployable controller that combines:
    1. user personalization from age + baseline
    2. rule-based state classification
    3. expert anchor actions
    4. tiny edge policy delta action
    5. safety clamping
    """

    def __init__(
        self,
        personalization_engine: Optional[PersonalizationEngine] = None,
        classifier: Optional[RuleBasedStateClassifier] = None,
        anchors: Optional[AnchorActionTable] = None,
        policy: Optional[TinyPolicyInterface] = None,
        observation_builder: Optional[ObservationBuilder] = None,
        safety_layer: Optional[SafetyLayer] = None,
    ) -> None:
        self.personalization_engine = personalization_engine or PersonalizationEngine()
        self.classifier = classifier or RuleBasedStateClassifier()
        self.anchors = anchors or AnchorActionTable()
        self.policy = policy or TinyPolicyInterface()
        self.observation_builder = observation_builder or ObservationBuilder()
        self.safety_layer = safety_layer or SafetyLayer()

    def decide_action(
        self,
        phase: SleepPhase,
        profile: UserProfile,
        physiology: PhysiologyFrame,
        previous_action: Optional[Dict[str, float]] = None,
        elapsed_min: float = 0.0,
    ) -> Dict[str, object]:
        strategy = self.personalization_engine.build(profile)
        state, confidence, rationale = self.classifier.classify(
            physiology=physiology,
            baseline=profile.baseline,
            strategy=strategy,
        )
        if confidence < 0.5:
            state = UserState.S4

        anchor_action = self.anchors.get(state=state, phase=phase, strategy=strategy)
        sleep_observation = SleepObservation(
            phase=phase,
            state=state,
            confidence=confidence,
            physiology=physiology,
            last_light_lux=(previous_action or {}).get("light_lux", 0.0),
            last_aroma_level=(previous_action or {}).get("aroma_level", 0.0),
            last_temp_delta_c=(previous_action or {}).get("temp_delta_c", 0.0),
            elapsed_min=elapsed_min,
            user_profile={
                "age": float(profile.age),
                "stress_trait": profile.stress_trait,
                "temperature_sensitivity": profile.temperature_sensitivity,
                "aroma_sensitivity": profile.aroma_sensitivity,
            },
        )
        obs_vector = self.observation_builder.build_vector(
            observation=sleep_observation,
            baseline=profile.baseline,
            strategy=strategy,
        )
        delta_action = self.policy.act(obs_vector)
        proposed_action = {
            "light_lux": anchor_action["light_lux"] + delta_action.get("light_lux", 0.0),
            "aroma_level": anchor_action["aroma_level"] + delta_action.get("aroma_level", 0.0),
            "temp_delta_c": anchor_action["temp_delta_c"] + delta_action.get("temp_delta_c", 0.0),
        }
        safe_action, violated = self.safety_layer.apply(
            proposed_action=proposed_action,
            previous_action=previous_action,
            state=state,
            strategy=strategy,
        )
        return {
            "strategy": strategy,
            "state": state,
            "confidence": confidence,
            "rationale": rationale,
            "anchor_action": anchor_action,
            "delta_action": delta_action,
            "safe_action": safe_action,
            "safety_violated": violated,
            "observation_dim": len(obs_vector),
        }
