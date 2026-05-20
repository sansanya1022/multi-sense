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
    expected_sleep_hr_bpm: Optional[float] = None
    expected_sleep_br_bpm: Optional[float] = None
    preferred_temp_c: Optional[float] = None
    pathological_insomnia_nights_7d: int = 0
    avg_sleep_latency_min_7d: float = 20.0


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


def _pct_delta(value: Optional[float], baseline: Optional[float]) -> float:
    if value is None or baseline in (None, 0):
        return 0.0
    return (value - baseline) / baseline


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class RuleBasedStateClassifier:
    """
    Cold-start state classifier adapted from the user's PDF definition.
    Optional features may be missing; when confidence is low, caller should
    fall back to S4.
    """

    def classify(
        self,
        physiology: PhysiologyFrame,
        baseline: UserBaseline,
    ) -> Tuple[UserState, float, str]:
        hr_delta = _pct_delta(physiology.hr_bpm, baseline.hr_bpm)
        br_delta = _pct_delta(physiology.br_bpm, baseline.br_bpm)
        rmssd_delta = _pct_delta(physiology.rmssd_ms, baseline.rmssd_ms)
        has_rmssd = physiology.rmssd_ms is not None and baseline.rmssd_ms is not None
        has_br_irregularity = physiology.br_irregularity is not None
        br_irreg = physiology.br_irregularity or 0.0
        stillness = physiology.stillness if physiology.stillness is not None else 0.5

        if (
            baseline.pathological_insomnia_nights_7d >= 4
            and baseline.avg_sleep_latency_min_7d > 30
        ):
            return UserState.S6, 0.90, "7-day trend matches pathological insomnia risk"

        if has_rmssd and has_br_irregularity and hr_delta > 0.08 and rmssd_delta < -0.15 and br_irreg > 0.20:
            return UserState.S1, 0.85, "sympathetic over-activation pattern"

        if has_rmssd and hr_delta > 0.05 and abs(rmssd_delta) <= 0.10 and br_delta > 0.15 and stillness > 0.70:
            return UserState.S2, 0.80, "high arousal after work/exercise pattern"

        if has_rmssd and has_br_irregularity and -0.05 <= hr_delta <= 0.05 and abs(rmssd_delta) <= 0.10 and 0.10 <= br_irreg <= 0.20:
            return UserState.S3, 0.70, "rumination with irregular breathing pattern"

        if has_rmssd and hr_delta < -0.05 and rmssd_delta < -0.05 and br_delta < -0.10 and stillness > 0.85:
            return UserState.S5, 0.78, "fatigue and low arousal pattern"

        if not has_rmssd or not has_br_irregularity:
            if hr_delta > 0.08 and br_delta > 0.10:
                return UserState.S1, 0.55, "reduced-feature anxious/aroused fallback"
            if hr_delta > 0.05 and br_delta > 0.15:
                return UserState.S2, 0.55, "reduced-feature excitement fallback"
            if hr_delta < -0.05 and br_delta < -0.10 and stillness > 0.80:
                return UserState.S5, 0.55, "reduced-feature fatigue fallback"
            return UserState.S4, 0.45, "reduced-feature calm fallback"

        confidence = 0.60
        return UserState.S4, confidence, "default calm baseline fallback"


class AnchorActionTable:
    """
    Expert priors distilled from the PDF. Values are normalized targets:
    - light_lux: 0-5 lux
    - aroma_level: 0-1
    - temp_delta_c: relative cooling/heating target
    """

    def get(self, state: UserState, phase: SleepPhase) -> Dict[str, float]:
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
    ) -> float:
        expected_hr = baseline.expected_sleep_hr_bpm or max(baseline.hr_bpm - 6.0, 45.0)
        expected_br = baseline.expected_sleep_br_bpm or max(baseline.br_bpm - 2.0, 10.0)
        preferred_temp = baseline.preferred_temp_c or baseline.skin_temp_c

        hr_gain = 1.0 - min(abs(current.hr_bpm - expected_hr) / max(expected_hr, 1.0), 1.0)
        br_gain = 1.0 - min(abs(current.br_bpm - expected_br) / max(expected_br, 1.0), 1.0)
        temp_gain = 1.0 - min(abs(current.skin_temp_c - preferred_temp) / 3.0, 1.0)
        stillness_gain = current.stillness if current.stillness is not None else 0.5

        jitter_cost = 0.0
        if prev_action:
            jitter_cost = (
                abs(action_delta.get("light_lux", 0.0) - prev_action.get("light_lux", 0.0))
                + abs(action_delta.get("aroma_level", 0.0) - prev_action.get("aroma_level", 0.0))
                + abs(action_delta.get("temp_delta_c", 0.0) - prev_action.get("temp_delta_c", 0.0))
            ) / 3.0

        safety_cost = 1.0 if safety_violated else 0.0

        reward = (
            self.weights.hr_relax * hr_gain
            + self.weights.br_regular * br_gain
            + self.weights.temp_comfort * temp_gain
            + self.weights.stillness * stillness_gain
            - self.weights.action_jitter * jitter_cost
            - self.weights.safety_violation * safety_cost
        )
        return reward

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
    ) -> Tuple[Dict[str, float], bool]:
        action = proposed_action.copy()
        violated = False

        if state == UserState.S6:
            action["light_lux"] = min(action.get("light_lux", 0.0), 2.0)
            action["aroma_level"] = min(action.get("aroma_level", 0.0), 0.3)
            action["temp_delta_c"] = _clip(action.get("temp_delta_c", 0.0), -1.0, 0.5)

        action["light_lux"] = _clip(
            action.get("light_lux", 0.0),
            self.bounds.light_lux_min,
            self.bounds.light_lux_max,
        )
        action["aroma_level"] = _clip(
            action.get("aroma_level", 0.0),
            self.bounds.aroma_level_min,
            self.bounds.aroma_level_max,
        )
        action["temp_delta_c"] = _clip(
            action.get("temp_delta_c", 0.0),
            self.bounds.temp_delta_min_c,
            self.bounds.temp_delta_max_c,
        )

        if previous_action:
            action["light_lux"] = _clip(
                action["light_lux"],
                previous_action["light_lux"] - self.bounds.max_light_step_lux,
                previous_action["light_lux"] + self.bounds.max_light_step_lux,
            )
            action["aroma_level"] = _clip(
                action["aroma_level"],
                previous_action["aroma_level"] - self.bounds.max_aroma_step,
                previous_action["aroma_level"] + self.bounds.max_aroma_step,
            )
            action["temp_delta_c"] = _clip(
                action["temp_delta_c"],
                previous_action["temp_delta_c"] - self.bounds.max_temp_step_c,
                previous_action["temp_delta_c"] + self.bounds.max_temp_step_c,
            )

        for key, bounded in action.items():
            if proposed_action.get(key) != bounded:
                violated = True
        return action, violated


class PPOPolicyInterface:
    """
    Placeholder interface. In production this should be backed by a torch model
    that outputs Gaussian policy parameters and value estimates.
    """

    def act(self, observation: Sequence[float]) -> Dict[str, float]:
        return {"light_lux": 0.0, "aroma_level": 0.0, "temp_delta_c": 0.0}


class ObservationBuilder:
    def build_vector(
        self,
        observation: SleepObservation,
        baseline: UserBaseline,
    ) -> List[float]:
        physiology = observation.physiology
        state_one_hot = [1.0 if observation.state == state else 0.0 for state in UserState]
        return [
            physiology.hr_bpm,
            physiology.br_bpm,
            physiology.skin_temp_c,
            _pct_delta(physiology.hr_bpm, baseline.hr_bpm),
            _pct_delta(physiology.br_bpm, baseline.br_bpm),
            _pct_delta(physiology.skin_temp_c, baseline.skin_temp_c),
            physiology.br_irregularity or 0.0,
            physiology.stillness if physiology.stillness is not None else 0.5,
            observation.confidence,
            observation.last_light_lux,
            observation.last_aroma_level,
            observation.last_temp_delta_c,
            observation.elapsed_min,
        ] + state_one_hot


class SafeSleepController:
    """
    A deployable controller that combines:
    1. rule-based state classification
    2. anchor actions from expert tables
    3. PPO delta action
    4. safety clamping
    """

    def __init__(
        self,
        classifier: Optional[RuleBasedStateClassifier] = None,
        anchors: Optional[AnchorActionTable] = None,
        policy: Optional[PPOPolicyInterface] = None,
        observation_builder: Optional[ObservationBuilder] = None,
        safety_layer: Optional[SafetyLayer] = None,
    ) -> None:
        self.classifier = classifier or RuleBasedStateClassifier()
        self.anchors = anchors or AnchorActionTable()
        self.policy = policy or PPOPolicyInterface()
        self.observation_builder = observation_builder or ObservationBuilder()
        self.safety_layer = safety_layer or SafetyLayer()

    def decide_action(
        self,
        phase: SleepPhase,
        physiology: PhysiologyFrame,
        baseline: UserBaseline,
        previous_action: Optional[Dict[str, float]] = None,
        elapsed_min: float = 0.0,
    ) -> Dict[str, object]:
        state, confidence, rationale = self.classifier.classify(physiology, baseline)
        if confidence < 0.5:
            state = UserState.S4

        anchor_action = self.anchors.get(state=state, phase=phase)
        sleep_observation = SleepObservation(
            phase=phase,
            state=state,
            confidence=confidence,
            physiology=physiology,
            last_light_lux=(previous_action or {}).get("light_lux", 0.0),
            last_aroma_level=(previous_action or {}).get("aroma_level", 0.0),
            last_temp_delta_c=(previous_action or {}).get("temp_delta_c", 0.0),
            elapsed_min=elapsed_min,
        )
        obs_vector = self.observation_builder.build_vector(sleep_observation, baseline)
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
        )
        return {
            "state": state,
            "confidence": confidence,
            "rationale": rationale,
            "anchor_action": anchor_action,
            "delta_action": delta_action,
            "safe_action": safe_action,
            "safety_violated": violated,
        }
