"""Top-level sleep controller that implements anchor + delta + safety logic."""

from __future__ import annotations

from sleep_controller.anchor_policy import AnchorPolicy
from sleep_controller.observation_builder import ObservationBuilder
from sleep_controller.personalized_engine import PersonalizedEngine
from sleep_controller.safety_layer import SafetyLayer
from sleep_controller.schemas import (
    Action,
    ActionLog,
    ControlDecision,
    PersonalizedStrategy,
    PhysiologySample,
    UserBaseline,
    UserProfile,
)
from sleep_controller.state_classifier import StateClassifier
from sleep_controller.tiny_model import TinyMLPPolicy


class SleepController:
    """One-step controller for the personalized sleep environment regulator."""

    def __init__(
        self,
        personalized_engine: PersonalizedEngine | None = None,
        state_classifier: StateClassifier | None = None,
        anchor_policy: AnchorPolicy | None = None,
        observation_builder: ObservationBuilder | None = None,
        tiny_policy: TinyMLPPolicy | None = None,
        safety_layer: SafetyLayer | None = None,
    ) -> None:
        self.personalized_engine = personalized_engine or PersonalizedEngine()
        self.state_classifier = state_classifier or StateClassifier()
        self.anchor_policy = anchor_policy or AnchorPolicy()
        self.observation_builder = observation_builder or ObservationBuilder()
        self.tiny_policy = tiny_policy or TinyMLPPolicy()
        self.safety_layer = safety_layer or SafetyLayer()

    def step(
        self,
        profile: UserProfile,
        baseline: UserBaseline,
        sample: PhysiologySample,
        phase: str,
        previous_action: Action | None = None,
        elapsed_min: float = 0.0,
        strategy_override: PersonalizedStrategy | None = None,
    ) -> ControlDecision:
        """Execute one full controller step."""

        strategy = strategy_override or self.personalized_engine.generate(profile, baseline)
        state_snapshot = self.state_classifier.classify(
            profile=profile,
            baseline=baseline,
            strategy=strategy,
            sample=sample,
            phase=phase,
        )
        anchor_action = self.anchor_policy.get_anchor(
            state=state_snapshot.state,
            phase=phase,
            strategy=strategy,
        )
        observation = self.observation_builder.build(
            profile=profile,
            baseline=baseline,
            strategy=strategy,
            sample=sample,
            state_snapshot=state_snapshot,
            elapsed_min=elapsed_min,
        )

        if state_snapshot.state == "S6":
            model_delta = Action.zero()
        else:
            model_delta = self.tiny_policy.predict_delta(observation)

        candidate_action = anchor_action + model_delta
        safe_action, safety_violated = self.safety_layer.apply(
            candidate_action=candidate_action,
            previous_action=previous_action,
            strategy=strategy,
            state=state_snapshot.state,
        )
        action_log = ActionLog(
            timestamp=sample.timestamp,
            user_id=sample.user_id,
            phase=phase,
            state=state_snapshot.state,
            anchor_light_lux=anchor_action.light_lux,
            anchor_aroma_level=anchor_action.aroma_level,
            anchor_temp_delta_c=anchor_action.temp_delta_c,
            model_light_delta=model_delta.light_lux,
            model_aroma_delta=model_delta.aroma_level,
            model_temp_delta_c=model_delta.temp_delta_c,
            final_light_lux=safe_action.light_lux,
            final_aroma_level=safe_action.aroma_level,
            final_temp_delta_c=safe_action.temp_delta_c,
            safety_violated=safety_violated,
        )
        return ControlDecision(
            strategy=strategy,
            state_snapshot=state_snapshot,
            anchor_action=anchor_action,
            model_delta=model_delta,
            final_safe_action=safe_action,
            safety_violated=safety_violated,
            observation=observation,
            action_log=action_log,
        )

