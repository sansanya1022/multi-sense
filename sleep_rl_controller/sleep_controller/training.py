"""Simplified but runnable training interfaces for PPO-style and distillation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.data import DataLoader, Dataset

from sleep_controller.schemas import Action, ControlDecision
from sleep_controller.tiny_model import TinyMLPPolicy
from sleep_controller.utils import set_global_seed


class OfflineDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Offline dataset of observations, actions, and rewards."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> None:
        if observations.shape[0] != actions.shape[0] or actions.shape[0] != rewards.shape[0]:
            raise ValueError("observations, actions, and rewards must have equal length")
        self.observations = torch.tensor(observations, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)
        self.rewards = torch.tensor(rewards, dtype=torch.float32)

    @classmethod
    def from_decisions(
        cls,
        decisions: Sequence[ControlDecision],
        rewards: Sequence[float],
    ) -> "OfflineDataset":
        """Build dataset from controller decisions and dense rewards."""

        observations = np.stack([decision.observation for decision in decisions], axis=0)
        actions = np.stack([decision.model_delta.as_array() for decision in decisions], axis=0)
        rewards_array = np.array(rewards, dtype=np.float32)
        return cls(observations=observations, actions=actions, rewards=rewards_array)

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.observations[index], self.actions[index], self.rewards[index]


@dataclass(slots=True)
class TrainingMetrics:
    """Simple training metrics container."""

    loss: float
    extra: dict[str, float]


class DistillationTrainer:
    """Train tiny policy to imitate target deltas from logs."""

    def __init__(
        self,
        policy: TinyMLPPolicy,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        set_global_seed(seed)
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def train_epoch(self, dataset: OfflineDataset, batch_size: int = 16) -> TrainingMetrics:
        """Run one epoch of delta imitation training."""

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        total_loss = 0.0
        batch_count = 0
        for observations, actions, _ in loader:
            predicted = self.policy(observations)
            loss = self.loss_fn(predicted, actions)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1
        average_loss = total_loss / max(batch_count, 1)
        return TrainingMetrics(loss=average_loss, extra={"distillation_loss": average_loss})


class PPOTrainer:
    """Small offline PPO-style trainer over logged observations and rewards."""

    def __init__(
        self,
        policy: TinyMLPPolicy,
        learning_rate: float = 1e-3,
        clip_epsilon: float = 0.2,
        action_std: float = 0.1,
        seed: int = 42,
    ) -> None:
        set_global_seed(seed)
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.clip_epsilon = clip_epsilon
        self.action_std = action_std

    def train_epoch(self, dataset: OfflineDataset, batch_size: int = 16) -> TrainingMetrics:
        """Run one simplified PPO-style epoch using logged actions as behavior actions."""

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        total_loss = 0.0
        total_surrogate = 0.0
        batch_count = 0
        for observations, actions, rewards in loader:
            means = self.policy(observations)
            old_means = actions.detach()
            std = torch.full_like(means, self.action_std)
            dist = Normal(means, std)
            old_dist = Normal(old_means, std)
            new_log_prob = dist.log_prob(actions).sum(dim=-1)
            old_log_prob = old_dist.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(new_log_prob - old_log_prob)
            advantages = rewards - rewards.mean()
            surrogate_1 = ratio * advantages
            surrogate_2 = torch.clamp(
                ratio,
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon,
            ) * advantages
            loss = -torch.mean(torch.minimum(surrogate_1, surrogate_2))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())
            total_surrogate += float(torch.mean(torch.minimum(surrogate_1, surrogate_2)).item())
            batch_count += 1
        return TrainingMetrics(
            loss=total_loss / max(batch_count, 1),
            extra={"ppo_surrogate": total_surrogate / max(batch_count, 1)},
        )


def train_tiny_policy_from_logs(
    decisions: Sequence[ControlDecision],
    rewards: Sequence[float],
    epochs: int = 3,
    mode: str = "distillation",
    seed: int = 42,
) -> TinyMLPPolicy:
    """Train a tiny policy from logged decisions using distillation or simplified PPO."""

    policy = TinyMLPPolicy(seed=seed)
    dataset = OfflineDataset.from_decisions(decisions, rewards)
    if mode == "ppo":
        trainer = PPOTrainer(policy=policy, seed=seed)
    else:
        trainer = DistillationTrainer(policy=policy, seed=seed)
    for _ in range(epochs):
        trainer.train_epoch(dataset)
    return policy

