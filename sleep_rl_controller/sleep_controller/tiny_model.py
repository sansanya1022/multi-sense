"""Tiny policy with PyTorch implementation and numpy fallback for demo usage."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from sleep_controller.schemas import Action

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - environment-dependent fallback
    torch = None
    nn = None


if torch is not None:

    class TinyMLPPolicy(nn.Module):
        """Small MLP with bounded action outputs for light/aroma/temperature deltas."""

        def __init__(
            self,
            input_dim: int = 20,
            hidden: Sequence[int] = (32, 16),
            output_dim: int = 3,
            quantization_target: str = "int8",
            seed: int = 42,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden = tuple(hidden)
            self.output_dim = output_dim
            self.quantization_target = quantization_target
            self.max_deltas = torch.tensor([1.0, 0.15, 0.5], dtype=torch.float32)

            layers: list[nn.Module] = []
            in_dim = input_dim
            for hidden_dim in self.hidden:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, output_dim))
            self.network = nn.Sequential(*layers)
            self._initialize_weights(seed)

        def _initialize_weights(self, seed: int) -> None:
            """Initialize model weights deterministically."""

            torch.manual_seed(seed)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

        def forward(self, observation: torch.Tensor) -> torch.Tensor:
            """Return bounded action deltas."""

            if observation.ndim == 1:
                observation = observation.unsqueeze(0)
            raw = self.network(observation)
            bounded = torch.tanh(raw) * self.max_deltas.to(raw.device)
            return bounded

        @torch.no_grad()
        def predict_delta(self, observation: np.ndarray | torch.Tensor) -> Action:
            """Predict one action delta and return it as an Action dataclass."""

            if isinstance(observation, np.ndarray):
                observation_tensor = torch.tensor(
                    observation.tolist(),
                    dtype=torch.float32,
                )
            else:
                observation_tensor = observation.to(dtype=torch.float32)
            deltas = self.forward(observation_tensor).squeeze(0).cpu().tolist()
            return Action(
                light_lux=float(deltas[0]),
                aroma_level=float(deltas[1]),
                temp_delta_c=float(deltas[2]),
            )

        def dummy_input(self) -> torch.Tensor:
            """Return a dummy input tensor for export flows."""

            return torch.zeros(1, self.input_dim, dtype=torch.float32)

        def save_state(self, path: str | Path) -> None:
            """Save model weights."""

            torch.save(self.state_dict(), path)

else:

    class TinyMLPPolicy:
        """Numpy fallback used when PyTorch is unavailable in the runtime."""

        def __init__(
            self,
            input_dim: int = 20,
            hidden: Sequence[int] = (32, 16),
            output_dim: int = 3,
            quantization_target: str = "int8",
            seed: int = 42,
        ) -> None:
            self.input_dim = input_dim
            self.hidden = tuple(hidden)
            self.output_dim = output_dim
            self.quantization_target = quantization_target
            self.max_deltas = np.array([1.0, 0.15, 0.5], dtype=np.float32)
            rng = np.random.default_rng(seed)
            dims = [input_dim, *self.hidden, output_dim]
            self.weights: list[np.ndarray] = []
            self.biases: list[np.ndarray] = []
            for in_dim, out_dim in zip(dims[:-1], dims[1:]):
                limit = np.sqrt(6.0 / (in_dim + out_dim))
                self.weights.append(rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32))
                self.biases.append(np.zeros(out_dim, dtype=np.float32))

        def forward(self, observation: np.ndarray) -> np.ndarray:
            """Run pure numpy forward inference."""

            x = observation.astype(np.float32)
            if x.ndim == 1:
                x = x[None, :]
            for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
                x = np.maximum(x @ weight + bias, 0.0)
            raw = x @ self.weights[-1] + self.biases[-1]
            return np.tanh(raw) * self.max_deltas

        def predict_delta(self, observation: np.ndarray) -> Action:
            """Predict one action delta and return it as an Action dataclass."""

            deltas = self.forward(np.asarray(observation, dtype=np.float32)).squeeze(0).tolist()
            return Action(
                light_lux=float(deltas[0]),
                aroma_level=float(deltas[1]),
                temp_delta_c=float(deltas[2]),
            )

        def dummy_input(self) -> np.ndarray:
            """Return a dummy input array."""

            return np.zeros((1, self.input_dim), dtype=np.float32)

        def save_state(self, path: str | Path) -> None:
            """Save fallback weights in numpy format."""

            payload = {
                f"weight_{index}": weight for index, weight in enumerate(self.weights)
            } | {f"bias_{index}": bias for index, bias in enumerate(self.biases)}
            np.savez(path, **payload)
