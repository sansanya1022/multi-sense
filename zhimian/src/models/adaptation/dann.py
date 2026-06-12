from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx: object, tensor: torch.Tensor, lambda_: float) -> torch.Tensor:
        setattr(ctx, "lambda_", lambda_)
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        lambda_ = getattr(ctx, "lambda_")
        return -lambda_ * grad_output, None


class DomainClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor, lambda_: float) -> torch.Tensor:
        reversed_features = GradientReversalFunction.apply(features, lambda_)
        return self.classifier(reversed_features)

