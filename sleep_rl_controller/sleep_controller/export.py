"""Export helpers for edge deployment handoff."""

from __future__ import annotations

from pathlib import Path

import torch

from sleep_controller.tiny_model import TinyMLPPolicy


def export_torchscript(model: TinyMLPPolicy, path: str | Path) -> None:
    """Export the tiny policy as TorchScript."""

    model.eval()
    scripted = torch.jit.trace(model, model.dummy_input())
    scripted.save(str(path))


def export_onnx(model: TinyMLPPolicy, path: str | Path) -> None:
    """Export the tiny policy to ONNX."""

    model.eval()
    torch.onnx.export(
        model,
        model.dummy_input(),
        str(path),
        input_names=["observation"],
        output_names=["delta_action"],
        opset_version=12,
    )


def export_int8_placeholder(model: TinyMLPPolicy, path: str | Path) -> None:
    """Write a placeholder note for later ESP-DL int8 export integration."""

    message = "\n".join(
        [
            "INT8 export placeholder",
            f"model_input_dim={model.input_dim}",
            f"hidden_layers={list(model.hidden)}",
            f"output_dim={model.output_dim}",
            "Next step: calibrate representative observations, quantize weights to int8,",
            "and map the MLP structure to ESP-DL runtime operators on ESP32-S3.",
        ]
    )
    Path(path).write_text(message, encoding="utf-8")

