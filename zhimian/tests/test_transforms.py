from __future__ import annotations

import torch

from src.preprocessing.transforms import SampleTransform, StandardizeTensor


def test_standardize_tensor_preserves_shape() -> None:
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    transform = StandardizeTensor()
    transformed = transform(tensor)
    assert transformed.shape == tensor.shape
    assert torch.isfinite(transformed).all()


def test_sample_transform_tensor_types() -> None:
    sample = {
        "bcg": torch.randn(256),
        "temp": torch.randn(16, 9),
        "label": 1,
        "hr": 70.0,
        "rr": 15.0,
    }
    transform = SampleTransform(signal_standardize=True, temp_standardize=True)
    transformed = transform(sample)
    assert transformed["bcg"].shape == (256,)
    assert transformed["temp"].shape == (16, 9)
    assert transformed["label"].dtype == torch.long
    assert transformed["hr"].dtype == torch.float32
    assert transformed["rr"].dtype == torch.float32

