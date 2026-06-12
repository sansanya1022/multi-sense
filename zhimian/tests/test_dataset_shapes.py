from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.datasets.registry import build_dataloaders
from src.datasets.synthetic import SyntheticSleepDataset


def test_synthetic_dataset_item_shapes() -> None:
    dataset = SyntheticSleepDataset(
        size=8,
        signal_length=256,
        temp_seq_len=16,
        temp_grid_size=9,
        num_classes=4,
        hr_range=(52.0, 86.0),
        rr_range=(10.0, 22.0),
        noise_std=0.05,
        seed=123,
    )
    sample = dataset[0]
    assert sample["bcg"].shape == (256,)
    assert sample["temp"].shape == (16, 9)
    assert isinstance(sample["label"], int)
    assert isinstance(sample["hr"], float)
    assert isinstance(sample["rr"], float)


def test_synthetic_label_distribution_is_balanced() -> None:
    dataset = SyntheticSleepDataset(
        size=20,
        signal_length=128,
        temp_seq_len=8,
        temp_grid_size=9,
        num_classes=4,
        hr_range=(52.0, 86.0),
        rr_range=(10.0, 22.0),
        noise_std=0.05,
        seed=123,
    )
    distribution = dataset.label_distribution()
    assert set(distribution.keys()) == {0, 1, 2, 3}
    assert min(distribution.values()) >= 5


def test_dataloader_batch_shapes() -> None:
    config = OmegaConf.create(
        {
            "run": {"seed": 123, "num_workers": 0},
            "dataset": {
                "train_size": 16,
                "val_size": 8,
                "signal_length": 256,
                "temp_seq_len": 16,
                "temp_grid_size": 9,
                "num_classes": 4,
                "hr_range": [52.0, 86.0],
                "rr_range": [10.0, 22.0],
                "noise_std": 0.05,
            },
            "preprocessing": {"signal_standardize": True, "temp_standardize": True},
            "optimization": {"batch_size": 4},
        }
    )
    train_loader, _ = build_dataloaders(config)
    batch = next(iter(train_loader))
    assert batch["bcg"].shape == (4, 256)
    assert batch["temp"].shape == (4, 16, 9)
    assert batch["label"].shape == (4,)
    assert batch["hr"].shape == (4,)
    assert batch["rr"].shape == (4,)
    assert batch["label"].dtype == torch.long

