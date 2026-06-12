from __future__ import annotations

from pathlib import Path

from src.datasets.simulated_night import SimulatedSleepNightDataset


DATA_ROOT = Path("data/raw/simulated_sleep_night_u001")


def test_simulated_night_dataset_reads_observed_structure() -> None:
    dataset = SimulatedSleepNightDataset(
        root_dir=DATA_ROOT,
        split="train",
        signal_length=30,
        temp_seq_len=16,
        temp_grid_size=9,
        split_ratio=0.8,
    )
    metadata = dataset.metadata()
    assert metadata.num_samples > 0
    assert "S1" in metadata.label_mapping


def test_simulated_night_sample_shapes_and_labels() -> None:
    dataset = SimulatedSleepNightDataset(
        root_dir=DATA_ROOT,
        split="train",
        signal_length=30,
        temp_seq_len=16,
        temp_grid_size=9,
        split_ratio=0.8,
    )
    sample = dataset[0]
    assert sample["bcg"].shape == (30,)
    assert sample["temp"].shape == (16, 9)
    assert isinstance(sample["label"], int)
    assert sample["state"] in {"S1", "S2", "S3", "S4", "S5", "S6"}
