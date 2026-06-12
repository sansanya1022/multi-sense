from __future__ import annotations

from typing import Any

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.datasets.simulated_night import SimulatedSleepNightDataset
from src.datasets.synthetic import SyntheticSleepDataset
from src.preprocessing.transforms import SampleTransform


def build_dataloaders(config: DictConfig) -> tuple[DataLoader[Any], DataLoader[Any]]:
    dataset_cfg = config.dataset
    dataset_kind = str(dataset_cfg.get("kind", "synthetic_sleep"))
    transform = SampleTransform(
        signal_standardize=bool(config.preprocessing.signal_standardize),
        temp_standardize=bool(config.preprocessing.temp_standardize),
    )
    if dataset_kind == "synthetic_sleep":
        train_dataset = SyntheticSleepDataset(
            size=int(dataset_cfg.train_size),
            signal_length=int(dataset_cfg.signal_length),
            temp_seq_len=int(dataset_cfg.temp_seq_len),
            temp_grid_size=int(dataset_cfg.temp_grid_size),
            num_classes=int(dataset_cfg.num_classes),
            hr_range=tuple(dataset_cfg.hr_range),
            rr_range=tuple(dataset_cfg.rr_range),
            noise_std=float(dataset_cfg.noise_std),
            seed=int(config.run.seed),
            transform=transform,
        )
        val_dataset = SyntheticSleepDataset(
            size=int(dataset_cfg.val_size),
            signal_length=int(dataset_cfg.signal_length),
            temp_seq_len=int(dataset_cfg.temp_seq_len),
            temp_grid_size=int(dataset_cfg.temp_grid_size),
            num_classes=int(dataset_cfg.num_classes),
            hr_range=tuple(dataset_cfg.hr_range),
            rr_range=tuple(dataset_cfg.rr_range),
            noise_std=float(dataset_cfg.noise_std),
            seed=int(config.run.seed) + 1,
            transform=transform,
        )
    elif dataset_kind == "simulated_sleep_night":
        train_dataset = SimulatedSleepNightDataset(
            root_dir=str(dataset_cfg.root_dir),
            split="train",
            signal_length=int(dataset_cfg.signal_length),
            temp_seq_len=int(dataset_cfg.temp_seq_len),
            temp_grid_size=int(dataset_cfg.temp_grid_size),
            split_ratio=float(dataset_cfg.train_split_ratio),
            transform=transform,
        )
        val_dataset = SimulatedSleepNightDataset(
            root_dir=str(dataset_cfg.root_dir),
            split="val",
            signal_length=int(dataset_cfg.signal_length),
            temp_seq_len=int(dataset_cfg.temp_seq_len),
            temp_grid_size=int(dataset_cfg.temp_grid_size),
            split_ratio=float(dataset_cfg.train_split_ratio),
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset kind: {dataset_kind}")
    batch_size = int(config.optimization.batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config.run.num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.run.num_workers),
    )
    return train_loader, val_loader


def build_adaptation_dataloaders(
    config: DictConfig,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    transform = SampleTransform(
        signal_standardize=bool(config.preprocessing.signal_standardize),
        temp_standardize=bool(config.preprocessing.temp_standardize),
    )
    source_cfg = config.dataset.source
    target_cfg = config.dataset.target
    source_train = _build_single_dataset(source_cfg, transform, split="train", seed=int(config.run.seed))
    target_train = _build_single_dataset(target_cfg, transform, split="train", seed=int(config.run.seed) + 1)
    val_dataset = _build_single_dataset(target_cfg, transform, split="val", seed=int(config.run.seed) + 2)
    batch_size = int(config.optimization.batch_size)
    return (
        DataLoader(source_train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(target_train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def _build_single_dataset(
    dataset_cfg: DictConfig,
    transform: SampleTransform,
    split: str,
    seed: int,
) -> Any:
    dataset_kind = str(dataset_cfg.get("kind", "synthetic_sleep"))
    if dataset_kind == "synthetic_sleep":
        return SyntheticSleepDataset(
            size=int(dataset_cfg.train_size if split == "train" else dataset_cfg.val_size),
            signal_length=int(dataset_cfg.signal_length),
            temp_seq_len=int(dataset_cfg.temp_seq_len),
            temp_grid_size=int(dataset_cfg.temp_grid_size),
            num_classes=int(dataset_cfg.num_classes),
            hr_range=tuple(dataset_cfg.hr_range),
            rr_range=tuple(dataset_cfg.rr_range),
            noise_std=float(dataset_cfg.noise_std),
            seed=seed,
            transform=transform,
        )
    if dataset_kind == "simulated_sleep_night":
        return SimulatedSleepNightDataset(
            root_dir=str(dataset_cfg.root_dir),
            split=split,
            signal_length=int(dataset_cfg.signal_length),
            temp_seq_len=int(dataset_cfg.temp_seq_len),
            temp_grid_size=int(dataset_cfg.temp_grid_size),
            split_ratio=float(dataset_cfg.train_split_ratio),
            transform=transform,
        )
    raise ValueError(f"Unsupported dataset kind: {dataset_kind}")
