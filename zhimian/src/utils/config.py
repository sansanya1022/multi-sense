from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def parse_config() -> DictConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=output_path)

