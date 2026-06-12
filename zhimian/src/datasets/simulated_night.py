from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class SimulatedNightMetadata:
    root_dir: Path
    num_samples: int
    label_distribution: dict[int, int]
    label_mapping: dict[str, int]


class SimulatedSleepNightDataset(Dataset[dict[str, Any]]):
    """
    Dataset adapter for the observed simulated_sleep_night_u001 package.

    Alignment rule:
    - Supervised samples are created only at timestamps that exist in
      state_snapshots.csv, avoiding interpolation of labels.
    - Each sample uses a rolling window over physiology_stream.csv to construct
      a fixed-length 1D signal tensor and a temperature sequence tensor.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        signal_length: int,
        temp_seq_len: int,
        temp_grid_size: int,
        split_ratio: float,
        transform: Any | None = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be train or val, got {split}")
        self.root_dir = Path(root_dir)
        self.signal_length = signal_length
        self.temp_seq_len = temp_seq_len
        self.temp_grid_size = temp_grid_size
        self.transform = transform
        self.label_mapping = {
            "S1": 0,
            "S2": 1,
            "S3": 2,
            "S4": 3,
            "S5": 4,
            "S6": 5,
        }
        self._profile = self._read_single_row_csv("user_profiles.csv")
        self._baseline = self._read_single_row_csv("user_baselines.csv")
        self._strategy = self._read_json("personalized_strategy.json")
        self._physiology_rows = self._read_csv("physiology_stream.csv")
        self._state_rows = self._read_csv("state_snapshots.csv")
        self._physiology_by_timestamp = {
            row["timestamp"]: row for row in self._physiology_rows
        }
        self._samples = self._build_supervised_samples()
        split_index = max(1, int(len(self._samples) * split_ratio))
        if split == "train":
            self.samples = self._samples[:split_index]
        else:
            self.samples = self._samples[split_index:]
        if not self.samples:
            raise ValueError(f"No samples available for split={split}")

    def _read_csv(self, filename: str) -> list[dict[str, str]]:
        with (self.root_dir / filename).open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _read_single_row_csv(self, filename: str) -> dict[str, str]:
        rows = self._read_csv(filename)
        if len(rows) != 1:
            raise ValueError(f"{filename} must contain exactly one row")
        return rows[0]

    def _read_json(self, filename: str) -> dict[str, Any]:
        return json.loads((self.root_dir / filename).read_text(encoding="utf-8"))

    def _build_supervised_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        physiology_timestamps = [row["timestamp"] for row in self._physiology_rows]
        timestamp_to_index = {
            timestamp: index for index, timestamp in enumerate(physiology_timestamps)
        }
        for snapshot in self._state_rows:
            timestamp = snapshot["timestamp"]
            if timestamp not in self._physiology_by_timestamp:
                continue
            physiology_index = timestamp_to_index[timestamp]
            physiology_row = self._physiology_by_timestamp[timestamp]
            samples.append(
                {
                    "timestamp": timestamp,
                    "physiology_index": physiology_index,
                    "state": snapshot["state"],
                    "label": self.label_mapping[snapshot["state"]],
                    "hr": float(physiology_row["hr_bpm"]),
                    "rr": float(physiology_row["br_bpm"]),
                }
            )
        return samples

    def label_distribution(self) -> dict[int, int]:
        counts = {index: 0 for index in self.label_mapping.values()}
        for sample in self.samples:
            counts[int(sample["label"])] += 1
        return counts

    def metadata(self) -> SimulatedNightMetadata:
        return SimulatedNightMetadata(
            root_dir=self.root_dir,
            num_samples=len(self.samples),
            label_distribution=self.label_distribution(),
            label_mapping=self.label_mapping,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        physiology_index = int(sample["physiology_index"])
        bcg_tensor = self._build_signal_tensor(physiology_index)
        temp_tensor = self._build_temp_tensor(physiology_index)
        payload: dict[str, Any] = {
            "bcg": bcg_tensor,
            "temp": temp_tensor,
            "label": int(sample["label"]),
            "hr": float(sample["hr"]),
            "rr": float(sample["rr"]),
            "timestamp": sample["timestamp"],
            "state": sample["state"],
        }
        if self.transform is not None:
            payload = self.transform(payload)
        return payload

    def _build_signal_tensor(self, physiology_index: int) -> torch.Tensor:
        feature_names = [
            "hr_bpm",
            "br_bpm",
            "rmssd_ms",
            "br_irregularity",
            "stillness",
        ]
        context_window = self.signal_length // len(feature_names)
        values: list[float] = []
        baseline_hr = float(self._baseline["hr_bpm"])
        baseline_br = float(self._baseline["br_bpm"])
        baseline_rmssd = float(self._baseline["rmssd_ms"])
        baseline_stillness = float(self._baseline["stillness"])
        for offset in range(context_window):
            row_index = max(0, physiology_index - (context_window - 1 - offset))
            row = self._physiology_rows[row_index]
            hr = float(row["hr_bpm"]) / max(baseline_hr, 1.0)
            br = float(row["br_bpm"]) / max(baseline_br, 1.0)
            rmssd = float(row["rmssd_ms"]) / max(baseline_rmssd, 1.0)
            br_irregularity = float(row["br_irregularity"])
            stillness = float(row["stillness"]) / max(baseline_stillness, 1e-6)
            values.extend([hr, br, rmssd, br_irregularity, stillness])
        tensor = torch.tensor(values[: self.signal_length], dtype=torch.float32)
        if tensor.numel() < self.signal_length:
            padding = torch.zeros(self.signal_length - tensor.numel(), dtype=torch.float32)
            tensor = torch.cat([tensor, padding], dim=0)
        return tensor

    def _build_temp_tensor(self, physiology_index: int) -> torch.Tensor:
        values: list[list[float]] = []
        baseline_temp = float(self._baseline["skin_temp_c"])
        for offset in range(self.temp_seq_len):
            row_index = max(0, physiology_index - (self.temp_seq_len - 1 - offset))
            row = self._physiology_rows[row_index]
            temp_value = float(row["skin_temp_c"]) - baseline_temp
            # Observed dataset only contains scalar skin_temp_c; for MVP we
            # repeat it across the 3x3 layout placeholder expected by the model.
            values.append([temp_value] * self.temp_grid_size)
        return torch.tensor(values, dtype=torch.float32)

