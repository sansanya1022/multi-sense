from __future__ import annotations

from typing import Any, Protocol


class SleepSample(dict[str, Any]):
    """Typed dictionary-like sample container."""


class SleepDatasetProtocol(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> SleepSample: ...

