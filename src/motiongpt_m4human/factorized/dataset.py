from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils import data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class FactorizedMotionDataset(data.Dataset):
    """Window sampler for factorized root/local motion cache files."""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        window_sizes: list[int] | tuple[int, ...] = (64, 128, 196),
        window_weights: list[float] | tuple[float, ...] | None = (0.25, 0.25, 0.5),
        min_motion_length: int = 40,
        max_sequences: int = 0,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.split = split
        self.window_sizes = sorted({int(size) for size in window_sizes})
        if not self.window_sizes:
            raise ValueError("window_sizes must not be empty")
        self.window_weights = (
            [float(weight) for weight in window_weights]
            if window_weights is not None
            else None
        )
        if self.window_weights is not None and len(self.window_weights) != len(self.window_sizes):
            raise ValueError("window_weights must match window_sizes")

        rows = [
            row for row in _load_jsonl(self.cache_root / "sequences.jsonl")
            if row.get("subset") == split
            and int(row.get("num_frames", 0)) >= max(min_motion_length, min(self.window_sizes))
        ]
        if max_sequences:
            rows = rows[:max_sequences]
        if not rows:
            raise RuntimeError(f"No factorized rows for split={split} under {self.cache_root}")
        self.rows = rows
        self.eligible_by_window = {
            size: [row for row in self.rows if int(row["num_frames"]) >= size]
            for size in self.window_sizes
        }
        for size, eligible in self.eligible_by_window.items():
            if not eligible:
                raise RuntimeError(f"No sequences are at least {size} frames long")

    def __len__(self) -> int:
        return len(self.rows)

    def _pick_window_size(self) -> int:
        return random.choices(
            self.window_sizes,
            weights=self.window_weights,
            k=1,
        )[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        window_size = self._pick_window_size()
        row = self.rows[index % len(self.rows)]
        if int(row["num_frames"]) < window_size:
            row = random.choice(self.eligible_by_window[window_size])
        path = self.cache_root / row["factorized"]
        arrays = dict(np.load(path, allow_pickle=False))
        length = int(arrays["valid_mask"].shape[0])
        start = random.randint(0, length - window_size)
        end = start + window_size
        sample: dict[str, Any] = {
            "id": row["id"],
            "source_domain": row["source_domain"],
            "subset": row["subset"],
            "start": start,
            "end": end,
            "length": window_size,
        }
        for key, value in arrays.items():
            if isinstance(value, np.ndarray) and value.shape[:1] == (length,):
                sample[key] = value[start:end].astype(value.dtype, copy=False)
            else:
                sample[key] = value
        return sample
