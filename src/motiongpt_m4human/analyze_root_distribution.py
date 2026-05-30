from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .features import motion_process


DEFAULT_M4HUMAN_CACHE = "/cpfs01/liangbo/data/MotionGPT/m4human_cache/v2_xz-y_param_joints_m4ref_20hz"
DEFAULT_OUT_ROOT = "/cpfs01/liangbo/data/MotionGPT/root_distribution_analysis/humanml3d_vs_m4human_v2_20hz"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_ids(split_file: Path) -> list[str]:
    with split_file.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    out = {f"p{q:02d}": float(np.percentile(values, q)) for q in qs}
    out["mean"] = float(values.mean())
    out["std"] = float(values.std())
    return out


def _hist(values: np.ndarray, bins: np.ndarray) -> dict[str, list[float] | list[int]]:
    counts, edges = np.histogram(values[np.isfinite(values)], bins=bins)
    return {"edges": edges.tolist(), "counts": counts.astype(int).tolist()}


def _recover_root_xz(features: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    _, root_pos = motion_process.recover_root_rot_pos(tensor)
    return root_pos[:, [0, 2]].cpu().numpy()


def _feature_y_values(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_y = features[:, 3].astype(np.float64, copy=False)
    ric = features[:, 4 : 4 + 21 * 3].reshape(features.shape[0], 21, 3)
    joint_y = np.concatenate([root_y[:, None], ric[:, :, 1].astype(np.float64, copy=False)], axis=1)
    frame_min_y = joint_y.min(axis=1)
    return root_y, frame_min_y, root_y - frame_min_y


def _window_stats(root_xz: np.ndarray, window: int, stride: int) -> tuple[list[float], list[float]]:
    path_lengths: list[float] = []
    bbox_diags: list[float] = []
    if root_xz.shape[0] < 2:
        return path_lengths, bbox_diags
    if root_xz.shape[0] < window:
        starts = [0]
    else:
        starts = list(range(0, root_xz.shape[0] - window + 1, stride))
        if starts[-1] + window < root_xz.shape[0]:
            starts.append(root_xz.shape[0] - window)
    for start in starts:
        chunk = root_xz[start : min(start + window, root_xz.shape[0])]
        if chunk.shape[0] < 2:
            continue
        steps = np.linalg.norm(np.diff(chunk, axis=0), axis=1)
        path_lengths.append(float(steps.sum()))
        bbox = chunk.max(axis=0) - chunk.min(axis=0)
        bbox_diags.append(float(np.linalg.norm(bbox)))
    return path_lengths, bbox_diags


class RootStats:
    def __init__(self, name: str) -> None:
        self.name = name
        self.sequence_count = 0
        self.frame_count = 0
        self.root_y: list[np.ndarray] = []
        self.frame_min_y: list[np.ndarray] = []
        self.root_to_frame_min_y: list[np.ndarray] = []
        self.seq_floor_min_y: list[float] = []
        self.seq_root_y_mean: list[float] = []
        self.seq_root_y_std: list[float] = []
        self.seq_root_y_range: list[float] = []
        self.seq_root_xz_path: list[float] = []
        self.seq_root_xz_bbox_diag: list[float] = []
        self.seq_root_xz_endpoint: list[float] = []
        self.seq_root_xz_mean_speed: list[float] = []
        self.window_root_xz_path: list[float] = []
        self.window_root_xz_bbox_diag: list[float] = []
        self.canonical_root_y_absdiff_max: list[float] = []
        self.canonical_min_y_absdiff_max: list[float] = []

    def add(self, features: np.ndarray, canonical_joints: np.ndarray | None = None) -> None:
        if features.ndim != 2 or features.shape[-1] != 263:
            raise ValueError(f"Expected features [T,263], got {features.shape}")
        if features.shape[0] < 2:
            return

        root_y, frame_min_y, root_to_min = _feature_y_values(features)
        root_xz = _recover_root_xz(features)
        steps = np.linalg.norm(np.diff(root_xz, axis=0), axis=1)
        bbox = root_xz.max(axis=0) - root_xz.min(axis=0)

        self.sequence_count += 1
        self.frame_count += int(features.shape[0])
        self.root_y.append(root_y)
        self.frame_min_y.append(frame_min_y)
        self.root_to_frame_min_y.append(root_to_min)
        self.seq_floor_min_y.append(float(frame_min_y.min()))
        self.seq_root_y_mean.append(float(root_y.mean()))
        self.seq_root_y_std.append(float(root_y.std()))
        self.seq_root_y_range.append(float(root_y.max() - root_y.min()))
        self.seq_root_xz_path.append(float(steps.sum()))
        self.seq_root_xz_bbox_diag.append(float(np.linalg.norm(bbox)))
        self.seq_root_xz_endpoint.append(float(np.linalg.norm(root_xz[-1] - root_xz[0])))
        self.seq_root_xz_mean_speed.append(float(steps.mean()))
        paths, bboxes = _window_stats(root_xz, window=196, stride=196)
        self.window_root_xz_path.extend(paths)
        self.window_root_xz_bbox_diag.extend(bboxes)

        if canonical_joints is not None:
            n = min(features.shape[0], canonical_joints.shape[0])
            canonical = canonical_joints[:n]
            canonical_root_y = canonical[:, 0, 1]
            canonical_min_y = canonical[:, :, 1].min(axis=1)
            self.canonical_root_y_absdiff_max.append(float(np.max(np.abs(root_y[:n] - canonical_root_y))))
            self.canonical_min_y_absdiff_max.append(float(np.max(np.abs(frame_min_y[:n] - canonical_min_y))))

    def summary(self) -> dict[str, Any]:
        root_y = np.concatenate(self.root_y) if self.root_y else np.array([])
        frame_min_y = np.concatenate(self.frame_min_y) if self.frame_min_y else np.array([])
        root_to_min = np.concatenate(self.root_to_frame_min_y) if self.root_to_frame_min_y else np.array([])
        return {
            "name": self.name,
            "sequence_count": self.sequence_count,
            "frame_count": self.frame_count,
            "root_y_frame_m": _quantiles(root_y),
            "frame_min_joint_y_m": _quantiles(frame_min_y),
            "root_to_frame_min_joint_y_m": _quantiles(root_to_min),
            "seq_floor_min_y_m": _quantiles(np.array(self.seq_floor_min_y)),
            "seq_root_y_mean_m": _quantiles(np.array(self.seq_root_y_mean)),
            "seq_root_y_std_m": _quantiles(np.array(self.seq_root_y_std)),
            "seq_root_y_range_m": _quantiles(np.array(self.seq_root_y_range)),
            "seq_root_xz_path_m": _quantiles(np.array(self.seq_root_xz_path)),
            "seq_root_xz_bbox_diag_m": _quantiles(np.array(self.seq_root_xz_bbox_diag)),
            "seq_root_xz_endpoint_m": _quantiles(np.array(self.seq_root_xz_endpoint)),
            "seq_root_xz_mean_speed_m_per_frame": _quantiles(np.array(self.seq_root_xz_mean_speed)),
            "window196_root_xz_path_m": _quantiles(np.array(self.window_root_xz_path)),
            "window196_root_xz_bbox_diag_m": _quantiles(np.array(self.window_root_xz_bbox_diag)),
            "canonical_root_y_absdiff_max_m": _quantiles(np.array(self.canonical_root_y_absdiff_max)),
            "canonical_min_y_absdiff_max_m": _quantiles(np.array(self.canonical_min_y_absdiff_max)),
        }

    def histograms(self) -> dict[str, Any]:
        root_y = np.concatenate(self.root_y) if self.root_y else np.array([])
        frame_min_y = np.concatenate(self.frame_min_y) if self.frame_min_y else np.array([])
        root_to_min = np.concatenate(self.root_to_frame_min_y) if self.root_to_frame_min_y else np.array([])
        return {
            "root_y_frame_m": _hist(root_y, np.linspace(-0.2, 1.6, 181)),
            "frame_min_joint_y_m": _hist(frame_min_y, np.linspace(-0.2, 0.8, 101)),
            "root_to_frame_min_joint_y_m": _hist(root_to_min, np.linspace(0.0, 1.6, 161)),
            "window196_root_xz_path_m": _hist(np.array(self.window_root_xz_path), np.linspace(0.0, 8.0, 161)),
            "window196_root_xz_bbox_diag_m": _hist(np.array(self.window_root_xz_bbox_diag), np.linspace(0.0, 6.0, 121)),
        }


def _humanml_items(root: Path, split: str, max_sequences: int) -> list[tuple[str, Path]]:
    ids = _read_ids(root / f"{split}.txt")
    items: list[tuple[str, Path]] = []
    for motion_id in ids:
        path = root / "new_joint_vecs" / f"{motion_id}.npy"
        if path.exists():
            items.append((motion_id, path))
    return items[:max_sequences] if max_sequences else items


def _m4human_items(cache_root: Path, split: str, max_sequences: int) -> list[dict[str, Any]]:
    rows = [row for row in _load_jsonl(cache_root / "sequences.jsonl") if row.get("subset") == split]
    return rows[:max_sequences] if max_sequences else rows


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    humanml_root = Path(args.humanml_root).expanduser().resolve()
    m4human_cache = Path(args.m4human_cache).expanduser().resolve()

    results: dict[str, Any] = {}
    histograms: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []

    for source in ("humanml3d", "m4human"):
        for split in args.splits:
            stats = RootStats(f"{source}_{split}")
            if source == "humanml3d":
                items = _humanml_items(humanml_root, split, args.max_humanml_sequences)
                for idx, (motion_id, path) in enumerate(items, 1):
                    features = np.load(path).astype(np.float32, copy=False)
                    stats.add(features)
                    if args.log_every and idx % args.log_every == 0:
                        print(f"{stats.name}: processed {idx}/{len(items)}")
            else:
                items = _m4human_items(m4human_cache, split, args.max_m4human_sequences)
                for idx, row in enumerate(items, 1):
                    features = np.load(m4human_cache / row["features"]).astype(np.float32, copy=False)
                    canonical = np.load(m4human_cache / row["canonical_joints"]).astype(np.float32, copy=False)
                    stats.add(features, canonical_joints=canonical)
                    if args.log_every and idx % args.log_every == 0:
                        print(f"{stats.name}: processed {idx}/{len(items)}")

            summary = stats.summary()
            results[stats.name] = summary
            histograms[stats.name] = stats.histograms()
            for metric, value in summary.items():
                if isinstance(value, dict) and "p50" in value:
                    csv_rows.append(
                        {
                            "source_split": stats.name,
                            "metric": metric,
                            "mean": value.get("mean"),
                            "std": value.get("std"),
                            "p01": value.get("p01"),
                            "p05": value.get("p05"),
                            "p25": value.get("p25"),
                            "p50": value.get("p50"),
                            "p75": value.get("p75"),
                            "p95": value.get("p95"),
                            "p99": value.get("p99"),
                        }
                    )

    payload = {
        "humanml_root": str(humanml_root),
        "m4human_cache": str(m4human_cache),
        "splits": args.splits,
        "max_humanml_sequences": args.max_humanml_sequences,
        "max_m4human_sequences": args.max_m4human_sequences,
        "results": results,
    }
    (out_root / "root_distribution_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_root / "root_distribution_histograms.json").write_text(
        json.dumps(histograms, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out_root / "root_distribution_quantiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_split", "metric", "mean", "std", "p01", "p05", "p25", "p50", "p75", "p95", "p99"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare HumanML3D and M4Human root/floor distributions.")
    parser.add_argument("--humanml-root", default="datasets/humanml3d")
    parser.add_argument("--m4human-cache", default=DEFAULT_M4HUMAN_CACHE)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=("train", "val", "test"))
    parser.add_argument("--max-humanml-sequences", type=int, default=0)
    parser.add_argument("--max-m4human-sequences", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1000)
    return parser


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
