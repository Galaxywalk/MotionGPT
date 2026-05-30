from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .recover import roundtrip_mpjpe_mm
from .representation import features_to_factorized


DEFAULT_M4HUMAN_CACHE = "/cpfs01/liangbo/data/MotionGPT/m4human_cache/v2_xz-y_param_joints_m4ref_20hz"
DEFAULT_OUT_ROOT = "/cpfs01/liangbo/data/MotionGPT/factorized_cache/v1_m4human_xz-y_20hz"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _quantiles(values: Iterable[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    out = {f"p{q:02d}": float(np.percentile(values, q)) for q in qs}
    out["mean"] = float(values.mean())
    out["std"] = float(values.std())
    return out


def _iter_m4human(cache_root: Path, splits: set[str], max_sequences: int):
    rows = _load_jsonl(cache_root / "sequences.jsonl")
    yielded = 0
    for row in rows:
        if splits and row.get("subset") not in splits:
            continue
        features = np.load(cache_root / row["features"]).astype(np.float32, copy=False)
        yield {
            "id": row["id"],
            "subset": row["subset"],
            "source_domain": "m4human",
            "features": features,
            "input_row": row,
        }
        yielded += 1
        if max_sequences and yielded >= max_sequences:
            return


def _iter_humanml(root: Path, splits: set[str], max_sequences: int):
    yielded = 0
    for split in sorted(splits or {"train", "val", "test"}):
        for motion_id in _read_ids(root / f"{split}.txt"):
            path = root / "new_joint_vecs" / f"{motion_id}.npy"
            if not path.exists():
                continue
            yield {
                "id": motion_id,
                "subset": split,
                "source_domain": "humanml3d",
                "features": np.load(path).astype(np.float32, copy=False),
                "input_row": {"features": str(path)},
            }
            yielded += 1
            if max_sequences and yielded >= max_sequences:
                return


def _safe_name(source_domain: str, subset: str, seq_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in seq_id)
    return f"{source_domain}_{subset}_{safe}.npz"


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root).expanduser().resolve()
    seq_dir = out_root / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    splits = set(args.splits)
    sources = []
    if "m4human" in args.sources:
        sources.append(_iter_m4human(
            Path(args.m4human_cache).expanduser().resolve(),
            splits,
            args.max_sequences,
        ))
    if "humanml3d" in args.sources:
        sources.append(_iter_humanml(
            Path(args.humanml_root).expanduser().resolve(),
            splits,
            args.max_sequences,
        ))

    rows: list[dict[str, Any]] = []
    roundtrip_errors: list[float] = []
    skipped: list[dict[str, str]] = []
    frame_count = 0

    for iterator in sources:
        for idx, item in enumerate(iterator, 1):
            features = item["features"]
            try:
                arrays = features_to_factorized(
                    features,
                    fps=args.fps,
                    source_domain=item["source_domain"],
                )
                error = roundtrip_mpjpe_mm(features, arrays)
            except Exception as exc:
                skipped.append({"id": item["id"], "reason": str(exc)})
                continue

            rel_path = Path("sequences") / _safe_name(
                item["source_domain"],
                item["subset"],
                item["id"],
            )
            np.savez_compressed(seq_dir / rel_path.name, **arrays)
            row = {
                "id": item["id"],
                "subset": item["subset"],
                "source_domain": item["source_domain"],
                "factorized": str(rel_path),
                "num_frames": int(features.shape[0]),
                "dt": float(1.0 / args.fps),
                "fps": float(args.fps),
                "roundtrip_mpjpe_mm": error,
                "input": item["input_row"],
            }
            rows.append(row)
            roundtrip_errors.append(error)
            frame_count += int(features.shape[0])
            if args.log_every and len(rows) % args.log_every == 0:
                print(f"wrote {len(rows)} sequences, {frame_count} frames")

    with (out_root / "sequences.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = {
        "version": 1,
        "description": "Root/local factorized cache derived from MotionGPT HumanML3D-style 263-D features.",
        "sources": args.sources,
        "splits": args.splits,
        "fps": float(args.fps),
        "dt": float(1.0 / args.fps),
        "coordinate_policy": {
            "m4human_axis_mode": args.m4human_axis_mode,
            "world_up_axis": "y",
            "ground_axes": ["x", "z"],
            "root_velocity_unit": "m/s",
            "root_yaw_velocity_unit": "rad/s",
        },
        "fields": {
            "local_joints": "[T,22,3] root-relative yaw-canonical joints",
            "local_joint_vel": "[T,22,3] local-frame joint velocities from 263-D feature",
            "local_rot6d": "[T,21,6] local joint rotations excluding root",
            "contacts": "[T,4] HumanML3D foot contact channels",
            "root_xy": "[T,2] recovered global ground-plane root position in meters",
            "root_yaw": "[T,1] recovered heading in MotionGPT yaw convention, radians",
            "root_height": "[T,1] root height in meters",
            "root_vel_local_mps": "[T,2] local x/z root velocity, m/s",
            "root_vel_global_mps": "[T,2] global x/z root velocity, m/s",
            "root_yaw_vel_radps": "[T,1] yaw velocity, rad/s",
            "dt": "scalar seconds per frame",
            "source_domain": "string domain label",
            "valid_mask": "[T] bool",
            "features_263": "[T,263] original feature kept for compatibility checks",
        },
        "sequence_count": len(rows),
        "frame_count": frame_count,
        "skipped": skipped,
        "roundtrip_mpjpe_mm": _quantiles(np.asarray(roundtrip_errors)),
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build root/local factorized motion cache.")
    parser.add_argument("--m4human-cache", default=DEFAULT_M4HUMAN_CACHE)
    parser.add_argument("--humanml-root", default="datasets/humanml3d")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--sources", nargs="+", default=["m4human"], choices=("m4human", "humanml3d"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=("train", "val", "test"))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--m4human-axis-mode", default="xz-y")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200)
    return parser


def main() -> None:
    build_cache(build_parser().parse_args())


if __name__ == "__main__":
    main()
