from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .features import HumanMLFeatureConverter
from .m4human import M4HumanReader, VALID_SUBSETS, MotionSegment, axis_to_humanml, resample_joints_linear


DEFAULT_OUT_ROOT = "/cpfs01/liangbo/data/MotionGPT/m4human_cache/v2_xz-y_param_joints_m4ref_20hz"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _segment_row(
    segment: MotionSegment,
    out_root: Path,
    feature_path: Path,
    canonical_path: Path,
    joints_radar_path: Path | None,
    feature_frames: int,
) -> dict[str, Any]:
    row = asdict(segment)
    row.pop("indicators")
    row.update(
        {
            "id": segment.sequence_id,
            "raw_frames": segment.raw_frames,
            "resampled_frames": feature_frames + 1,
            "feature_frames": feature_frames,
            "features": _relative(feature_path, out_root),
            "canonical_joints": _relative(canonical_path, out_root),
        }
    )
    if joints_radar_path is not None:
        row["joints_radar"] = _relative(joints_radar_path, out_root)
    return row


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root).expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_root} already exists and is not empty; pass --overwrite to replace files")
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)

    features_dir = out_root / "features"
    canonical_dir = out_root / "canonical_joints"
    joints_radar_dir = out_root / "joints_radar"
    meta_dir = out_root / "meta"
    for path in (features_dir, canonical_dir, meta_dir):
        path.mkdir(parents=True, exist_ok=True)
    if args.save_joints_radar:
        joints_radar_dir.mkdir(parents=True, exist_ok=True)

    subsets = list(VALID_SUBSETS) if args.subset == "all" else [args.subset]
    reader = M4HumanReader(
        root=Path(args.m4human_root),
        pose_source=args.pose_source,
        smplx_model_root=Path(args.smplx_model_root) if args.smplx_model_root else None,
        num_joints=args.num_joints,
        normalize_z=args.normalize_z,
    )
    converter = HumanMLFeatureConverter(
        reference_joints=Path(args.reference_joints) if args.reference_joints else None,
        foot_threshold=args.foot_threshold,
        num_joints=args.num_joints,
    )

    sequence_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    processed = 0
    total_raw_frames = 0
    total_feature_frames = 0

    try:
        for subset in subsets:
            segments = reader.build_segments(
                protocol=args.protocol,
                split_id=args.split_id,
                subset=subset,
                min_raw_frames=args.min_feature_frames + 1,
            )
            for segment in segments:
                if args.max_sequences and processed >= args.max_sequences:
                    break
                try:
                    joints_radar = reader.load_segment_joints_radar(segment)
                    if not np.isfinite(joints_radar).all():
                        raise ValueError("non-finite joints_radar")
                    if args.resample_method == "linear_joints":
                        joints_radar = resample_joints_linear(
                            joints_radar,
                            source_fps=args.source_fps,
                            target_fps=args.target_fps,
                        )
                    joints_hml = axis_to_humanml(joints_radar, args.axis_mode)
                    converted = converter.convert(joints_hml)
                    if converted.features.shape[0] < args.min_feature_frames:
                        continue
                    if not np.isfinite(converted.features).all():
                        raise ValueError("non-finite MotionGPT features")
                    if not np.isfinite(converted.canonical_joints).all():
                        raise ValueError("non-finite canonical joints")

                    feature_path = features_dir / f"{segment.sequence_id}.npy"
                    canonical_path = canonical_dir / f"{segment.sequence_id}.npy"
                    joints_path = joints_radar_dir / f"{segment.sequence_id}.npy" if args.save_joints_radar else None
                    np.save(feature_path, converted.features)
                    np.save(canonical_path, converted.canonical_joints)
                    if joints_path is not None:
                        np.save(joints_path, joints_radar.astype(np.float32, copy=False))

                    sequence_rows.append(
                        _segment_row(
                            segment=segment,
                            out_root=out_root,
                            feature_path=feature_path,
                            canonical_path=canonical_path,
                            joints_radar_path=joints_path,
                            feature_frames=int(converted.features.shape[0]),
                        )
                    )
                    processed += 1
                    total_raw_frames += int(segment.raw_frames)
                    total_feature_frames += int(converted.features.shape[0])
                    if args.log_every and processed % args.log_every == 0:
                        print(f"processed {processed} sequences, {total_feature_frames} feature frames")
                except Exception as exc:
                    error_rows.append(
                        {
                            "id": segment.sequence_id,
                            "subset": segment.subset,
                            "subject": segment.subject,
                            "action": segment.action,
                            "start_frame": segment.start_frame,
                            "end_frame": segment.end_frame,
                            "error": repr(exc),
                        }
                    )
                    if args.fail_fast:
                        raise
            if args.max_sequences and processed >= args.max_sequences:
                break
    finally:
        reader.close()

    _write_jsonl(out_root / "sequences.jsonl", sequence_rows)
    _write_jsonl(out_root / "errors.jsonl", error_rows)

    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "motiongpt_git_commit": _git_commit(),
        "m4human_root": str(Path(args.m4human_root).expanduser().resolve()),
        "cache_path": str(reader.cache_path),
        "out_root": str(out_root),
        "protocol": args.protocol,
        "split_id": args.split_id,
        "subsets": subsets,
        "axis_mode": args.axis_mode,
        "pose_source": args.pose_source,
        "smplx_model_root": str(reader.smplx_model_root),
        "reference_joints": str(Path(args.reference_joints).expanduser().resolve()) if args.reference_joints else "",
        "reference_mode": "file" if args.reference_joints else "first_converted_sequence_pose",
        "num_joints": args.num_joints,
        "normalize_z": args.normalize_z,
        "foot_threshold": args.foot_threshold,
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "resample_method": args.resample_method,
        "min_feature_frames": args.min_feature_frames,
        "save_joints_radar": args.save_joints_radar,
        "sequence_count": len(sequence_rows),
        "error_count": len(error_rows),
        "raw_frame_count": total_raw_frames,
        "feature_frame_count": total_feature_frames,
        "features_dir": "features",
        "canonical_joints_dir": "canonical_joints",
        "joints_radar_dir": "joints_radar" if args.save_joints_radar else "",
        "sequences_jsonl": "sequences.jsonl",
        "errors_jsonl": "errors.jsonl",
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (meta_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a MotionGPT feature cache from M4Human LMDB annotations.")
    parser.add_argument("--m4human-root", default="/cpfs01/liangbo/widouble_workspace")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--protocol", default="p1", choices=("p1", "p2", "p3"))
    parser.add_argument("--split-id", default="s2", choices=("s1", "s2", "s3"))
    parser.add_argument("--subset", default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--axis-mode", default="xz-y", choices=("xzy", "xz-y", "-xzy", "x-zy"))
    parser.add_argument("--pose-source", default="param_joints", choices=("param_joints", "smplx"))
    parser.add_argument("--smplx-model-root", default="/cpfs01/liangbo/widouble_workspace/models")
    parser.add_argument("--reference-joints", default="")
    parser.add_argument("--num-joints", type=int, default=22)
    parser.add_argument("--normalize-z", action="store_true")
    parser.add_argument("--foot-threshold", type=float, default=0.002)
    parser.add_argument("--source-fps", type=float, default=10.0)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--resample-method", default="linear_joints", choices=("linear_joints",))
    parser.add_argument("--min-feature-frames", type=int, default=40)
    parser.add_argument("--save-joints-radar", action="store_true")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    build_cache(build_parser().parse_args())


if __name__ == "__main__":
    main()
