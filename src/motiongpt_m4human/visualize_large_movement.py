from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .m4human import M4HumanReader, axis_to_humanml, resample_joints_linear
from .visualize_coordinate_check import plot_motion_overview


def _window_starts(feature_frames: int, window_frames: int, stride: int, min_window_frames: int, include_tail: bool):
    starts: list[tuple[int, int]] = []
    if window_frames > 0:
        for start in range(0, max(feature_frames - window_frames + 1, 0), stride):
            starts.append((start, window_frames))
        covered_end = starts[-1][0] + starts[-1][1] if starts else 0
        if include_tail and feature_frames > covered_end:
            tail_len = ((feature_frames - covered_end) // 4) * 4
            if tail_len >= min_window_frames:
                starts.append((covered_end, tail_len))
        if not starts and include_tail:
            tail_len = (feature_frames // 4) * 4
            if tail_len >= min_window_frames:
                starts.append((0, tail_len))
    else:
        full_len = (feature_frames // 4) * 4
        if full_len >= min_window_frames:
            starts.append((0, full_len))
    return starts


def _movement_metrics(joints: np.ndarray) -> dict[str, float]:
    root_xz = joints[:, 0, [0, 2]]
    deltas = np.diff(root_xz, axis=0)
    path_length = float(np.linalg.norm(deltas, axis=-1).sum()) if len(deltas) else 0.0
    displacement = float(np.linalg.norm(root_xz[-1] - root_xz[0])) if len(root_xz) else 0.0
    bounds = root_xz.max(axis=0) - root_xz.min(axis=0)
    range_diag = float(np.linalg.norm(bounds))
    return {
        "root_path_length_m": path_length,
        "root_displacement_m": displacement,
        "root_range_m": range_diag,
        "root_range_x_m": float(bounds[0]),
        "root_range_z_m": float(bounds[1]),
    }


def find_large_movement(args: argparse.Namespace) -> tuple[list[dict], list[tuple[str, np.ndarray]]]:
    reader = M4HumanReader(
        root=Path(args.m4human_root),
        pose_source=args.pose_source,
        smplx_model_root=Path(args.smplx_model_root) if args.smplx_model_root else None,
        num_joints=22,
        normalize_z=False,
    )
    rows: list[dict] = []
    motions: dict[str, np.ndarray] = {}
    try:
        subsets = ("train", "val", "test") if args.subset == "all" else (args.subset,)
        for subset in subsets:
            min_raw_frames = int(np.ceil((args.min_window_frames + 1) * args.source_fps / args.target_fps))
            segments = reader.build_segments(
                protocol=args.protocol,
                split_id=args.split_id,
                subset=subset,
                min_raw_frames=min_raw_frames,
            )
            for segment in segments:
                joints_radar = reader.load_segment_joints_radar(segment)
                joints_radar = resample_joints_linear(joints_radar, args.source_fps, args.target_fps)
                joints_hml = axis_to_humanml(joints_radar, args.axis_mode)
                feature_frames = joints_hml.shape[0] - 1
                for start, frames in _window_starts(
                    feature_frames=feature_frames,
                    window_frames=args.window_frames,
                    stride=args.stride,
                    min_window_frames=args.min_window_frames,
                    include_tail=args.include_tail,
                ):
                    clip = joints_hml[start : start + frames]
                    metrics = _movement_metrics(clip)
                    clip_id = f"{segment.sequence_id}_w{start:05d}_{start + frames:05d}"
                    row = {
                        "id": clip_id,
                        "sequence_id": segment.sequence_id,
                        "subset": subset,
                        "subject": segment.subject,
                        "action": segment.action,
                        "segment_start_frame": segment.start_frame,
                        "segment_end_frame": segment.end_frame,
                        "window_start": start,
                        "window_end": start + frames,
                        "window_frames": frames,
                        **metrics,
                    }
                    rows.append(row)
                    motions[clip_id] = clip.astype(np.float32, copy=False)
    finally:
        reader.close()

    sort_key = args.rank_by
    rows.sort(key=lambda row: float(row[sort_key]), reverse=True)
    selected_rows = rows[: args.count]
    selected_motions = [(f"{row['id']}\nrange={row['root_range_m']:.2f}m path={row['root_path_length_m']:.2f}m", motions[row["id"]]) for row in selected_rows]
    return selected_rows, selected_motions


def build_visualization(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows, motions = find_large_movement(args)
    (out_root / "top_cases.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    image_path = out_root / "top_large_movement.png"
    plot_motion_overview(
        motions,
        image_path,
        frames_per_motion=args.frames_per_strip,
        title=f"M4Human top {len(rows)} root movement clips ({args.axis_mode}, {args.target_fps:g}Hz)",
    )
    html = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'><title>M4Human Large Movement</title>",
            "<style>body{font-family:sans-serif;margin:24px} img{max-width:100%;border:1px solid #ddd} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:4px 8px}</style>",
            "</head><body>",
            "<h1>M4Human Large Movement Clips</h1>",
            "<img src='top_large_movement.png' />",
            "<h2>Top cases</h2>",
            "<table><tr><th>rank</th><th>id</th><th>range m</th><th>path m</th><th>disp m</th><th>frames</th></tr>",
            *[
                (
                    f"<tr><td>{idx + 1}</td><td>{row['id']}</td><td>{row['root_range_m']:.3f}</td>"
                    f"<td>{row['root_path_length_m']:.3f}</td><td>{row['root_displacement_m']:.3f}</td>"
                    f"<td>{row['window_frames']}</td></tr>"
                )
                for idx, row in enumerate(rows)
            ],
            "</table>",
            "</body></html>",
        ]
    )
    (out_root / "index.html").write_text(html + "\n", encoding="utf-8")
    manifest = {
        "out_root": str(out_root),
        "image": str(image_path),
        "top_cases_json": str(out_root / "top_cases.json"),
        "count": args.count,
        "rank_by": args.rank_by,
        "axis_mode": args.axis_mode,
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "window_frames": args.window_frames,
        "stride": args.stride,
        "include_tail": args.include_tail,
        "min_window_frames": args.min_window_frames,
        "subset": args.subset,
        "pose_source": args.pose_source,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": manifest, "top_cases": rows[: min(10, len(rows))]}, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find and render M4Human clips with large root movement.")
    parser.add_argument("--m4human-root", default="/cpfs01/liangbo/widouble_workspace")
    parser.add_argument("--out-root", default="/cpfs01/liangbo/data/MotionGPT/m4human_large_movement")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--rank-by", default="root_range_m", choices=("root_range_m", "root_path_length_m", "root_displacement_m"))
    parser.add_argument("--frames-per-strip", type=int, default=6)
    parser.add_argument("--protocol", default="p1", choices=("p1", "p2", "p3"))
    parser.add_argument("--split-id", default="s2", choices=("s1", "s2", "s3"))
    parser.add_argument("--subset", default="all", choices=("train", "val", "test", "all"))
    parser.add_argument("--pose-source", default="param_joints", choices=("param_joints", "smplx"))
    parser.add_argument("--smplx-model-root", default="/cpfs01/liangbo/widouble_workspace/models")
    parser.add_argument("--axis-mode", default="xz-y", choices=("xzy", "xz-y", "-xzy", "x-zy"))
    parser.add_argument("--source-fps", type=float, default=10.0)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--window-frames", type=int, default=196)
    parser.add_argument("--stride", type=int, default=196)
    parser.add_argument("--include-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-window-frames", type=int, default=40)
    return parser


def main() -> None:
    build_visualization(build_parser().parse_args())


if __name__ == "__main__":
    main()
