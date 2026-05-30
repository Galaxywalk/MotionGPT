from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .features import HumanMLFeatureConverter, recover_from_ric
from .m4human import M4HumanReader, axis_to_humanml, resample_joints_linear


KINEMATIC_CHAIN_22 = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]
CHAIN_COLORS = ["#d62728", "#1f77b4", "#222222", "#d62728", "#1f77b4"]
AXIS_MODES = ("xzy", "xz-y", "-xzy", "x-zy")


def _prepare_motion(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32).copy()
    joints[:, :, 1] -= np.nanmin(joints[:, :, 1])
    return joints


def _plot_cell(ax, joints: np.ndarray, frame_idx: int, radius: float) -> None:
    current = joints[frame_idx]
    root = current[0].copy()
    root[1] = 0.0
    pose = current.copy()
    pose[:, 0] -= root[0]
    pose[:, 2] -= root[2]
    trajectory = joints[: frame_idx + 1, 0].copy()
    trajectory[:, 0] -= root[0]
    trajectory[:, 1] = 0.0
    trajectory[:, 2] -= root[2]

    for chain, color in zip(KINEMATIC_CHAIN_22, CHAIN_COLORS):
        ax.plot3D(pose[chain, 0], pose[chain, 2], pose[chain, 1], color=color, linewidth=2.0)
    if len(trajectory) > 1:
        ax.plot3D(trajectory[:, 0], trajectory[:, 2], trajectory[:, 1], color="#666666", linewidth=1.0, alpha=0.7)
    ax.scatter([0], [0], [pose[0, 1]], color="#111111", s=8)

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(0.0, radius * 1.6)
    ax.view_init(elev=18, azim=-65)
    ax.set_box_aspect((1, 1, 1.1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)


def _motion_radius(joints_list: list[np.ndarray]) -> float:
    values = []
    for joints in joints_list:
        root = joints[:, :1, :].copy()
        root[:, :, 1] = 0.0
        centered = joints - root
        values.append(np.nanpercentile(np.abs(centered[..., [0, 2]]), 95))
        values.append(np.nanpercentile(centered[..., 1], 95) / 1.6)
    radius = float(max(values)) if values else 2.0
    return max(radius * 1.25, 1.2)


def plot_motion_overview(
    motions: list[tuple[str, np.ndarray]],
    out_path: Path,
    frames_per_motion: int = 6,
    title: str = "",
) -> None:
    prepared = [(name, _prepare_motion(joints)) for name, joints in motions]
    radius = _motion_radius([joints for _, joints in prepared])
    rows = len(prepared)
    fig = plt.figure(figsize=(frames_per_motion * 2.6, max(rows * 2.1, 2.4)), dpi=120)
    if title:
        fig.suptitle(title, fontsize=12)
    for row, (name, joints) in enumerate(prepared):
        indices = np.linspace(0, len(joints) - 1, frames_per_motion).round().astype(int)
        for col, frame_idx in enumerate(indices):
            ax = fig.add_subplot(rows, frames_per_motion, row * frames_per_motion + col + 1, projection="3d")
            _plot_cell(ax, joints, int(frame_idx), radius)
            if col == 0:
                ax.set_title(name, fontsize=8, loc="left")
            else:
                ax.set_title(f"f{int(frame_idx)}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.98 if title else 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_axis_grid(
    variants: list[tuple[str, np.ndarray]],
    out_path: Path,
    frames_per_variant: int = 6,
    title: str = "",
) -> None:
    prepared = [(name, _prepare_motion(joints)) for name, joints in variants]
    radius = _motion_radius([joints for _, joints in prepared])
    rows = len(prepared)
    fig = plt.figure(figsize=(frames_per_variant * 2.6, rows * 2.25), dpi=120)
    if title:
        fig.suptitle(title, fontsize=12)
    for row, (name, joints) in enumerate(prepared):
        indices = np.linspace(0, len(joints) - 1, frames_per_variant).round().astype(int)
        for col, frame_idx in enumerate(indices):
            ax = fig.add_subplot(rows, frames_per_variant, row * frames_per_variant + col + 1, projection="3d")
            _plot_cell(ax, joints, int(frame_idx), radius)
            if col == 0:
                ax.set_title(name, fontsize=9, loc="left")
            else:
                ax.set_title(f"f{int(frame_idx)}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _select_evenly(items: list, count: int):
    if len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    seen = set()
    selected = []
    for idx in indices:
        idx = int(idx)
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(items[idx])
    return selected[:count]


def load_motiongpt_samples(
    dataset_root: Path,
    count: int,
    max_frames: int,
    min_frames: int,
) -> list[tuple[str, np.ndarray]]:
    split_path = dataset_root / "train.txt"
    ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    candidates = []
    for motion_id in ids:
        path = dataset_root / "new_joint_vecs" / f"{motion_id}.npy"
        if not path.exists():
            continue
        features = np.load(path)
        if features.ndim == 2 and features.shape[1] == 263 and features.shape[0] >= min_frames:
            candidates.append((motion_id, path, int(features.shape[0])))
    chosen = _select_evenly(candidates, count)

    motions: list[tuple[str, np.ndarray]] = []
    for motion_id, path, _length in chosen:
        features = np.load(path).astype(np.float32, copy=False)[:max_frames]
        joints = recover_from_ric(torch.from_numpy(features).unsqueeze(0).float(), 22).squeeze(0).numpy()
        motions.append((motion_id, joints.astype(np.float32, copy=False)))
    return motions


def load_m4human_variants(
    args: argparse.Namespace,
) -> list[dict]:
    reader = M4HumanReader(
        root=Path(args.m4human_root),
        pose_source=args.pose_source,
        smplx_model_root=Path(args.smplx_model_root) if args.smplx_model_root else None,
        num_joints=22,
        normalize_z=False,
    )
    converter = HumanMLFeatureConverter(foot_threshold=args.foot_threshold, num_joints=22)
    try:
        min_raw_frames = int(np.ceil((args.min_frames + 1) * args.source_fps / args.target_fps))
        segments = reader.build_segments(
            protocol=args.protocol,
            split_id=args.split_id,
            subset=args.subset,
            min_raw_frames=min_raw_frames,
        )
        chosen = _select_evenly(segments, args.count)
        outputs = []
        for segment in chosen:
            joints_radar = reader.load_segment_joints_radar(segment)
            joints_radar = resample_joints_linear(joints_radar, args.source_fps, args.target_fps)
            joints_radar = joints_radar[: args.max_frames]
            raw_variants = []
            canonical_variants = []
            for mode in AXIS_MODES:
                transformed = axis_to_humanml(joints_radar, mode)
                raw_variants.append((mode, transformed))
                converted = converter.convert(transformed)
                recovered = recover_from_ric(
                    torch.from_numpy(converted.features).unsqueeze(0).float(),
                    22,
                ).squeeze(0).numpy()
                canonical_variants.append((mode, recovered.astype(np.float32, copy=False)))
            outputs.append(
                {
                    "id": segment.sequence_id,
                    "subject": segment.subject,
                    "action": segment.action,
                    "start_frame": segment.start_frame,
                    "end_frame": segment.end_frame,
                    "raw_variants": raw_variants,
                    "canonical_variants": canonical_variants,
                }
            )
        return outputs
    finally:
        reader.close()


def write_index(out_root: Path, motiongpt_path: Path, m4human_items: list[dict]) -> None:
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>M4Human Axis Check</title>",
        "<style>body{font-family:sans-serif;margin:24px} img{max-width:100%;border:1px solid #ddd;margin:8px 0 24px}</style>",
        "</head><body>",
        "<h1>M4Human Axis Check</h1>",
        "<h2>MotionGPT / HumanML3D reference</h2>",
        f"<img src='{motiongpt_path.name}' />",
        "<h2>M4Human raw axis transforms</h2>",
    ]
    for item in m4human_items:
        raw = f"m4human_raw/{item['id']}.png"
        canonical = f"m4human_canonical/{item['id']}.png"
        lines.extend(
            [
                f"<h3>{item['id']}</h3>",
                "<p>Raw transformed joints before HumanML3D canonicalization:</p>",
                f"<img src='{raw}' />",
                "<p>Recovered joints after HumanML3D feature conversion:</p>",
                f"<img src='{canonical}' />",
            ]
        )
    lines.extend(["</body></html>"])
    (out_root / "index.html").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_visualizations(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    motiongpt_samples = load_motiongpt_samples(
        dataset_root=Path(args.motiongpt_root),
        count=args.count,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
    )
    motiongpt_path = out_root / "motiongpt_humanml3d_reference.png"
    plot_motion_overview(
        motiongpt_samples,
        motiongpt_path,
        frames_per_motion=args.frames_per_strip,
        title="MotionGPT / HumanML3D recovered joints",
    )

    m4human_items = load_m4human_variants(args)
    for item in m4human_items:
        plot_axis_grid(
            item["raw_variants"],
            out_root / "m4human_raw" / f"{item['id']}.png",
            frames_per_variant=args.frames_per_strip,
            title=f"{item['id']} raw transformed joints",
        )
        plot_axis_grid(
            item["canonical_variants"],
            out_root / "m4human_canonical" / f"{item['id']}.png",
            frames_per_variant=args.frames_per_strip,
            title=f"{item['id']} after HumanML3D feature conversion",
        )

    manifest = {
        "motiongpt_root": str(Path(args.motiongpt_root).expanduser().resolve()),
        "m4human_root": str(Path(args.m4human_root).expanduser().resolve()),
        "out_root": str(out_root),
        "count": args.count,
        "max_frames": args.max_frames,
        "frames_per_strip": args.frames_per_strip,
        "axis_modes": list(AXIS_MODES),
        "m4human_ids": [item["id"] for item in m4human_items],
        "motiongpt_ids": [name for name, _ in motiongpt_samples],
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "pose_source": args.pose_source,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_index(out_root, motiongpt_path, m4human_items)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render MotionGPT and M4Human coordinate transform checks.")
    parser.add_argument("--motiongpt-root", default="/cpfs01/liangbo/data/MotionGPT/datasets/humanml3d")
    parser.add_argument("--m4human-root", default="/cpfs01/liangbo/widouble_workspace")
    parser.add_argument("--out-root", default="/cpfs01/liangbo/data/MotionGPT/m4human_axis_check")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=196)
    parser.add_argument("--min-frames", type=int, default=80)
    parser.add_argument("--frames-per-strip", type=int, default=6)
    parser.add_argument("--protocol", default="p1", choices=("p1", "p2", "p3"))
    parser.add_argument("--split-id", default="s2", choices=("s1", "s2", "s3"))
    parser.add_argument("--subset", default="test", choices=("train", "val", "test"))
    parser.add_argument("--pose-source", default="param_joints", choices=("param_joints", "smplx"))
    parser.add_argument("--smplx-model-root", default="/cpfs01/liangbo/widouble_workspace/models")
    parser.add_argument("--source-fps", type=float, default=10.0)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--foot-threshold", type=float, default=0.002)
    return parser


def main() -> None:
    build_visualizations(build_parser().parse_args())


if __name__ == "__main__":
    main()
