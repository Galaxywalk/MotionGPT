from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .features import motion_process
from .vqvae import DEFAULT_MEAN, DEFAULT_STD, load_vqvae, resolve_device


DEFAULT_M4HUMAN_CACHE = "/cpfs01/liangbo/data/MotionGPT/m4human_cache/v2_xz-y_param_joints_m4ref_20hz"
DEFAULT_MIXED_CKPT = (
    "experiments/mgpt/VQVAE_HumanML3D_M4Human20Hz_mix30_bs256_finetune/"
    "checkpoints/min-MPJPEep=0.ckpt"
)
DEFAULT_OUT_ROOT = "/cpfs01/liangbo/data/MotionGPT/root_reconstruction_analysis/mixed500_min_mpjpe"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    out = {f"p{q:02d}": float(np.percentile(values, q)) for q in qs}
    out["mean"] = float(values.mean())
    out["std"] = float(values.std())
    return out


def _iter_humanml_features(root: Path, split: str, max_sequences: int):
    ids = _read_ids(root / f"{split}.txt")
    yielded = 0
    for motion_id in ids:
        path = root / "new_joint_vecs" / f"{motion_id}.npy"
        if not path.exists():
            continue
        yield motion_id, np.load(path).astype(np.float32, copy=False)
        yielded += 1
        if max_sequences and yielded >= max_sequences:
            break


def _iter_m4human_features(cache_root: Path, split: str, max_sequences: int):
    rows = [row for row in _load_jsonl(cache_root / "sequences.jsonl") if row.get("subset") == split]
    for idx, row in enumerate(rows):
        yield row["id"], np.load(cache_root / row["features"]).astype(np.float32, copy=False)
        if max_sequences and idx + 1 >= max_sequences:
            break


def _window_starts(length: int, window_frames: int, stride: int, min_window_frames: int, include_tail: bool):
    if length < min_window_frames:
        return []
    starts: list[tuple[int, int]] = []
    if window_frames > 0:
        for start in range(0, max(length - window_frames + 1, 0), stride):
            starts.append((start, window_frames))
        covered_end = starts[-1][0] + starts[-1][1] if starts else 0
        if include_tail and length > covered_end:
            tail_len = ((length - covered_end) // 4) * 4
            if tail_len >= min_window_frames:
                starts.append((covered_end, tail_len))
        if not starts and include_tail:
            tail_len = (length // 4) * 4
            if tail_len >= min_window_frames:
                starts.append((0, tail_len))
    else:
        full_len = (length // 4) * 4
        if full_len >= min_window_frames:
            starts.append((0, full_len))
    return starts


def _iter_windows(source: str, root: Path, split: str, args: argparse.Namespace):
    if source == "humanml3d":
        seq_iter = _iter_humanml_features(root, split, args.max_sequences)
    elif source == "m4human":
        seq_iter = _iter_m4human_features(root, split, args.max_sequences)
    else:
        raise ValueError(f"Unknown source {source}")

    for seq_id, features in seq_iter:
        for start, frames in _window_starts(
            length=int(features.shape[0]),
            window_frames=args.window_frames,
            stride=args.stride,
            min_window_frames=args.min_window_frames,
            include_tail=args.include_tail,
        ):
            yield {
                "id": seq_id,
                "start": start,
                "end": start + frames,
                "features": features[start : start + frames],
            }


class RootReconstructionStats:
    def __init__(self, source: str, split: str) -> None:
        self.source = source
        self.split = split
        self.window_count = 0
        self.frame_count = 0
        self.example_windows: list[str] = []

        self.yaw_errors: list[np.ndarray] = []
        self.root_y_errors: list[np.ndarray] = []
        self.vx_errors: list[np.ndarray] = []
        self.vz_errors: list[np.ndarray] = []
        self.linear_error_l2: list[np.ndarray] = []
        self.ref_speed: list[np.ndarray] = []
        self.recon_speed: list[np.ndarray] = []
        self.speed_errors: list[np.ndarray] = []

        self.root_xz_error: list[np.ndarray] = []
        self.root_xyz_error: list[np.ndarray] = []
        self.window_final_xz_error: list[float] = []
        self.window_mean_xz_error: list[float] = []
        self.window_max_xz_error: list[float] = []
        self.window_ref_path: list[float] = []
        self.window_recon_path: list[float] = []
        self.window_path_error: list[float] = []

    def update(self, batch: list[dict[str, Any]], features: torch.Tensor, recon: torch.Tensor) -> None:
        ref_np = features.detach().cpu().numpy()
        recon_np = recon.detach().cpu().numpy()

        yaw_err = recon_np[..., 0] - ref_np[..., 0]
        root_y_err = recon_np[..., 3] - ref_np[..., 3]
        vel_err = recon_np[..., 1:3] - ref_np[..., 1:3]
        ref_speed = np.linalg.norm(ref_np[..., 1:3], axis=-1)
        recon_speed = np.linalg.norm(recon_np[..., 1:3], axis=-1)
        linear_l2 = np.linalg.norm(vel_err, axis=-1)

        self.yaw_errors.append(yaw_err.reshape(-1))
        self.root_y_errors.append(root_y_err.reshape(-1))
        self.vx_errors.append(vel_err[..., 0].reshape(-1))
        self.vz_errors.append(vel_err[..., 1].reshape(-1))
        self.linear_error_l2.append(linear_l2.reshape(-1))
        self.ref_speed.append(ref_speed.reshape(-1))
        self.recon_speed.append(recon_speed.reshape(-1))
        self.speed_errors.append((recon_speed - ref_speed).reshape(-1))

        _, ref_root = motion_process.recover_root_rot_pos(features)
        _, recon_root = motion_process.recover_root_rot_pos(recon)
        root_delta = recon_root - ref_root
        xz_err = torch.linalg.norm(root_delta[..., [0, 2]], dim=-1).detach().cpu().numpy()
        xyz_err = torch.linalg.norm(root_delta, dim=-1).detach().cpu().numpy()
        self.root_xz_error.append(xz_err.reshape(-1))
        self.root_xyz_error.append(xyz_err.reshape(-1))

        ref_xz = ref_root[..., [0, 2]].detach().cpu().numpy()
        recon_xz = recon_root[..., [0, 2]].detach().cpu().numpy()
        for i, item in enumerate(batch):
            ref_steps = np.linalg.norm(np.diff(ref_xz[i], axis=0), axis=1)
            recon_steps = np.linalg.norm(np.diff(recon_xz[i], axis=0), axis=1)
            ref_path = float(ref_steps.sum())
            recon_path = float(recon_steps.sum())
            self.window_final_xz_error.append(float(xz_err[i, -1]))
            self.window_mean_xz_error.append(float(xz_err[i].mean()))
            self.window_max_xz_error.append(float(xz_err[i].max()))
            self.window_ref_path.append(ref_path)
            self.window_recon_path.append(recon_path)
            self.window_path_error.append(recon_path - ref_path)
            self.window_count += 1
            self.frame_count += int(ref_np.shape[1])
            if len(self.example_windows) < 5:
                self.example_windows.append(f"{item['id']}:{item['start']}-{item['end']}")

    def summary(self, fps: float) -> dict[str, Any]:
        def cat(items: list[np.ndarray]) -> np.ndarray:
            return np.concatenate(items) if items else np.array([], dtype=np.float64)

        yaw = cat(self.yaw_errors)
        root_y = cat(self.root_y_errors)
        vx = cat(self.vx_errors)
        vz = cat(self.vz_errors)
        lin = cat(self.linear_error_l2)
        ref_speed = cat(self.ref_speed)
        recon_speed = cat(self.recon_speed)
        speed_err = cat(self.speed_errors)
        xz_err = cat(self.root_xz_error)
        xyz_err = cat(self.root_xyz_error)

        yaw_deg_per_s = yaw * fps * 180.0 / math.pi
        root_y_mm = root_y * 1000.0
        vx_mm_s = vx * fps * 1000.0
        vz_mm_s = vz * fps * 1000.0
        lin_mm_s = lin * fps * 1000.0
        speed_err_mm_s = speed_err * fps * 1000.0
        ref_speed_m_s = ref_speed * fps
        recon_speed_m_s = recon_speed * fps

        return {
            "source": self.source,
            "split": self.split,
            "window_count": self.window_count,
            "frame_count": self.frame_count,
            "example_windows": self.example_windows,
            "ref_speed_m_per_s": _quantiles(ref_speed_m_s),
            "recon_speed_m_per_s": _quantiles(recon_speed_m_s),
            "speed_error_mm_per_s": _quantiles(speed_err_mm_s),
            "linear_velocity_error_mm_per_s_l2": _quantiles(lin_mm_s),
            "vx_error_mm_per_s": _quantiles(vx_mm_s),
            "vz_error_mm_per_s": _quantiles(vz_mm_s),
            "yaw_velocity_error_deg_per_s": _quantiles(yaw_deg_per_s),
            "root_y_error_mm": _quantiles(root_y_mm),
            "root_xz_position_error_mm": _quantiles(xz_err * 1000.0),
            "root_xyz_position_error_mm": _quantiles(xyz_err * 1000.0),
            "window_final_xz_error_mm": _quantiles(np.array(self.window_final_xz_error) * 1000.0),
            "window_mean_xz_error_mm": _quantiles(np.array(self.window_mean_xz_error) * 1000.0),
            "window_max_xz_error_mm": _quantiles(np.array(self.window_max_xz_error) * 1000.0),
            "window_ref_path_m": _quantiles(np.array(self.window_ref_path)),
            "window_recon_path_m": _quantiles(np.array(self.window_recon_path)),
            "window_path_error_m": _quantiles(np.array(self.window_path_error)),
        }


def _flush(
    pending: dict[int, list[dict[str, Any]]],
    length: int,
    stats: RootReconstructionStats,
    model,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> None:
    batch = pending.get(length, [])
    if not batch:
        return
    features = torch.as_tensor(np.stack([item["features"] for item in batch]), device=device, dtype=torch.float32)
    norm_features = (features - mean) / std
    with torch.no_grad():
        recon_norm, _, _ = model(norm_features)
        recon = recon_norm * std + mean
    stats.update(batch, features, recon)
    pending[length] = []


def evaluate_source_split(
    source: str,
    split: str,
    root: Path,
    args: argparse.Namespace,
    model,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    stats = RootReconstructionStats(source, split)
    pending: dict[int, list[dict[str, Any]]] = {}
    seen = 0
    for item in _iter_windows(source, root, split, args):
        length = int(item["features"].shape[0])
        pending.setdefault(length, []).append(item)
        seen += 1
        if len(pending[length]) >= args.batch_size:
            _flush(pending, length, stats, model, mean, std, device)
        if args.log_every and seen % args.log_every == 0:
            print(f"{source}_{split}: queued {seen} windows")
    for length in list(pending):
        _flush(pending, length, stats, model, mean, std, device)
    if stats.window_count == 0:
        raise RuntimeError(f"No windows for {source} split={split}")
    return stats.summary(args.fps)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    calibration_domain = args.calibration_domain
    if calibration_domain == "auto":
        calibration_domain = args.sources[0] if len(args.sources) == 1 else "none"
    model, ckpt_meta = load_vqvae(Path(args.checkpoint), device, calibration_domain)
    mean = torch.from_numpy(np.load(args.mean).astype(np.float32)).to(device)
    std = torch.from_numpy(np.load(args.std).astype(np.float32)).to(device)

    roots = {
        "humanml3d": Path(args.humanml_root).expanduser().resolve(),
        "m4human": Path(args.m4human_cache).expanduser().resolve(),
    }
    results: dict[str, Any] = {}
    for source in args.sources:
        for split in args.splits:
            key = f"{source}_{split}"
            results[key] = evaluate_source_split(source, split, roots[source], args, model, mean, std, device)

    payload = {
        **ckpt_meta,
        "checkpoint": str(Path(args.checkpoint)),
        "humanml_root": str(roots["humanml3d"]),
        "m4human_cache": str(roots["m4human"]),
        "window_frames": args.window_frames,
        "stride": args.stride,
        "include_tail": args.include_tail,
        "min_window_frames": args.min_window_frames,
        "fps": args.fps,
        "calibration_domain": calibration_domain,
        "results": results,
    }
    (out_root / "root_reconstruction_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (out_root / "root_reconstruction_quantiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_split", "metric", "mean", "std", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"],
        )
        writer.writeheader()
        for source_split, summary in results.items():
            for metric, value in summary.items():
                if isinstance(value, dict) and "p50" in value:
                    writer.writerow({"source_split": source_split, "metric": metric, **value})

    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure VQVAE root velocity and integrated root reconstruction errors.")
    parser.add_argument("--checkpoint", default=DEFAULT_MIXED_CKPT)
    parser.add_argument("--humanml-root", default="datasets/humanml3d")
    parser.add_argument("--m4human-cache", default=DEFAULT_M4HUMAN_CACHE)
    parser.add_argument("--mean", default=DEFAULT_MEAN)
    parser.add_argument("--std", default=DEFAULT_STD)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--sources", nargs="+", default=["humanml3d", "m4human"], choices=("humanml3d", "m4human"))
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=("train", "val", "test"))
    parser.add_argument("--window-frames", type=int, default=196)
    parser.add_argument("--stride", type=int, default=196)
    parser.add_argument("--include-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-window-frames", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-domain", default="auto")
    parser.add_argument("--log-every", type=int, default=500)
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
