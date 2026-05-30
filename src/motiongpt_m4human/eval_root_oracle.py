from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .features import motion_process, recover_from_ric
from .vqvae import DEFAULT_MEAN, DEFAULT_STD, load_vqvae, resolve_device


DEFAULT_M4HUMAN_CACHE = "/cpfs01/liangbo/data/MotionGPT/m4human_cache/v2_xz-y_param_joints_m4ref_20hz"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _filter_finite_rows(cache_root: Path, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    valid_rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for row in rows:
        features = np.load(cache_root / row["features"], mmap_mode="r")
        if np.isfinite(features).all():
            valid_rows.append(row)
        else:
            skipped.append(row["id"])
    return valid_rows, skipped


def _iter_windows(
    cache_root: Path,
    rows: list[dict[str, Any]],
    window_frames: int,
    stride: int,
    min_window_frames: int,
    include_tail: bool,
    max_windows: int,
):
    yielded = 0
    for row in rows:
        features = np.load(cache_root / row["features"]).astype(np.float32, copy=False)
        length = int(features.shape[0])
        if length < min_window_frames:
            continue

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

        for start, frames in starts:
            end = start + frames
            yield {
                "id": row["id"],
                "start": start,
                "end": end,
                "features": features[start:end],
            }
            yielded += 1
            if max_windows > 0 and yielded >= max_windows:
                return


def _case_features(ref: torch.Tensor, pred: torch.Tensor) -> dict[str, torch.Tensor]:
    cases = {
        "case0_pred_yaw_pred_vel": pred,
        "case1_gt_yaw_pred_vel": pred.clone(),
        "case2_pred_yaw_gt_vel": pred.clone(),
        "case3_gt_yaw_gt_vel": pred.clone(),
    }
    cases["case1_gt_yaw_pred_vel"][..., 0:1] = ref[..., 0:1]
    cases["case2_pred_yaw_gt_vel"][..., 1:3] = ref[..., 1:3]
    cases["case3_gt_yaw_gt_vel"][..., 0:3] = ref[..., 0:3]
    return cases


class CaseStats:
    def __init__(self) -> None:
        self.recon_sum = 0.0
        self.recon_count = 0
        self.root_sum = 0.0
        self.root_count = 0
        self.root_xz_error_sum = 0.0
        self.root_xz_error_count = 0
        self.final_xz_errors: list[float] = []
        self.path_errors: list[float] = []
        self.speed_error_sum = 0.0
        self.speed_error_count = 0

    def update(self, ref_features: torch.Tensor, case_features: torch.Tensor, fps: float) -> None:
        ref_joints = recover_from_ric(ref_features, 22)
        case_joints = recover_from_ric(case_features, 22)
        joint_err = torch.linalg.norm(case_joints - ref_joints, dim=-1)
        self.recon_sum += float(joint_err.sum().item())
        self.recon_count += int(joint_err.numel())

        ref_ra = ref_joints - ref_joints[..., :1, :]
        case_ra = case_joints - case_joints[..., :1, :]
        root_err = torch.linalg.norm(case_ra - ref_ra, dim=-1)
        self.root_sum += float(root_err.sum().item())
        self.root_count += int(root_err.numel())

        _, ref_root = motion_process.recover_root_rot_pos(ref_features)
        _, case_root = motion_process.recover_root_rot_pos(case_features)
        root_xz_err = torch.linalg.norm(
            case_root[..., [0, 2]] - ref_root[..., [0, 2]],
            dim=-1,
        )
        self.root_xz_error_sum += float(root_xz_err.sum().item())
        self.root_xz_error_count += int(root_xz_err.numel())
        self.final_xz_errors.extend(root_xz_err[..., -1].detach().cpu().tolist())

        ref_steps = torch.linalg.norm(
            ref_root[..., 1:, [0, 2]] - ref_root[..., :-1, [0, 2]],
            dim=-1,
        )
        case_steps = torch.linalg.norm(
            case_root[..., 1:, [0, 2]] - case_root[..., :-1, [0, 2]],
            dim=-1,
        )
        self.path_errors.extend((case_steps.sum(dim=-1) - ref_steps.sum(dim=-1)).detach().cpu().tolist())
        speed_errors = (case_steps - ref_steps) * fps * 1000.0
        self.speed_error_sum += float(speed_errors.sum().item())
        self.speed_error_count += int(speed_errors.numel())

    def summary(self) -> dict[str, float]:
        mpjpe = self.recon_sum / self.recon_count * 1000.0
        root_aligned = self.root_sum / self.root_count * 1000.0
        return {
            "mpjpe_mm": mpjpe,
            "root_aligned_mpjpe_mm": root_aligned,
            "root_gap_mm": mpjpe - root_aligned,
            "root_xz_mean_error_mm": self.root_xz_error_sum / self.root_xz_error_count * 1000.0,
            "final_xz_error_mm": float(np.mean(self.final_xz_errors) * 1000.0),
            "path_error_m": float(np.mean(self.path_errors)),
            "speed_bias_mm_per_s": self.speed_error_sum / self.speed_error_count,
        }


def _flush_batch(
    batch: list[dict[str, Any]],
    model,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    stats: dict[str, CaseStats],
    fps: float,
) -> None:
    features = torch.as_tensor(np.stack([item["features"] for item in batch]), device=device, dtype=torch.float32)
    norm_features = (features - mean) / std
    with torch.no_grad():
        recon_norm, _, _ = model(norm_features)
        recon_features = recon_norm * std + mean
        for name, case_features in _case_features(features, recon_features).items():
            stats[name].update(features, case_features, fps)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    cache_root = Path(args.cache_root).expanduser().resolve()
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _load_jsonl(cache_root / "sequences.jsonl")
    if args.subset != "all":
        rows = [row for row in rows if row["subset"] == args.subset]
    original_sequence_count = len(rows)
    rows, skipped_nonfinite = _filter_finite_rows(cache_root, rows)
    if not rows:
        raise RuntimeError(f"No finite rows found for subset={args.subset}")

    device = resolve_device(args.device)
    mean = torch.from_numpy(np.load(args.mean).astype(np.float32)).to(device)
    std = torch.from_numpy(np.load(args.std).astype(np.float32)).to(device)
    model, ckpt_meta = load_vqvae(Path(args.checkpoint), device, args.calibration_domain)

    stats = {
        "case0_pred_yaw_pred_vel": CaseStats(),
        "case1_gt_yaw_pred_vel": CaseStats(),
        "case2_pred_yaw_gt_vel": CaseStats(),
        "case3_gt_yaw_gt_vel": CaseStats(),
    }
    pending: dict[int, list[dict[str, Any]]] = {}
    window_count = 0
    frame_count = 0
    example_windows: list[str] = []

    def flush(length: int) -> None:
        batch = pending.get(length, [])
        if not batch:
            return
        _flush_batch(batch, model, mean, std, device, stats, args.fps)
        pending[length] = []

    for item in _iter_windows(
        cache_root=cache_root,
        rows=rows,
        window_frames=args.window_frames,
        stride=args.stride,
        min_window_frames=args.min_window_frames,
        include_tail=args.include_tail,
        max_windows=args.max_windows,
    ):
        length = int(item["features"].shape[0])
        pending.setdefault(length, []).append(item)
        window_count += 1
        frame_count += length
        if len(example_windows) < 5:
            example_windows.append(f"{item['id']}:{item['start']}-{item['end']}")
        if len(pending[length]) >= args.batch_size:
            flush(length)
        if args.log_every and window_count % args.log_every == 0:
            print(f"processed {window_count} windows, {frame_count} feature frames")

    for length in list(pending):
        flush(length)

    payload = {
        **ckpt_meta,
        "checkpoint": str(Path(args.checkpoint)),
        "cache_root": str(cache_root),
        "cache_manifest": str(manifest_path),
        "cache_axis_mode": manifest.get("axis_mode"),
        "cache_reference_mode": manifest.get("reference_mode"),
        "subset": args.subset,
        "original_sequence_count": original_sequence_count,
        "sequence_count": len(rows),
        "skipped_nonfinite_sequence_count": len(skipped_nonfinite),
        "window_frames": args.window_frames,
        "stride": args.stride,
        "include_tail": args.include_tail,
        "min_window_frames": args.min_window_frames,
        "window_count": window_count,
        "feature_frame_count": frame_count,
        "fps": args.fps,
        "device": str(device),
        "batch_size": args.batch_size,
        "example_windows": example_windows,
        "results": {name: case_stats.summary() for name, case_stats in stats.items()},
    }

    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run yaw/local-velocity oracle decomposition for M4Human cached features.")
    parser.add_argument("--cache-root", default=DEFAULT_M4HUMAN_CACHE)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mean", default=DEFAULT_MEAN)
    parser.add_argument("--std", default=DEFAULT_STD)
    parser.add_argument("--subset", default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--window-frames", type=int, default=196)
    parser.add_argument("--stride", type=int, default=196)
    parser.add_argument("--include-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-window-frames", type=int, default=40)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-domain", default="none")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--out-json", default="")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
