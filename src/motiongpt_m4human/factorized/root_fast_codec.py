from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.fft import dct, idct

from ..features import motion_process, recover_from_ric
from .local_vq import DEFAULT_CACHE_ROOT, FactorizedLocalStore
from .root_branch import _features_with_root_controls, _root_controls_from_features


DEFAULT_OUT_DIR = "/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_dct"
ROOT_CONTROL_DIM = 4


def dct_root_control_coefficients(
    root_controls: np.ndarray,
    chunk_size: int,
    coeff_count: int,
) -> tuple[np.ndarray, int]:
    """Return low-frequency DCT coefficients for chunked root controls.

    Args:
        root_controls: Array with shape [B, T, 4] in physical root-control
            units: yaw rad/s, local vx m/s, local vz m/s, root height m.
        chunk_size: Number of frames per DCT chunk.
        coeff_count: Number of low-frequency DCT coefficients to keep per
            chunk and root-control dimension.

    Returns:
        coeffs: Array with shape [B, chunks, coeff_count, 4].
        frames: Original unpadded frame count.
    """
    if root_controls.ndim != 3 or root_controls.shape[-1] != ROOT_CONTROL_DIM:
        raise ValueError(f"Expected [B,T,{ROOT_CONTROL_DIM}], got {root_controls.shape}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if coeff_count <= 0 or coeff_count > chunk_size:
        raise ValueError("coeff_count must satisfy 0 < coeff_count <= chunk_size")

    batch, frames, dims = root_controls.shape
    chunks = int(math.ceil(frames / chunk_size))
    padded_frames = chunks * chunk_size
    pad = padded_frames - frames
    if pad:
        pad_values = np.repeat(root_controls[:, -1:, :], pad, axis=1)
        padded = np.concatenate([root_controls, pad_values], axis=1)
    else:
        padded = root_controls

    chunked = padded.reshape(batch, chunks, chunk_size, dims)
    coeffs = dct(chunked, type=2, axis=2, norm="ortho")
    return coeffs[:, :, :coeff_count, :].astype(np.float32, copy=False), frames


def idct_root_control_coefficients(
    coeffs: np.ndarray,
    frames: int,
    chunk_size: int,
) -> np.ndarray:
    """Reconstruct root controls from low-frequency DCT coefficients."""
    if coeffs.ndim != 4 or coeffs.shape[-1] != ROOT_CONTROL_DIM:
        raise ValueError(f"Expected [B,chunks,K,{ROOT_CONTROL_DIM}], got {coeffs.shape}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if frames <= 0:
        raise ValueError("frames must be positive")
    coeff_count = int(coeffs.shape[2])
    if coeff_count <= 0 or coeff_count > chunk_size:
        raise ValueError("coeff_count must satisfy 0 < coeff_count <= chunk_size")

    batch, chunks, _, dims = coeffs.shape
    full_coeffs = np.zeros((batch, chunks, chunk_size, dims), dtype=np.float32)
    full_coeffs[:, :, :coeff_count, :] = coeffs.astype(np.float32, copy=False)
    recon = idct(full_coeffs, type=2, axis=2, norm="ortho")
    padded_frames = chunks * chunk_size
    return recon.reshape(batch, padded_frames, dims)[:, :frames, :].astype(np.float32, copy=False)


def dct_reconstruct_root_controls(
    root_controls: np.ndarray,
    chunk_size: int,
    coeff_count: int,
) -> np.ndarray:
    """Chunk root controls, keep low-frequency DCT coefficients, and reconstruct.

    Args:
        root_controls: Array with shape [B, T, 4] in physical root-control
            units: yaw rad/s, local vx m/s, local vz m/s, root height m.
        chunk_size: Number of frames per DCT chunk.
        coeff_count: Number of low-frequency DCT coefficients to keep per
            chunk and root-control dimension.
    """
    coeffs, frames = dct_root_control_coefficients(root_controls, chunk_size, coeff_count)
    return idct_root_control_coefficients(coeffs, frames, chunk_size)


def _flush_eval(
    batch: list[dict[str, Any]],
    chunk_size: int,
    coeff_count: int,
    fps: float,
    device: torch.device,
    sums: dict[str, float],
) -> None:
    features_np = np.stack([item["features_263"] for item in batch]).astype(np.float32, copy=False)
    root_controls = np.stack([
        _root_controls_from_features(item["features_263"], fps)
        for item in batch
    ]).astype(np.float32, copy=False)
    pred_root_controls = dct_reconstruct_root_controls(root_controls, chunk_size, coeff_count)
    pred_features_np = _features_with_root_controls(features_np, pred_root_controls, fps)

    ref_features = torch.from_numpy(features_np).to(device)
    pred_features = torch.from_numpy(pred_features_np).to(device)
    with torch.no_grad():
        ref_joints = recover_from_ric(ref_features, 22)
        pred_joints = recover_from_ric(pred_features, 22)
        joint_err = torch.linalg.norm(pred_joints - ref_joints, dim=-1)
        ref_ra = ref_joints - ref_joints[..., :1, :]
        pred_ra = pred_joints - pred_joints[..., :1, :]
        ra_err = torch.linalg.norm(pred_ra - ref_ra, dim=-1)

        _, ref_root = motion_process.recover_root_rot_pos(ref_features)
        _, pred_root = motion_process.recover_root_rot_pos(pred_features)
        root_xz_err = torch.linalg.norm(pred_root[..., [0, 2]] - ref_root[..., [0, 2]], dim=-1)
        root_y_err = torch.abs(pred_root[..., 1] - ref_root[..., 1])
        ref_steps = torch.linalg.norm(ref_root[..., 1:, [0, 2]] - ref_root[..., :-1, [0, 2]], dim=-1)
        pred_steps = torch.linalg.norm(pred_root[..., 1:, [0, 2]] - pred_root[..., :-1, [0, 2]], dim=-1)

    root_control_abs = np.abs(pred_root_controls - root_controls)
    speed_bias = (pred_steps - ref_steps) * fps * 1000.0
    sums["mpjpe_sum"] += float(joint_err.sum().item())
    sums["mpjpe_count"] += int(joint_err.numel())
    sums["ra_sum"] += float(ra_err.sum().item())
    sums["ra_count"] += int(ra_err.numel())
    sums["root_xz_sum"] += float(root_xz_err.sum().item())
    sums["root_xz_count"] += int(root_xz_err.numel())
    sums["root_y_sum"] += float(root_y_err.sum().item())
    sums["root_y_count"] += int(root_y_err.numel())
    sums["final_xz_sum"] += float(root_xz_err[:, -1].sum().item())
    sums["path_error_sum"] += float((pred_steps.sum(dim=-1) - ref_steps.sum(dim=-1)).sum().item())
    sums["speed_bias_sum"] += float(speed_bias.sum().item())
    sums["speed_bias_count"] += int(speed_bias.numel())
    sums["control_l1_sum"] += float(root_control_abs.sum())
    sums["control_l1_count"] += int(root_control_abs.size)
    sums["yaw_l1_sum"] += float(root_control_abs[..., 0].sum())
    sums["vel_l1_sum"] += float(root_control_abs[..., 1:3].sum())
    sums["height_l1_sum"] += float(root_control_abs[..., 3].sum())
    sums["window_count"] += len(batch)
    sums["frame_count"] += int(features_np.shape[0] * features_np.shape[1])


def evaluate_combo(args: argparse.Namespace, chunk_size: int, coeff_count: int) -> dict[str, Any]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    store = FactorizedLocalStore(
        args.cache_root,
        split=args.split,
        max_sequences=args.max_sequences,
        preload_features=True,
    )
    sums = {
        "mpjpe_sum": 0.0,
        "mpjpe_count": 0,
        "ra_sum": 0.0,
        "ra_count": 0,
        "root_xz_sum": 0.0,
        "root_xz_count": 0,
        "root_y_sum": 0.0,
        "root_y_count": 0,
        "final_xz_sum": 0.0,
        "path_error_sum": 0.0,
        "speed_bias_sum": 0.0,
        "speed_bias_count": 0,
        "control_l1_sum": 0.0,
        "control_l1_count": 0,
        "yaw_l1_sum": 0.0,
        "vel_l1_sum": 0.0,
        "height_l1_sum": 0.0,
        "window_count": 0,
        "frame_count": 0,
    }
    pending: dict[int, list[dict[str, Any]]] = {}
    examples: list[str] = []
    for item in store.iter_windows(
        args.window_frames,
        args.stride,
        args.min_window_frames,
        args.include_tail,
    ):
        length = int(item["features_263"].shape[0])
        pending.setdefault(length, []).append(item)
        if len(examples) < 5:
            examples.append(f"{item['id']}:{item['start']}-{item['end']}")
        if len(pending[length]) >= args.batch_size:
            _flush_eval(pending[length], chunk_size, coeff_count, args.fps, device, sums)
            pending[length] = []
    for batch in list(pending.values()):
        if batch:
            _flush_eval(batch, chunk_size, coeff_count, args.fps, device, sums)

    chunks_per_window = int(math.ceil(args.window_frames / chunk_size)) if args.window_frames > 0 else None
    continuous_values = (
        chunks_per_window * coeff_count * ROOT_CONTROL_DIM
        if chunks_per_window is not None
        else None
    )
    raw_values = args.window_frames * ROOT_CONTROL_DIM if args.window_frames > 0 else None
    result = {
        "cache_root": str(Path(args.cache_root).expanduser().resolve()),
        "split": args.split,
        "window_frames": args.window_frames,
        "stride": args.stride,
        "include_tail": args.include_tail,
        "chunk_size": chunk_size,
        "coeff_count": coeff_count,
        "chunks_per_window": chunks_per_window,
        "continuous_values_per_window": continuous_values,
        "raw_root_values_per_window": raw_values,
        "values_vs_raw": (
            float(continuous_values) / float(raw_values)
            if continuous_values is not None and raw_values
            else None
        ),
        "compression_ratio_raw_to_dct": (
            float(raw_values) / float(continuous_values)
            if continuous_values
            else None
        ),
        "window_count": int(sums["window_count"]),
        "frame_count": int(sums["frame_count"]),
        "example_windows": examples,
        "mpjpe_mm": sums["mpjpe_sum"] / max(sums["mpjpe_count"], 1) * 1000.0,
        "root_aligned_mpjpe_mm": sums["ra_sum"] / max(sums["ra_count"], 1) * 1000.0,
        "root_gap_mm": (
            sums["mpjpe_sum"] / max(sums["mpjpe_count"], 1)
            - sums["ra_sum"] / max(sums["ra_count"], 1)
        ) * 1000.0,
        "root_xz_mean_error_mm": sums["root_xz_sum"] / max(sums["root_xz_count"], 1) * 1000.0,
        "root_y_mean_error_mm": sums["root_y_sum"] / max(sums["root_y_count"], 1) * 1000.0,
        "final_xz_error_mm": sums["final_xz_sum"] / max(sums["window_count"], 1) * 1000.0,
        "path_error_m": sums["path_error_sum"] / max(sums["window_count"], 1),
        "speed_bias_mm_per_s": sums["speed_bias_sum"] / max(sums["speed_bias_count"], 1),
        "root_control_l1": sums["control_l1_sum"] / max(sums["control_l1_count"], 1),
        "yaw_rate_l1_radps": sums["yaw_l1_sum"] / max(sums["frame_count"], 1),
        "local_velocity_l1_mps": sums["vel_l1_sum"] / max(sums["frame_count"] * 2, 1),
        "height_l1_m": sums["height_l1_sum"] / max(sums["frame_count"], 1),
    }
    return result


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for chunk_size in args.chunk_sizes:
        for coeff_count in args.coeff_counts:
            if coeff_count > chunk_size:
                continue
            result = evaluate_combo(args, chunk_size, coeff_count)
            results.append(result)
            out_path = out_dir / (
                f"{args.split}_w{args.window_frames}_chunk{chunk_size}_k{coeff_count}.json"
            )
            out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(result, indent=2, sort_keys=True))

    results = sorted(
        results,
        key=lambda item: (
            item["continuous_values_per_window"],
            item["mpjpe_mm"],
            item["root_gap_mm"],
        ),
    )
    summary = {
        "cache_root": str(Path(args.cache_root).expanduser().resolve()),
        "split": args.split,
        "window_frames": args.window_frames,
        "stride": args.stride,
        "chunk_sizes": args.chunk_sizes,
        "coeff_counts": args.coeff_counts,
        "results": results,
    }
    summary_path = out_dir / f"{args.split}_w{args.window_frames}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FAST-like chunked DCT root command codec.")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--window-frames", type=int, default=196)
    parser.add_argument("--stride", type=int, default=196)
    parser.add_argument("--include-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-window-frames", type=int, default=40)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--coeff-counts", nargs="+", type=int, default=[2, 4, 6, 8])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
