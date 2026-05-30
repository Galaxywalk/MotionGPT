from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from ..features import motion_process, recover_from_ric
from ..vqvae import DEFAULT_MEAN, DEFAULT_STD, load_vqvae, resolve_device


DEFAULT_M4HUMAN_CACHE = "/cpfs01/liangbo/data/MotionGPT/m4human_cache/v2_xz-y_param_joints_m4ref_20hz"
DEFAULT_EXP3_CKPT = (
    "experiments/mgpt/"
    "VQVAE_HumanML3D_M4Human20Hz_mix70_lr2e5_multilen_path_finetune/"
    "checkpoints/epoch=649.ckpt"
)
DEFAULT_OUT_ROOT = "/cpfs01/liangbo/data/MotionGPT/factorized_audit/root_quality_exp3"
FOOT_JOINTS = [7, 10, 8, 11]


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
    values = np.asarray(
        list(values) if not isinstance(values, np.ndarray) else values,
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    out = {f"p{q:02d}": float(np.percentile(values, q)) for q in qs}
    out["mean"] = float(values.mean())
    out["std"] = float(values.std())
    return out


def _concat(items: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(items) if items else np.array([], dtype=np.float64)


def _feature_to_joints(features: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    return recover_from_ric(tensor, 22).cpu().numpy()


def _recover_root(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    root_quat, root_pos = motion_process.recover_root_rot_pos(tensor)
    # MotionGPT stores yaw as a half-angle quaternion around Y:
    # q=(cos(theta), 0, sin(theta), 0). The accumulated theta is sufficient for
    # comparing yaw smoothness/velocity in the same convention as the features.
    yaw = torch.atan2(root_quat[..., 2], root_quat[..., 0])
    return yaw.cpu().numpy(), root_pos.cpu().numpy()


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.shape[0] < 3:
        return values.copy()
    window = min(window, values.shape[0])
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, [(pad, pad)] + [(0, 0)] * (values.ndim - 1), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    flat = padded.reshape(padded.shape[0], -1)
    out = np.stack([
        np.convolve(flat[:, i], kernel, mode="valid")
        for i in range(flat.shape[1])
    ], axis=-1)
    return out.reshape(values.shape).astype(values.dtype, copy=False)


def _smooth_signal(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    if window <= 1 or values.shape[0] < 3:
        return values.copy()
    window = min(window, values.shape[0])
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return _moving_average(values, window)
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(
            values,
            window_length=window,
            polyorder=min(polyorder, window - 1),
            axis=0,
            mode="interp",
        ).astype(values.dtype, copy=False)
    except Exception:
        return _moving_average(values, window)


def smooth_root_features(
    features: np.ndarray,
    window_frames: int,
    polyorder: int,
) -> np.ndarray:
    smoothed = features.copy()
    smoothed[:, 0:4] = _smooth_signal(
        features[:, 0:4].astype(np.float64, copy=False),
        window=window_frames,
        polyorder=polyorder,
    ).astype(np.float32, copy=False)
    return smoothed


def _iter_humanml_features(root: Path, split: str, max_sequences: int):
    yielded = 0
    for motion_id in _read_ids(root / f"{split}.txt"):
        path = root / "new_joint_vecs" / f"{motion_id}.npy"
        if not path.exists():
            continue
        yield motion_id, np.load(path).astype(np.float32, copy=False), {}
        yielded += 1
        if max_sequences and yielded >= max_sequences:
            break


def _iter_m4human_features(cache_root: Path, split: str, max_sequences: int):
    rows = [
        row for row in _load_jsonl(cache_root / "sequences.jsonl")
        if row.get("subset") == split
    ]
    for idx, row in enumerate(rows):
        yield (
            row["id"],
            np.load(cache_root / row["features"]).astype(np.float32, copy=False),
            row,
        )
        if max_sequences and idx + 1 >= max_sequences:
            break


def _window_slices(
    length: int,
    window_frames: int,
    stride: int,
    min_window_frames: int,
    include_tail: bool,
) -> list[tuple[int, int]]:
    if length < min_window_frames:
        return []
    if window_frames <= 0:
        full_len = (length // 4) * 4
        return [(0, full_len)] if full_len >= min_window_frames else []

    slices: list[tuple[int, int]] = []
    for start in range(0, max(length - window_frames + 1, 0), stride):
        slices.append((start, window_frames))
    covered_end = slices[-1][0] + slices[-1][1] if slices else 0
    if include_tail and length > covered_end:
        tail_len = ((length - covered_end) // 4) * 4
        if tail_len >= min_window_frames:
            slices.append((covered_end, tail_len))
    if not slices and include_tail:
        tail_len = (length // 4) * 4
        if tail_len >= min_window_frames:
            slices.append((0, tail_len))
    return slices


class RootQualityStats:
    def __init__(self, source_split: str, fps: float, foot_slip_threshold: float) -> None:
        self.source_split = source_split
        self.fps = float(fps)
        self.foot_slip_threshold = float(foot_slip_threshold)
        self.sequence_count = 0
        self.frame_count = 0
        self.window_count = 0
        self.example_sequences: list[str] = []

        self.local_speed_mps: list[np.ndarray] = []
        self.global_speed_mps: list[np.ndarray] = []
        self.local_velocity_x_mps: list[np.ndarray] = []
        self.local_velocity_z_mps: list[np.ndarray] = []
        self.root_accel_mps2: list[np.ndarray] = []
        self.yaw_vel_degps: list[np.ndarray] = []
        self.yaw_accel_degps2: list[np.ndarray] = []
        self.root_height_m: list[np.ndarray] = []
        self.root_height_delta_mps: list[np.ndarray] = []

        self.seq_root_height_std_m: list[float] = []
        self.seq_root_height_range_m: list[float] = []
        self.seq_path_m: list[float] = []
        self.seq_endpoint_m: list[float] = []
        self.seq_bbox_diag_m: list[float] = []
        self.window_path_m: list[float] = []
        self.window_endpoint_m: list[float] = []
        self.window_bbox_diag_m: list[float] = []

        self.contact_values: list[np.ndarray] = []
        self.contact_foot_speed_mps: list[np.ndarray] = []
        self.sliding_flags: list[np.ndarray] = []
        self.low_height_contact_foot_speed_mps: list[np.ndarray] = []

        self.smooth_mpjpe_mm: list[float] = []
        self.smooth_root_xz_mean_change_mm: list[float] = []
        self.smooth_final_xz_change_mm: list[float] = []
        self.smooth_path_change_m: list[float] = []

    def add_sequence(
        self,
        seq_id: str,
        features: np.ndarray,
        window_frames: int,
        stride: int,
        min_window_frames: int,
        include_tail: bool,
        smooth_window_frames: int,
        smooth_polyorder: int,
    ) -> None:
        if features.ndim != 2 or features.shape[-1] != 263 or features.shape[0] < 2:
            return

        self.sequence_count += 1
        self.frame_count += int(features.shape[0])
        if len(self.example_sequences) < 5:
            self.example_sequences.append(seq_id)

        yaw, root = _recover_root(features)
        root_xz = root[:, [0, 2]]
        global_steps = np.diff(root_xz, axis=0)
        global_speed = np.linalg.norm(global_steps, axis=-1) * self.fps
        local_vel = features[:, 1:3].astype(np.float64, copy=False) * self.fps
        local_speed = np.linalg.norm(local_vel, axis=-1)
        yaw_vel = features[:, 0].astype(np.float64, copy=False) * self.fps * 180.0 / math.pi
        root_height = features[:, 3].astype(np.float64, copy=False)

        self.local_speed_mps.append(local_speed)
        self.local_velocity_x_mps.append(local_vel[:, 0])
        self.local_velocity_z_mps.append(local_vel[:, 1])
        self.global_speed_mps.append(global_speed)
        self.yaw_vel_degps.append(yaw_vel)
        self.root_height_m.append(root_height)
        if local_vel.shape[0] > 1:
            self.root_accel_mps2.append(np.linalg.norm(np.diff(local_vel, axis=0), axis=-1) * self.fps)
            self.yaw_accel_degps2.append(np.diff(yaw_vel) * self.fps)
            self.root_height_delta_mps.append(np.diff(root_height) * self.fps)

        steps = np.linalg.norm(global_steps, axis=-1)
        bbox = root_xz.max(axis=0) - root_xz.min(axis=0)
        self.seq_path_m.append(float(steps.sum()))
        self.seq_endpoint_m.append(float(np.linalg.norm(root_xz[-1] - root_xz[0])))
        self.seq_bbox_diag_m.append(float(np.linalg.norm(bbox)))
        self.seq_root_height_std_m.append(float(root_height.std()))
        self.seq_root_height_range_m.append(float(root_height.max() - root_height.min()))

        for start, frames in _window_slices(
            len(features),
            window_frames,
            stride,
            min_window_frames,
            include_tail,
        ):
            window_root = root_xz[start:start + frames]
            if len(window_root) < 2:
                continue
            window_steps = np.linalg.norm(np.diff(window_root, axis=0), axis=-1)
            window_bbox = window_root.max(axis=0) - window_root.min(axis=0)
            self.window_path_m.append(float(window_steps.sum()))
            self.window_endpoint_m.append(float(np.linalg.norm(window_root[-1] - window_root[0])))
            self.window_bbox_diag_m.append(float(np.linalg.norm(window_bbox)))
            self.window_count += 1

        self._add_contacts(features)
        self._add_smoothing_stats(
            features,
            root_xz,
            smooth_window_frames,
            smooth_polyorder,
        )

    def _add_contacts(self, features: np.ndarray) -> None:
        if features.shape[0] < 2:
            return
        joints = _feature_to_joints(features)
        foot_speed = np.linalg.norm(
            joints[1:, FOOT_JOINTS, :] - joints[:-1, FOOT_JOINTS, :],
            axis=-1,
        ) * self.fps
        contacts = features[:-1, -4:].astype(np.float64, copy=False)
        contact_mask = contacts > 0.5
        self.contact_values.append(contacts.reshape(-1))
        if np.any(contact_mask):
            contact_speed = foot_speed[contact_mask]
            self.contact_foot_speed_mps.append(contact_speed)
            self.sliding_flags.append(contact_speed > self.foot_slip_threshold)

        foot_height = joints[:-1, FOOT_JOINTS, 1]
        low_height_mask = foot_height < 0.08
        if np.any(low_height_mask):
            self.low_height_contact_foot_speed_mps.append(foot_speed[low_height_mask])

    def _add_smoothing_stats(
        self,
        features: np.ndarray,
        root_xz: np.ndarray,
        smooth_window_frames: int,
        smooth_polyorder: int,
    ) -> None:
        smoothed = smooth_root_features(
            features,
            window_frames=smooth_window_frames,
            polyorder=smooth_polyorder,
        )
        _, smooth_root = _recover_root(smoothed)
        smooth_xz = smooth_root[:, [0, 2]]
        root_delta = smooth_xz - root_xz
        self.smooth_root_xz_mean_change_mm.append(
            float(np.linalg.norm(root_delta, axis=-1).mean() * 1000.0))
        self.smooth_final_xz_change_mm.append(
            float(np.linalg.norm(root_delta[-1]) * 1000.0))

        path = np.linalg.norm(np.diff(root_xz, axis=0), axis=-1).sum()
        smooth_path = np.linalg.norm(np.diff(smooth_xz, axis=0), axis=-1).sum()
        self.smooth_path_change_m.append(float(smooth_path - path))

        ref_joints = _feature_to_joints(features)
        smooth_joints = _feature_to_joints(smoothed)
        joint_err = np.linalg.norm(smooth_joints - ref_joints, axis=-1)
        self.smooth_mpjpe_mm.append(float(joint_err.mean() * 1000.0))

    def summary(self) -> dict[str, Any]:
        contact_values = _concat(self.contact_values)
        contact_speed = _concat(self.contact_foot_speed_mps)
        sliding = _concat(self.sliding_flags).astype(np.float64, copy=False)
        low_height_speed = _concat(self.low_height_contact_foot_speed_mps)
        return {
            "source_split": self.source_split,
            "sequence_count": self.sequence_count,
            "frame_count": self.frame_count,
            "window_count": self.window_count,
            "example_sequences": self.example_sequences,
            "root_local_speed_mps": _quantiles(_concat(self.local_speed_mps)),
            "root_global_speed_mps": _quantiles(_concat(self.global_speed_mps)),
            "root_local_vx_mps": _quantiles(_concat(self.local_velocity_x_mps)),
            "root_local_vz_mps": _quantiles(_concat(self.local_velocity_z_mps)),
            "root_accel_mps2": _quantiles(_concat(self.root_accel_mps2)),
            "yaw_velocity_degps": _quantiles(_concat(self.yaw_vel_degps)),
            "yaw_accel_degps2": _quantiles(_concat(self.yaw_accel_degps2)),
            "root_height_m": _quantiles(_concat(self.root_height_m)),
            "root_height_delta_mps": _quantiles(_concat(self.root_height_delta_mps)),
            "seq_root_height_std_m": _quantiles(np.asarray(self.seq_root_height_std_m)),
            "seq_root_height_range_m": _quantiles(np.asarray(self.seq_root_height_range_m)),
            "seq_path_m": _quantiles(np.asarray(self.seq_path_m)),
            "seq_endpoint_m": _quantiles(np.asarray(self.seq_endpoint_m)),
            "seq_bbox_diag_m": _quantiles(np.asarray(self.seq_bbox_diag_m)),
            "window_path_m": _quantiles(np.asarray(self.window_path_m)),
            "window_endpoint_m": _quantiles(np.asarray(self.window_endpoint_m)),
            "window_bbox_diag_m": _quantiles(np.asarray(self.window_bbox_diag_m)),
            "contact_value": _quantiles(contact_values),
            "contact_ratio": float(contact_values.mean()) if contact_values.size else 0.0,
            "contact_foot_speed_mps": _quantiles(contact_speed),
            "contact_sliding_ratio": float(sliding.mean()) if sliding.size else 0.0,
            "low_height_foot_speed_mps": _quantiles(low_height_speed),
            "smooth_root_mpjpe_mm": _quantiles(np.asarray(self.smooth_mpjpe_mm)),
            "smooth_root_xz_mean_change_mm": _quantiles(
                np.asarray(self.smooth_root_xz_mean_change_mm)),
            "smooth_final_xz_change_mm": _quantiles(
                np.asarray(self.smooth_final_xz_change_mm)),
            "smooth_path_change_m": _quantiles(np.asarray(self.smooth_path_change_m)),
        }


class ReconstructionCaseStats:
    def __init__(self, name: str) -> None:
        self.name = name
        self.recon_sum = 0.0
        self.recon_count = 0
        self.root_aligned_sum = 0.0
        self.root_aligned_count = 0
        self.root_xz_error_sum = 0.0
        self.root_xz_error_count = 0
        self.final_xz_errors: list[float] = []
        self.path_errors: list[float] = []

    def update(self, ref_features: torch.Tensor, case_features: torch.Tensor) -> None:
        ref_joints = recover_from_ric(ref_features, 22)
        case_joints = recover_from_ric(case_features, 22)
        joint_err = torch.linalg.norm(case_joints - ref_joints, dim=-1)
        self.recon_sum += float(joint_err.sum().item())
        self.recon_count += int(joint_err.numel())

        ref_ra = ref_joints - ref_joints[..., :1, :]
        case_ra = case_joints - case_joints[..., :1, :]
        ra_err = torch.linalg.norm(case_ra - ref_ra, dim=-1)
        self.root_aligned_sum += float(ra_err.sum().item())
        self.root_aligned_count += int(ra_err.numel())

        _, ref_root = motion_process.recover_root_rot_pos(ref_features)
        _, case_root = motion_process.recover_root_rot_pos(case_features)
        xz_err = torch.linalg.norm(
            case_root[..., [0, 2]] - ref_root[..., [0, 2]],
            dim=-1,
        )
        self.root_xz_error_sum += float(xz_err.sum().item())
        self.root_xz_error_count += int(xz_err.numel())
        self.final_xz_errors.extend(xz_err[..., -1].detach().cpu().tolist())

        ref_path = torch.linalg.norm(
            ref_root[..., 1:, [0, 2]] - ref_root[..., :-1, [0, 2]],
            dim=-1,
        ).sum(dim=-1)
        case_path = torch.linalg.norm(
            case_root[..., 1:, [0, 2]] - case_root[..., :-1, [0, 2]],
            dim=-1,
        ).sum(dim=-1)
        self.path_errors.extend((case_path - ref_path).detach().cpu().tolist())

    def summary(self) -> dict[str, float]:
        mpjpe = self.recon_sum / max(self.recon_count, 1) * 1000.0
        root_aligned = (
            self.root_aligned_sum / max(self.root_aligned_count, 1) * 1000.0)
        return {
            "mpjpe_mm": mpjpe,
            "root_aligned_mpjpe_mm": root_aligned,
            "root_gap_mm": mpjpe - root_aligned,
            "root_xz_mean_error_mm": (
                self.root_xz_error_sum / max(self.root_xz_error_count, 1) * 1000.0),
            "final_xz_error_mm": float(np.mean(self.final_xz_errors) * 1000.0)
            if self.final_xz_errors else 0.0,
            "path_error_m": float(np.mean(self.path_errors)) if self.path_errors else 0.0,
        }


def _case_features(
    ref: torch.Tensor,
    pred: torch.Tensor,
    smooth_ref: torch.Tensor,
) -> dict[str, torch.Tensor]:
    cases = {
        "normal_pred_root_pred_local": pred,
        "gt_root_pred_local": pred.clone(),
        "pred_root_gt_local": ref.clone(),
        "gt_root_gt_local": ref,
        "smooth_gt_root_pred_local": pred.clone(),
        "smooth_gt_root_gt_local": ref.clone(),
    }
    cases["gt_root_pred_local"][..., 0:4] = ref[..., 0:4]
    cases["pred_root_gt_local"][..., 0:4] = pred[..., 0:4]
    cases["smooth_gt_root_pred_local"][..., 0:4] = smooth_ref[..., 0:4]
    cases["smooth_gt_root_gt_local"][..., 0:4] = smooth_ref[..., 0:4]
    return cases


def _flush_reconstruction_batch(
    batch: list[dict[str, Any]],
    model,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    stats: dict[str, ReconstructionCaseStats],
    smooth_window_frames: int,
    smooth_polyorder: int,
) -> None:
    ref_np = np.stack([item["features"] for item in batch]).astype(np.float32, copy=False)
    smooth_np = np.stack([
        smooth_root_features(item["features"], smooth_window_frames, smooth_polyorder)
        for item in batch
    ]).astype(np.float32, copy=False)
    ref = torch.as_tensor(ref_np, device=device)
    smooth_ref = torch.as_tensor(smooth_np, device=device)
    norm_ref = (ref - mean) / std
    with torch.no_grad():
        pred_norm, _, _ = model(norm_ref)
        pred = pred_norm * std + mean
        for name, case in _case_features(ref, pred, smooth_ref).items():
            stats[name].update(ref, case)


def _iter_source(source: str, split: str, args: argparse.Namespace):
    if source == "humanml3d":
        yield from _iter_humanml_features(
            Path(args.humanml_root).expanduser().resolve(),
            split,
            args.max_humanml_sequences,
        )
    elif source == "m4human":
        yield from _iter_m4human_features(
            Path(args.m4human_cache).expanduser().resolve(),
            split,
            args.max_m4human_sequences,
        )
    else:
        raise ValueError(f"Unsupported source={source}")


def _write_quantile_csv(results: dict[str, Any], out_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for source_split, summary in results.items():
        for metric, value in summary.items():
            if isinstance(value, dict) and "p50" in value:
                rows.append({"source_split": source_split, "metric": metric, **value})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_split",
                "metric",
                "mean",
                "std",
                "p00",
                "p01",
                "p05",
                "p10",
                "p25",
                "p50",
                "p75",
                "p90",
                "p95",
                "p99",
                "p100",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    smooth_window_frames = int(round(args.smooth_seconds * args.fps))
    if smooth_window_frames % 2 == 0:
        smooth_window_frames += 1
    smooth_window_frames = max(3, smooth_window_frames)

    quality_results: dict[str, Any] = {}
    reconstruction_results: dict[str, Any] = {}

    model = None
    mean = std = None
    device = torch.device("cpu")
    if args.checkpoint:
        device = resolve_device(args.device)
        mean = torch.from_numpy(np.load(args.mean).astype(np.float32)).to(device)
        std = torch.from_numpy(np.load(args.std).astype(np.float32)).to(device)
        model, ckpt_meta = load_vqvae(
            Path(args.checkpoint),
            device,
            args.calibration_domain,
        )
    else:
        ckpt_meta = {}

    for source in args.sources:
        for split in args.splits:
            key = f"{source}_{split}"
            stats = RootQualityStats(
                key,
                fps=args.fps,
                foot_slip_threshold=args.foot_slip_threshold,
            )
            recon_stats = {
                name: ReconstructionCaseStats(name)
                for name in (
                    "normal_pred_root_pred_local",
                    "gt_root_pred_local",
                    "pred_root_gt_local",
                    "gt_root_gt_local",
                    "smooth_gt_root_pred_local",
                    "smooth_gt_root_gt_local",
                )
            }
            pending: dict[int, list[dict[str, Any]]] = {}
            seq_count = 0
            window_count = 0

            for seq_id, features, meta in _iter_source(source, split, args):
                seq_count += 1
                stats.add_sequence(
                    seq_id,
                    features,
                    window_frames=args.window_frames,
                    stride=args.stride,
                    min_window_frames=args.min_window_frames,
                    include_tail=args.include_tail,
                    smooth_window_frames=smooth_window_frames,
                    smooth_polyorder=args.smooth_polyorder,
                )

                if model is not None:
                    for start, frames in _window_slices(
                        len(features),
                        args.window_frames,
                        args.stride,
                        args.min_window_frames,
                        args.include_tail,
                    ):
                        item = {
                            "id": seq_id,
                            "start": start,
                            "end": start + frames,
                            "features": features[start:start + frames],
                        }
                        pending.setdefault(frames, []).append(item)
                        window_count += 1
                        if len(pending[frames]) >= args.batch_size:
                            _flush_reconstruction_batch(
                                pending[frames],
                                model,
                                mean,
                                std,
                                device,
                                recon_stats,
                                smooth_window_frames,
                                args.smooth_polyorder,
                            )
                            pending[frames] = []
                        if args.max_windows and window_count >= args.max_windows:
                            break

                if args.log_every and seq_count % args.log_every == 0:
                    print(f"{key}: processed {seq_count} sequences")
                if args.max_windows and window_count >= args.max_windows:
                    break

            if model is not None:
                for frames, batch in list(pending.items()):
                    if not batch:
                        continue
                    _flush_reconstruction_batch(
                        batch,
                        model,
                        mean,
                        std,
                        device,
                        recon_stats,
                        smooth_window_frames,
                        args.smooth_polyorder,
                    )
                    pending[frames] = []

            quality_results[key] = stats.summary()
            if model is not None:
                reconstruction_results[key] = {
                    name: case_stats.summary()
                    for name, case_stats in recon_stats.items()
                }

    payload = {
        **ckpt_meta,
        "checkpoint": str(Path(args.checkpoint)) if args.checkpoint else "",
        "humanml_root": str(Path(args.humanml_root).expanduser().resolve()),
        "m4human_cache": str(Path(args.m4human_cache).expanduser().resolve()),
        "sources": args.sources,
        "splits": args.splits,
        "fps": args.fps,
        "window_frames": args.window_frames,
        "stride": args.stride,
        "include_tail": args.include_tail,
        "min_window_frames": args.min_window_frames,
        "smooth_seconds": args.smooth_seconds,
        "smooth_window_frames": smooth_window_frames,
        "smooth_polyorder": args.smooth_polyorder,
        "foot_slip_threshold_mps": args.foot_slip_threshold,
        "quality": quality_results,
        "reconstruction_upper_bounds": reconstruction_results,
    }
    (out_root / "root_quality_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_quantile_csv(quality_results, out_root / "root_quality_quantiles.csv")
    if reconstruction_results:
        with (out_root / "root_reconstruction_upper_bounds.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            fieldnames = [
                "source_split",
                "case",
                "mpjpe_mm",
                "root_aligned_mpjpe_mm",
                "root_gap_mm",
                "root_xz_mean_error_mm",
                "final_xz_error_mm",
                "path_error_m",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for key, cases in reconstruction_results.items():
                for case, values in cases.items():
                    writer.writerow({"source_split": key, "case": case, **values})
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit root trajectory quality for the root/local tokenizer.")
    parser.add_argument("--humanml-root", default="datasets/humanml3d")
    parser.add_argument("--m4human-cache", default=DEFAULT_M4HUMAN_CACHE)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--sources", nargs="+", default=["humanml3d", "m4human"], choices=("humanml3d", "m4human"))
    parser.add_argument("--splits", nargs="+", default=["test"], choices=("train", "val", "test"))
    parser.add_argument("--max-humanml-sequences", type=int, default=0)
    parser.add_argument("--max-m4human-sequences", type=int, default=0)
    parser.add_argument("--checkpoint", default=DEFAULT_EXP3_CKPT)
    parser.add_argument("--mean", default=DEFAULT_MEAN)
    parser.add_argument("--std", default=DEFAULT_STD)
    parser.add_argument("--calibration-domain", default="none")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--window-frames", type=int, default=196)
    parser.add_argument("--stride", type=int, default=196)
    parser.add_argument("--include-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-window-frames", type=int, default=40)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--smooth-seconds", type=float, default=0.35)
    parser.add_argument("--smooth-polyorder", type=int, default=2)
    parser.add_argument("--foot-slip-threshold", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=500)
    return parser


def main() -> None:
    audit(build_parser().parse_args())


if __name__ == "__main__":
    main()
