from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..features import motion_process, recover_from_ric
from .local_vq import (
    DEFAULT_CACHE_ROOT,
    FactorizedLocalStore,
    LOCAL_JOINT_DIM,
    _load_checkpoint as load_local_checkpoint,
    local_vector_to_features,
)
from .root_branch import _features_with_root_controls, _root_controls_from_features
from .root_fast_codec import (
    ROOT_CONTROL_DIM,
    dct_root_control_coefficients,
    idct_root_control_coefficients,
)
from .root_fast_quantize import ResidualVectorKMeansQuantizer, VectorKMeansQuantizer


DEFAULT_OUT_DIR = "/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_full_eval"
DEFAULT_LOCAL_CKPT = (
    "/cpfs01/liangbo/data/MotionGPT/factorized_experiments/"
    "local_vq_m4human_scratch_full_v1/checkpoints/best.pt"
)


def load_rvq_quantizer(path: str | Path) -> tuple[ResidualVectorKMeansQuantizer, dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    payload = np.load(path, allow_pickle=False)
    if "depth" not in payload:
        raise ValueError(f"{path} does not look like an RVQ quantizer")
    depth = int(np.asarray(payload["depth"]).reshape(()))
    metadata = json.loads(str(np.asarray(payload["metadata"]).reshape(())))
    quantizers: list[VectorKMeansQuantizer] = []
    for idx in range(depth):
        quantizers.append(
            VectorKMeansQuantizer(
                centroids_norm=payload[f"centroids_norm_{idx}"].astype(np.float32, copy=False),
                mean=payload[f"mean_{idx}"].astype(np.float32, copy=False),
                std=payload[f"std_{idx}"].astype(np.float32, copy=False),
                normalize=bool(np.asarray(payload[f"normalize_{idx}"]).reshape(())),
            )
        )
    return ResidualVectorKMeansQuantizer(quantizers=quantizers), metadata


def _empty_sums() -> dict[str, float]:
    return {
        "mpjpe_sum": 0.0,
        "mpjpe_count": 0,
        "ra_sum": 0.0,
        "ra_count": 0,
        "local_only_mpjpe_sum": 0.0,
        "local_only_ra_sum": 0.0,
        "root_only_mpjpe_sum": 0.0,
        "root_only_ra_sum": 0.0,
        "root_xz_sum": 0.0,
        "root_xz_count": 0,
        "root_y_sum": 0.0,
        "root_y_count": 0,
        "final_xz_sum": 0.0,
        "path_error_sum": 0.0,
        "speed_bias_sum": 0.0,
        "speed_bias_count": 0,
        "local_body_sum": 0.0,
        "local_body_count": 0,
        "root_control_l1_sum": 0.0,
        "root_control_l1_count": 0,
        "window_count": 0,
        "frame_count": 0,
    }


def _joint_metrics(
    ref_features: torch.Tensor,
    pred_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ref_joints = recover_from_ric(ref_features, 22)
    pred_joints = recover_from_ric(pred_features, 22)
    joint_err = torch.linalg.norm(pred_joints - ref_joints, dim=-1)
    ref_ra = ref_joints - ref_joints[..., :1, :]
    pred_ra = pred_joints - pred_joints[..., :1, :]
    ra_err = torch.linalg.norm(pred_ra - ref_ra, dim=-1)
    return joint_err, ra_err


def _flush_eval(
    batch: list[dict[str, Any]],
    local_model: torch.nn.Module,
    local_mean: torch.Tensor,
    local_std: torch.Tensor,
    rvq: ResidualVectorKMeansQuantizer,
    rvq_metadata: dict[str, Any],
    fps: float,
    device: torch.device,
    sums: dict[str, float],
    local_code_hist: torch.Tensor,
    rvq_code_hist: np.ndarray,
) -> None:
    local_np = np.stack([item["local"] for item in batch]).astype(np.float32, copy=False)
    features_np = np.stack([item["features_263"] for item in batch]).astype(np.float32, copy=False)

    local = torch.from_numpy(local_np).to(device)
    local_norm = (local - local_mean) / local_std
    with torch.no_grad():
        pred_local_norm, _, _ = local_model(local_norm)
        pred_local = pred_local_norm * local_std + local_mean
        local_codes, _ = local_model.encode(local_norm)
        local_code_hist += torch.bincount(
            local_codes.reshape(-1).cpu(),
            minlength=local_code_hist.numel(),
        )
    pred_local_np = pred_local.detach().cpu().numpy().astype(np.float32, copy=False)
    pred_local_features_np = local_vector_to_features(features_np, pred_local_np)

    chunk_size = int(rvq_metadata["chunk_size"])
    coeff_count = int(rvq_metadata["coeff_count"])
    root_controls = np.stack([
        _root_controls_from_features(item["features_263"], fps)
        for item in batch
    ]).astype(np.float32, copy=False)
    coeffs, frames = dct_root_control_coefficients(root_controls, chunk_size, coeff_count)
    flat = coeffs.reshape(-1, coeff_count * ROOT_CONTROL_DIM)
    quantized_flat, rvq_codes = rvq.quantize(flat)
    for stage in range(rvq_codes.shape[1]):
        np.add.at(rvq_code_hist[stage], rvq_codes[:, stage], 1)
    pred_root_controls = idct_root_control_coefficients(
        quantized_flat.reshape(coeffs.shape),
        frames,
        chunk_size,
    )
    pred_full_features_np = _features_with_root_controls(
        pred_local_features_np,
        pred_root_controls,
        fps,
    )
    pred_root_only_features_np = _features_with_root_controls(
        features_np,
        pred_root_controls,
        fps,
    )

    ref_features = torch.from_numpy(features_np).to(device)
    pred_local_features = torch.from_numpy(pred_local_features_np).to(device)
    pred_root_only_features = torch.from_numpy(pred_root_only_features_np).to(device)
    pred_full_features = torch.from_numpy(pred_full_features_np).to(device)

    with torch.no_grad():
        joint_err, ra_err = _joint_metrics(ref_features, pred_full_features)
        local_only_err, local_only_ra = _joint_metrics(ref_features, pred_local_features)
        root_only_err, root_only_ra = _joint_metrics(ref_features, pred_root_only_features)

        _, ref_root = motion_process.recover_root_rot_pos(ref_features)
        _, pred_root = motion_process.recover_root_rot_pos(pred_full_features)
        root_xz_err = torch.linalg.norm(pred_root[..., [0, 2]] - ref_root[..., [0, 2]], dim=-1)
        root_y_err = torch.abs(pred_root[..., 1] - ref_root[..., 1])
        ref_steps = torch.linalg.norm(ref_root[..., 1:, [0, 2]] - ref_root[..., :-1, [0, 2]], dim=-1)
        pred_steps = torch.linalg.norm(pred_root[..., 1:, [0, 2]] - pred_root[..., :-1, [0, 2]], dim=-1)

    body_err = torch.linalg.norm(
        pred_local[..., :LOCAL_JOINT_DIM].reshape(pred_local.shape[0], pred_local.shape[1], 21, 3)
        - local[..., :LOCAL_JOINT_DIM].reshape(local.shape[0], local.shape[1], 21, 3),
        dim=-1,
    )
    root_control_abs = np.abs(pred_root_controls - root_controls)
    speed_bias = (pred_steps - ref_steps) * fps * 1000.0

    sums["mpjpe_sum"] += float(joint_err.sum().item())
    sums["mpjpe_count"] += int(joint_err.numel())
    sums["ra_sum"] += float(ra_err.sum().item())
    sums["ra_count"] += int(ra_err.numel())
    sums["local_only_mpjpe_sum"] += float(local_only_err.sum().item())
    sums["local_only_ra_sum"] += float(local_only_ra.sum().item())
    sums["root_only_mpjpe_sum"] += float(root_only_err.sum().item())
    sums["root_only_ra_sum"] += float(root_only_ra.sum().item())
    sums["root_xz_sum"] += float(root_xz_err.sum().item())
    sums["root_xz_count"] += int(root_xz_err.numel())
    sums["root_y_sum"] += float(root_y_err.sum().item())
    sums["root_y_count"] += int(root_y_err.numel())
    sums["final_xz_sum"] += float(root_xz_err[:, -1].sum().item())
    sums["path_error_sum"] += float((pred_steps.sum(dim=-1) - ref_steps.sum(dim=-1)).sum().item())
    sums["speed_bias_sum"] += float(speed_bias.sum().item())
    sums["speed_bias_count"] += int(speed_bias.numel())
    sums["local_body_sum"] += float(body_err.sum().item())
    sums["local_body_count"] += int(body_err.numel())
    sums["root_control_l1_sum"] += float(root_control_abs.sum())
    sums["root_control_l1_count"] += int(root_control_abs.size)
    sums["window_count"] += len(batch)
    sums["frame_count"] += int(features_np.shape[0] * features_np.shape[1])


def evaluate_one(args: argparse.Namespace, split: str) -> dict[str, Any]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    local_model, local_mean, local_std, local_payload = load_local_checkpoint(Path(args.local_checkpoint), device)
    rvq, rvq_metadata = load_rvq_quantizer(args.root_rvq_quantizer)
    local_code_num = int(local_payload["args"]["code_num"])
    rvq_depth = int(rvq.depth)
    rvq_vocab = int(rvq.codebook_size)
    local_code_hist = torch.zeros(local_code_num, dtype=torch.long)
    rvq_code_hist = np.zeros((rvq_depth, rvq_vocab), dtype=np.int64)

    store = FactorizedLocalStore(
        args.cache_root,
        split=split,
        max_sequences=args.max_sequences,
        preload_features=True,
    )
    sums = _empty_sums()
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
            _flush_eval(
                pending[length],
                local_model,
                local_mean,
                local_std,
                rvq,
                rvq_metadata,
                args.fps,
                device,
                sums,
                local_code_hist,
                rvq_code_hist,
            )
            pending[length] = []
    for batch in list(pending.values()):
        if batch:
            _flush_eval(
                batch,
                local_model,
                local_mean,
                local_std,
                rvq,
                rvq_metadata,
                args.fps,
                device,
                sums,
                local_code_hist,
                rvq_code_hist,
            )

    local_probs = local_code_hist.float() / max(int(local_code_hist.sum().item()), 1)
    local_entropy = float((-(local_probs[local_probs > 0] * torch.log(local_probs[local_probs > 0])).sum()).item())
    rvq_unique = []
    rvq_effective = []
    for stage in range(rvq_code_hist.shape[0]):
        hist = rvq_code_hist[stage]
        probs = hist.astype(np.float64) / max(float(hist.sum()), 1.0)
        entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
        rvq_unique.append(int(np.count_nonzero(hist)))
        rvq_effective.append(float(math.exp(entropy)))

    chunks_per_window = int(math.ceil(args.window_frames / int(rvq_metadata["chunk_size"])))
    local_tokens = float(local_code_hist.sum().item()) / max(sums["window_count"], 1)
    root_tokens = chunks_per_window * rvq_depth
    result = {
        "cache_root": str(Path(args.cache_root).expanduser().resolve()),
        "split": split,
        "window_frames": args.window_frames,
        "stride": args.stride,
        "include_tail": args.include_tail,
        "window_count": int(sums["window_count"]),
        "frame_count": int(sums["frame_count"]),
        "example_windows": examples,
        "local_checkpoint": str(Path(args.local_checkpoint).expanduser().resolve()),
        "local_checkpoint_label": args.local_label,
        "local_code_num": local_code_num,
        "local_unique_codes": int((local_code_hist > 0).sum().item()),
        "local_effective_codes": float(math.exp(local_entropy)),
        "local_tokens_per_window_mean": local_tokens,
        "root_rvq_quantizer": str(Path(args.root_rvq_quantizer).expanduser().resolve()),
        "root_chunk_size": int(rvq_metadata["chunk_size"]),
        "root_coeff_count": int(rvq_metadata["coeff_count"]),
        "root_rvq_vocab": rvq_vocab,
        "root_rvq_depth": rvq_depth,
        "root_chunks_per_window": chunks_per_window,
        "root_tokens_per_window": root_tokens,
        "root_bits_per_window": float(root_tokens) * math.log2(max(rvq_vocab, 2)),
        "rvq_unique_codes_per_stage": rvq_unique,
        "rvq_effective_vocab_per_stage": rvq_effective,
        "total_tokens_per_window_mean": local_tokens + root_tokens,
        "mpjpe_mm": sums["mpjpe_sum"] / max(sums["mpjpe_count"], 1) * 1000.0,
        "root_aligned_mpjpe_mm": sums["ra_sum"] / max(sums["ra_count"], 1) * 1000.0,
        "root_gap_mm": (
            sums["mpjpe_sum"] / max(sums["mpjpe_count"], 1)
            - sums["ra_sum"] / max(sums["ra_count"], 1)
        ) * 1000.0,
        "local_only_mpjpe_mm": sums["local_only_mpjpe_sum"] / max(sums["mpjpe_count"], 1) * 1000.0,
        "local_only_root_aligned_mpjpe_mm": sums["local_only_ra_sum"] / max(sums["ra_count"], 1) * 1000.0,
        "root_only_mpjpe_mm": sums["root_only_mpjpe_sum"] / max(sums["mpjpe_count"], 1) * 1000.0,
        "root_only_root_aligned_mpjpe_mm": sums["root_only_ra_sum"] / max(sums["ra_count"], 1) * 1000.0,
        "local_body_mpjpe_mm": sums["local_body_sum"] / max(sums["local_body_count"], 1) * 1000.0,
        "root_xz_mean_error_mm": sums["root_xz_sum"] / max(sums["root_xz_count"], 1) * 1000.0,
        "root_y_mean_error_mm": sums["root_y_sum"] / max(sums["root_y_count"], 1) * 1000.0,
        "final_xz_error_mm": sums["final_xz_sum"] / max(sums["window_count"], 1) * 1000.0,
        "path_error_m": sums["path_error_sum"] / max(sums["window_count"], 1),
        "speed_bias_mm_per_s": sums["speed_bias_sum"] / max(sums["speed_bias_count"], 1),
        "root_control_l1": sums["root_control_l1_sum"] / max(sums["root_control_l1_count"], 1),
    }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for split in args.splits:
        result = evaluate_one(args, split)
        results.append(result)
        name = (
            f"{args.local_label}_{split}_w{args.window_frames}_"
            f"chunk{result['root_chunk_size']}_k{result['root_coeff_count']}_"
            f"vocab{result['root_rvq_vocab']}_d{result['root_rvq_depth']}.json"
        )
        (out_dir / name).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
    summary = {
        "out_dir": str(out_dir),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate local VQ + Root-FAST RVQ full tokenizer reconstruction.")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--local-checkpoint", default=DEFAULT_LOCAL_CKPT)
    parser.add_argument("--local-label", default="local_scratch_full")
    parser.add_argument("--root-rvq-quantizer", required=True)
    parser.add_argument("--splits", nargs="+", default=["test"], choices=("train", "val", "test"))
    parser.add_argument("--window-frames", type=int, default=196)
    parser.add_argument("--stride", type=int, default=196)
    parser.add_argument("--include-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-window-frames", type=int, default=40)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fps", type=float, default=20.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
