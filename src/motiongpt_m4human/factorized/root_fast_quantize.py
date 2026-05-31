from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..features import motion_process, recover_from_ric
from .local_vq import DEFAULT_CACHE_ROOT, FactorizedLocalStore
from .root_branch import _features_with_root_controls, _root_controls_from_features
from .root_fast_codec import (
    ROOT_CONTROL_DIM,
    dct_root_control_coefficients,
    idct_root_control_coefficients,
)


DEFAULT_OUT_DIR = "/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_quantized"


def _nearest_indices(x: np.ndarray, centroids: np.ndarray, batch_size: int = 32768) -> np.ndarray:
    labels = np.empty((x.shape[0],), dtype=np.int64)
    centroid_norm = np.sum(np.square(centroids), axis=1)[None, :]
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        block = x[start:end]
        dist = (
            np.sum(np.square(block), axis=1, keepdims=True)
            + centroid_norm
            - 2.0 * block @ centroids.T
        )
        labels[start:end] = np.argmin(dist, axis=1)
    return labels


def _kmeans_plus_plus(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centroids = np.empty((k, x.shape[1]), dtype=np.float32)
    first = int(rng.integers(0, x.shape[0]))
    centroids[0] = x[first]
    closest = np.sum(np.square(x - centroids[0]), axis=1)
    for idx in range(1, k):
        total = float(closest.sum())
        if not np.isfinite(total) or total <= 1e-12:
            centroids[idx:] = x[rng.choice(x.shape[0], size=k - idx, replace=True)]
            break
        next_idx = int(rng.choice(x.shape[0], p=closest / total))
        centroids[idx] = x[next_idx]
        dist = np.sum(np.square(x - centroids[idx]), axis=1)
        closest = np.minimum(closest, dist)
    return centroids


@dataclass
class VectorKMeansQuantizer:
    centroids_norm: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    normalize: bool

    @property
    def codebook_size(self) -> int:
        return int(self.centroids_norm.shape[0])

    def quantize(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self._normalize(vectors)
        labels = _nearest_indices(x, self.centroids_norm)
        quantized = self._denormalize(self.centroids_norm[labels])
        return quantized.astype(np.float32, copy=False), labels

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        vectors = vectors.astype(np.float32, copy=False)
        if not self.normalize:
            return vectors
        return ((vectors - self.mean) / self.std).astype(np.float32, copy=False)

    def _denormalize(self, vectors: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return vectors.astype(np.float32, copy=False)
        return (vectors * self.std + self.mean).astype(np.float32, copy=False)

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            centroids_norm=self.centroids_norm.astype(np.float32, copy=False),
            mean=self.mean.astype(np.float32, copy=False),
            std=self.std.astype(np.float32, copy=False),
            normalize=np.asarray(self.normalize, dtype=np.bool_),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )


@dataclass
class ProductKMeansQuantizer:
    quantizers: list[VectorKMeansQuantizer]
    coeff_count: int

    @property
    def groups(self) -> int:
        return len(self.quantizers)

    @property
    def codebook_size(self) -> int:
        return int(self.quantizers[0].codebook_size)

    def quantize(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vectors = vectors.astype(np.float32, copy=False)
        coeffs = vectors.reshape(vectors.shape[0], self.coeff_count, ROOT_CONTROL_DIM)
        quantized = np.empty_like(coeffs)
        codes = np.empty((vectors.shape[0], self.groups), dtype=np.int64)
        for dim, quantizer in enumerate(self.quantizers):
            q_dim, code_dim = quantizer.quantize(coeffs[:, :, dim])
            quantized[:, :, dim] = q_dim
            codes[:, dim] = code_dim
        return quantized.reshape(vectors.shape).astype(np.float32, copy=False), codes

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "coeff_count": np.asarray(self.coeff_count, dtype=np.int64),
            "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
        for idx, quantizer in enumerate(self.quantizers):
            arrays[f"centroids_norm_{idx}"] = quantizer.centroids_norm.astype(np.float32, copy=False)
            arrays[f"mean_{idx}"] = quantizer.mean.astype(np.float32, copy=False)
            arrays[f"std_{idx}"] = quantizer.std.astype(np.float32, copy=False)
            arrays[f"normalize_{idx}"] = np.asarray(quantizer.normalize, dtype=np.bool_)
        np.savez_compressed(path, **arrays)


@dataclass
class ResidualVectorKMeansQuantizer:
    quantizers: list[VectorKMeansQuantizer]

    @property
    def depth(self) -> int:
        return len(self.quantizers)

    @property
    def codebook_size(self) -> int:
        return int(self.quantizers[0].codebook_size)

    def quantize(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        residual = vectors.astype(np.float32, copy=True)
        quantized = np.zeros_like(residual)
        codes = np.empty((vectors.shape[0], self.depth), dtype=np.int64)
        for idx, quantizer in enumerate(self.quantizers):
            q_step, code_step = quantizer.quantize(residual)
            quantized += q_step
            residual -= q_step
            codes[:, idx] = code_step
        return quantized.astype(np.float32, copy=False), codes

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "depth": np.asarray(self.depth, dtype=np.int64),
            "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
        for idx, quantizer in enumerate(self.quantizers):
            arrays[f"centroids_norm_{idx}"] = quantizer.centroids_norm.astype(np.float32, copy=False)
            arrays[f"mean_{idx}"] = quantizer.mean.astype(np.float32, copy=False)
            arrays[f"std_{idx}"] = quantizer.std.astype(np.float32, copy=False)
            arrays[f"normalize_{idx}"] = np.asarray(quantizer.normalize, dtype=np.bool_)
        np.savez_compressed(path, **arrays)


@dataclass
class ScalarUniformQuantizer:
    lo: np.ndarray
    hi: np.ndarray
    bits: int

    @property
    def levels(self) -> int:
        return int(2 ** self.bits)

    def quantize(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vectors = vectors.astype(np.float32, copy=False)
        scale = np.maximum(self.hi - self.lo, 1e-8)
        normalized = np.clip((vectors - self.lo) / scale, 0.0, 1.0)
        codes = np.rint(normalized * (self.levels - 1)).astype(np.int64)
        quantized = self.lo + (codes.astype(np.float32) / float(self.levels - 1)) * scale
        return quantized.astype(np.float32, copy=False), codes

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            lo=self.lo.astype(np.float32, copy=False),
            hi=self.hi.astype(np.float32, copy=False),
            bits=np.asarray(self.bits, dtype=np.int64),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )


def _fit_vector_kmeans(
    vectors: np.ndarray,
    codebook_size: int,
    normalize: bool,
    max_iters: int,
    seed: int,
) -> tuple[VectorKMeansQuantizer, dict[str, Any]]:
    if vectors.ndim != 2:
        raise ValueError(f"Expected [N,D] vectors, got {vectors.shape}")
    if vectors.shape[0] == 0:
        raise ValueError("No vectors to fit")
    k = min(int(codebook_size), int(vectors.shape[0]))
    rng = np.random.default_rng(seed)
    mean = vectors.mean(axis=0).astype(np.float32)
    std = np.maximum(vectors.std(axis=0), 1e-6).astype(np.float32)
    x = ((vectors - mean) / std).astype(np.float32) if normalize else vectors.astype(np.float32)
    centroids = _kmeans_plus_plus(x, k, rng)
    labels = np.zeros((x.shape[0],), dtype=np.int64)
    prev_inertia = math.inf
    iterations = 0

    for step in range(max_iters):
        labels = _nearest_indices(x, centroids)
        new_centroids = np.zeros_like(centroids)
        np.add.at(new_centroids, labels, x)
        counts = np.bincount(labels, minlength=k).astype(np.float32)
        nonempty = counts > 0
        new_centroids[nonempty] /= counts[nonempty, None]
        if np.any(~nonempty):
            new_centroids[~nonempty] = x[rng.choice(x.shape[0], size=int((~nonempty).sum()), replace=True)]
        shift = float(np.mean(np.sum(np.square(new_centroids - centroids), axis=1)))
        centroids = new_centroids
        residual = x - centroids[labels]
        inertia = float(np.mean(np.sum(np.square(residual), axis=1)))
        iterations = step + 1
        if step > 0 and (
            abs(prev_inertia - inertia) <= 1e-6 * max(prev_inertia, 1.0)
            or shift <= 1e-8
        ):
            prev_inertia = inertia
            break
        prev_inertia = inertia

    quantizer = VectorKMeansQuantizer(
        centroids_norm=centroids.astype(np.float32, copy=False),
        mean=mean,
        std=std,
        normalize=normalize,
    )
    quantized, train_labels = quantizer.quantize(vectors)
    hist = np.bincount(train_labels, minlength=k).astype(np.float64)
    probs = hist / max(hist.sum(), 1.0)
    entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
    stats = {
        "train_vector_count": int(vectors.shape[0]),
        "requested_codebook_size": int(codebook_size),
        "codebook_size": int(k),
        "normalize": bool(normalize),
        "kmeans_iters": int(iterations),
        "kmeans_inertia_norm": float(prev_inertia),
        "train_coeff_mse": float(np.mean(np.square(quantized - vectors))),
        "train_coeff_l1": float(np.mean(np.abs(quantized - vectors))),
        "train_unique_codes": int(np.count_nonzero(hist)),
        "train_effective_vocab": float(math.exp(entropy)),
    }
    return quantizer, stats


def _fit_product_kmeans(
    vectors: np.ndarray,
    coeff_count: int,
    codebook_size: int,
    normalize: bool,
    max_iters: int,
    seed: int,
) -> tuple[ProductKMeansQuantizer, dict[str, Any]]:
    if vectors.ndim != 2 or vectors.shape[1] != coeff_count * ROOT_CONTROL_DIM:
        raise ValueError(f"Expected [N,{coeff_count * ROOT_CONTROL_DIM}] vectors, got {vectors.shape}")
    coeffs = vectors.reshape(vectors.shape[0], coeff_count, ROOT_CONTROL_DIM)
    quantizers: list[VectorKMeansQuantizer] = []
    group_stats: list[dict[str, Any]] = []
    for dim in range(ROOT_CONTROL_DIM):
        quantizer, stats = _fit_vector_kmeans(
            coeffs[:, :, dim].astype(np.float32, copy=False),
            codebook_size=codebook_size,
            normalize=normalize,
            max_iters=max_iters,
            seed=seed + dim * 10007,
        )
        quantizers.append(quantizer)
        group_stats.append(stats)
    product = ProductKMeansQuantizer(quantizers=quantizers, coeff_count=coeff_count)
    quantized, codes = product.quantize(vectors)
    stats = {
        "train_vector_count": int(vectors.shape[0]),
        "requested_codebook_size": int(codebook_size),
        "codebook_size": int(product.codebook_size),
        "product_groups": int(product.groups),
        "normalize": bool(normalize),
        "train_coeff_mse": float(np.mean(np.square(quantized - vectors))),
        "train_coeff_l1": float(np.mean(np.abs(quantized - vectors))),
        "train_unique_codes_per_group": [
            int(len(np.unique(codes[:, dim]))) for dim in range(codes.shape[1])
        ],
        "train_effective_vocab_per_group": [
            float(group["train_effective_vocab"]) for group in group_stats
        ],
        "group_stats": group_stats,
    }
    return product, stats


def _fit_rvq(
    vectors: np.ndarray,
    codebook_size: int,
    depth: int,
    normalize: bool,
    max_iters: int,
    seed: int,
) -> tuple[ResidualVectorKMeansQuantizer, dict[str, Any]]:
    if vectors.ndim != 2:
        raise ValueError(f"Expected [N,D] vectors, got {vectors.shape}")
    if depth <= 0:
        raise ValueError("depth must be positive")

    residual = vectors.astype(np.float32, copy=True)
    reconstruction = np.zeros_like(residual)
    quantizers: list[VectorKMeansQuantizer] = []
    stage_stats: list[dict[str, Any]] = []
    for stage in range(depth):
        quantizer, stats = _fit_vector_kmeans(
            residual,
            codebook_size=codebook_size,
            normalize=normalize,
            max_iters=max_iters,
            seed=seed + stage * 10007,
        )
        q_step, codes = quantizer.quantize(residual)
        reconstruction += q_step
        residual = vectors - reconstruction
        hist = np.bincount(codes, minlength=quantizer.codebook_size).astype(np.float64)
        probs = hist / max(hist.sum(), 1.0)
        entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
        stats = {
            **stats,
            "stage": int(stage),
            "residual_coeff_mse_after_stage": float(np.mean(np.square(residual))),
            "residual_coeff_l1_after_stage": float(np.mean(np.abs(residual))),
            "stage_unique_codes": int(np.count_nonzero(hist)),
            "stage_effective_vocab": float(math.exp(entropy)),
        }
        quantizers.append(quantizer)
        stage_stats.append(stats)

    rvq = ResidualVectorKMeansQuantizer(quantizers=quantizers)
    quantized, train_codes = rvq.quantize(vectors)
    stats = {
        "train_vector_count": int(vectors.shape[0]),
        "requested_codebook_size": int(codebook_size),
        "codebook_size": int(rvq.codebook_size),
        "rvq_depth": int(rvq.depth),
        "normalize": bool(normalize),
        "train_coeff_mse": float(np.mean(np.square(quantized - vectors))),
        "train_coeff_l1": float(np.mean(np.abs(quantized - vectors))),
        "train_unique_codes_per_stage": [
            int(len(np.unique(train_codes[:, stage]))) for stage in range(train_codes.shape[1])
        ],
        "train_effective_vocab_per_stage": [
            float(stage["stage_effective_vocab"]) for stage in stage_stats
        ],
        "stage_stats": stage_stats,
    }
    return rvq, stats


def _fit_scalar_uniform(
    vectors: np.ndarray,
    bits: int,
    range_quantile: float,
) -> tuple[ScalarUniformQuantizer, dict[str, Any]]:
    if vectors.ndim != 2:
        raise ValueError(f"Expected [N,D] vectors, got {vectors.shape}")
    if vectors.shape[0] == 0:
        raise ValueError("No vectors to fit")
    if not 0.0 < range_quantile <= 1.0:
        raise ValueError("range_quantile must be in (0, 1]")
    tail = (1.0 - range_quantile) * 0.5
    if range_quantile >= 1.0:
        lo = vectors.min(axis=0)
        hi = vectors.max(axis=0)
    else:
        lo = np.quantile(vectors, tail, axis=0)
        hi = np.quantile(vectors, 1.0 - tail, axis=0)
    quantizer = ScalarUniformQuantizer(
        lo=lo.astype(np.float32),
        hi=np.maximum(hi, lo + 1e-8).astype(np.float32),
        bits=int(bits),
    )
    quantized, codes = quantizer.quantize(vectors)
    stats = {
        "train_vector_count": int(vectors.shape[0]),
        "scalar_bits": int(bits),
        "scalar_levels": int(quantizer.levels),
        "range_quantile": float(range_quantile),
        "train_coeff_mse": float(np.mean(np.square(quantized - vectors))),
        "train_coeff_l1": float(np.mean(np.abs(quantized - vectors))),
        "train_unique_scalar_codes_mean": float(np.mean([
            len(np.unique(codes[:, dim])) for dim in range(codes.shape[1])
        ])),
    }
    return quantizer, stats


def _collect_train_vectors(
    args: argparse.Namespace,
    chunk_size: int,
    coeff_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    store = FactorizedLocalStore(
        args.cache_root,
        split=args.train_split,
        max_sequences=args.max_train_sequences,
        preload_features=True,
    )
    vectors: list[np.ndarray] = []
    window_count = 0
    chunk_count = 0
    example_windows: list[str] = []
    for item in store.iter_windows(
        args.window_frames,
        args.stride,
        args.min_window_frames,
        args.include_tail,
    ):
        features = item["features_263"].astype(np.float32, copy=False)
        root_controls = _root_controls_from_features(features, args.fps)[None]
        coeffs, _ = dct_root_control_coefficients(root_controls, chunk_size, coeff_count)
        flat = coeffs.reshape(-1, coeff_count * ROOT_CONTROL_DIM)
        vectors.append(flat)
        window_count += 1
        chunk_count += int(flat.shape[0])
        if len(example_windows) < 5:
            example_windows.append(f"{item['id']}:{item['start']}-{item['end']}")
    if not vectors:
        raise RuntimeError("No train DCT coefficient vectors collected")
    all_vectors = np.concatenate(vectors, axis=0).astype(np.float32, copy=False)
    if args.max_train_vectors and all_vectors.shape[0] > args.max_train_vectors:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(all_vectors.shape[0], size=args.max_train_vectors, replace=False)
        all_vectors = all_vectors[indices]
    stats = {
        "train_split": args.train_split,
        "train_window_count": int(window_count),
        "train_chunk_count_before_sampling": int(chunk_count),
        "train_vector_count": int(all_vectors.shape[0]),
        "train_vector_dim": int(all_vectors.shape[1]),
        "example_train_windows": example_windows,
    }
    return all_vectors, stats


def _empty_sums() -> dict[str, float]:
    return {
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
        "coeff_l1_sum": 0.0,
        "coeff_mse_sum": 0.0,
        "coeff_count": 0,
        "window_count": 0,
        "frame_count": 0,
    }


def _flush_eval(
    batch: list[dict[str, Any]],
    quantizer: VectorKMeansQuantizer | ProductKMeansQuantizer | ResidualVectorKMeansQuantizer | ScalarUniformQuantizer,
    quantizer_mode: str,
    chunk_size: int,
    coeff_count: int,
    fps: float,
    device: torch.device,
    sums: dict[str, float],
    code_hist: np.ndarray | None,
) -> None:
    features_np = np.stack([item["features_263"] for item in batch]).astype(np.float32, copy=False)
    root_controls = np.stack([
        _root_controls_from_features(item["features_263"], fps)
        for item in batch
    ]).astype(np.float32, copy=False)
    coeffs, frames = dct_root_control_coefficients(root_controls, chunk_size, coeff_count)
    flat = coeffs.reshape(-1, coeff_count * ROOT_CONTROL_DIM)
    quantized_flat, codes = quantizer.quantize(flat)
    quantized_coeffs = quantized_flat.reshape(coeffs.shape)
    pred_root_controls = idct_root_control_coefficients(quantized_coeffs, frames, chunk_size)
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

    coeff_diff = quantized_flat - flat
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
    sums["coeff_l1_sum"] += float(np.abs(coeff_diff).sum())
    sums["coeff_mse_sum"] += float(np.square(coeff_diff).sum())
    sums["coeff_count"] += int(coeff_diff.size)
    sums["window_count"] += len(batch)
    sums["frame_count"] += int(features_np.shape[0] * features_np.shape[1])
    if quantizer_mode == "vector" and code_hist is not None:
        np.add.at(code_hist, codes, 1)
    elif quantizer_mode in ("product", "rvq") and code_hist is not None:
        for group in range(codes.shape[1]):
            np.add.at(code_hist[group], codes[:, group], 1)


def evaluate_quantizer(
    args: argparse.Namespace,
    split: str,
    quantizer: VectorKMeansQuantizer | ProductKMeansQuantizer | ResidualVectorKMeansQuantizer | ScalarUniformQuantizer,
    quantizer_mode: str,
    train_stats: dict[str, Any],
    fit_stats: dict[str, Any],
    chunk_size: int,
    coeff_count: int,
) -> dict[str, Any]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    store = FactorizedLocalStore(
        args.cache_root,
        split=split,
        max_sequences=args.max_eval_sequences,
        preload_features=True,
    )
    sums = _empty_sums()
    code_hist = (
        np.zeros((quantizer.codebook_size,), dtype=np.int64)
        if quantizer_mode == "vector" and isinstance(quantizer, VectorKMeansQuantizer)
        else np.zeros((quantizer.groups, quantizer.codebook_size), dtype=np.int64)
        if quantizer_mode == "product" and isinstance(quantizer, ProductKMeansQuantizer)
        else np.zeros((quantizer.depth, quantizer.codebook_size), dtype=np.int64)
        if quantizer_mode == "rvq" and isinstance(quantizer, ResidualVectorKMeansQuantizer)
        else None
    )
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
                quantizer,
                quantizer_mode,
                chunk_size,
                coeff_count,
                args.fps,
                device,
                sums,
                code_hist,
            )
            pending[length] = []
    for batch in list(pending.values()):
        if batch:
            _flush_eval(
                batch,
                quantizer,
                quantizer_mode,
                chunk_size,
                coeff_count,
                args.fps,
                device,
                sums,
                code_hist,
            )

    chunks_per_window = int(math.ceil(args.window_frames / chunk_size)) if args.window_frames > 0 else None
    coeff_values = (
        chunks_per_window * coeff_count * ROOT_CONTROL_DIM
        if chunks_per_window is not None
        else None
    )
    raw_values = args.window_frames * ROOT_CONTROL_DIM if args.window_frames > 0 else None
    result: dict[str, Any] = {
        "cache_root": str(Path(args.cache_root).expanduser().resolve()),
        "split": split,
        "train_split": args.train_split,
        "window_frames": args.window_frames,
        "stride": args.stride,
        "include_tail": args.include_tail,
        "chunk_size": chunk_size,
        "coeff_count": coeff_count,
        "chunks_per_window": chunks_per_window,
        "root_fast_tokens_per_window": (
            chunks_per_window
            if quantizer_mode == "vector"
            else chunks_per_window * ROOT_CONTROL_DIM
            if quantizer_mode == "product" and chunks_per_window is not None
            else chunks_per_window * quantizer.depth
            if quantizer_mode == "rvq"
            and isinstance(quantizer, ResidualVectorKMeansQuantizer)
            and chunks_per_window is not None
            else None
        ),
        "scalar_codes_per_window": coeff_values if quantizer_mode == "scalar" else None,
        "coeff_values_per_window": coeff_values,
        "raw_root_values_per_window": raw_values,
        "coeff_values_vs_raw": (
            float(coeff_values) / float(raw_values)
            if coeff_values is not None and raw_values
            else None
        ),
        "raw_to_coeff_compression": (
            float(raw_values) / float(coeff_values)
            if coeff_values
            else None
        ),
        "quantizer_mode": quantizer_mode,
        "train_stats": train_stats,
        "fit_stats": fit_stats,
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
        "coeff_quant_l1": sums["coeff_l1_sum"] / max(sums["coeff_count"], 1),
        "coeff_quant_mse": sums["coeff_mse_sum"] / max(sums["coeff_count"], 1),
    }
    if quantizer_mode == "vector" and code_hist is not None:
        probs = code_hist.astype(np.float64) / max(float(code_hist.sum()), 1.0)
        entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
        vocab = int(len(code_hist))
        result.update({
            "codebook_size": vocab,
            "bits_per_window": (
                float(chunks_per_window) * math.log2(max(vocab, 2))
                if chunks_per_window is not None
                else None
            ),
            "unique_codes": int(np.count_nonzero(code_hist)),
            "effective_vocab": float(math.exp(entropy)),
        })
    elif quantizer_mode == "scalar" and isinstance(quantizer, ScalarUniformQuantizer):
        result.update({
            "scalar_bits": int(quantizer.bits),
            "scalar_levels": int(quantizer.levels),
            "bits_per_window": (
                float(coeff_values) * float(quantizer.bits)
                if coeff_values is not None
                else None
            ),
        })
    elif quantizer_mode == "product" and isinstance(quantizer, ProductKMeansQuantizer):
        product_stats: dict[str, Any] = {}
        if code_hist is not None:
            unique = []
            effective = []
            for group in range(code_hist.shape[0]):
                group_hist = code_hist[group]
                probs = group_hist.astype(np.float64) / max(float(group_hist.sum()), 1.0)
                entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
                unique.append(int(np.count_nonzero(group_hist)))
                effective.append(float(math.exp(entropy)))
            product_stats = {
                "unique_codes_per_group": unique,
                "effective_vocab_per_group": effective,
            }
        result.update({
            "codebook_size": int(quantizer.codebook_size),
            "product_groups": int(quantizer.groups),
            "bits_per_window": (
                float(chunks_per_window) * float(quantizer.groups) * math.log2(max(quantizer.codebook_size, 2))
                if chunks_per_window is not None
                else None
            ),
            **product_stats,
        })
    elif quantizer_mode == "rvq" and isinstance(quantizer, ResidualVectorKMeansQuantizer):
        rvq_stats: dict[str, Any] = {}
        if code_hist is not None:
            unique = []
            effective = []
            for stage in range(code_hist.shape[0]):
                stage_hist = code_hist[stage]
                probs = stage_hist.astype(np.float64) / max(float(stage_hist.sum()), 1.0)
                entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
                unique.append(int(np.count_nonzero(stage_hist)))
                effective.append(float(math.exp(entropy)))
            rvq_stats = {
                "unique_codes_per_stage": unique,
                "effective_vocab_per_stage": effective,
            }
        result.update({
            "codebook_size": int(quantizer.codebook_size),
            "rvq_depth": int(quantizer.depth),
            "bits_per_window": (
                float(chunks_per_window) * float(quantizer.depth) * math.log2(max(quantizer.codebook_size, 2))
                if chunks_per_window is not None
                else None
            ),
            **rvq_stats,
        })
    return result


def _write_result(out_dir: Path, name: str, result: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    quantizer_dir = out_dir / "quantizers"
    results: list[dict[str, Any]] = []
    modes = (
        ["vector", "product", "rvq", "scalar"]
        if args.mode == "all"
        else ["vector", "scalar"]
        if args.mode == "both"
        else [args.mode]
    )
    for chunk_size in args.chunk_sizes:
        for coeff_count in args.coeff_counts:
            if coeff_count > chunk_size:
                continue
            train_vectors, train_stats = _collect_train_vectors(args, chunk_size, coeff_count)
            if "vector" in modes:
                for codebook_size in args.codebook_sizes:
                    quantizer, fit_stats = _fit_vector_kmeans(
                        train_vectors,
                        codebook_size=codebook_size,
                        normalize=args.normalize,
                        max_iters=args.kmeans_iters,
                        seed=args.seed + chunk_size * 1000 + coeff_count * 100 + codebook_size,
                    )
                    quantizer_name = f"vector_chunk{chunk_size}_k{coeff_count}_vocab{quantizer.codebook_size}"
                    quantizer.save(
                        quantizer_dir / f"{quantizer_name}.npz",
                        {
                            "mode": "vector",
                            "chunk_size": chunk_size,
                            "coeff_count": coeff_count,
                            "codebook_size": quantizer.codebook_size,
                            "train_stats": train_stats,
                            "fit_stats": fit_stats,
                        },
                    )
                    for split in args.eval_splits:
                        result = evaluate_quantizer(
                            args,
                            split=split,
                            quantizer=quantizer,
                            quantizer_mode="vector",
                            train_stats=train_stats,
                            fit_stats=fit_stats,
                            chunk_size=chunk_size,
                            coeff_count=coeff_count,
                        )
                        name = f"vector_{split}_w{args.window_frames}_chunk{chunk_size}_k{coeff_count}_vocab{quantizer.codebook_size}"
                        _write_result(out_dir, name, result)
                        results.append(result)
                        print(json.dumps(result, indent=2, sort_keys=True))
            if "scalar" in modes:
                for bits in args.scalar_bits:
                    quantizer, fit_stats = _fit_scalar_uniform(
                        train_vectors,
                        bits=bits,
                        range_quantile=args.range_quantile,
                    )
                    quantizer_name = f"scalar_chunk{chunk_size}_k{coeff_count}_{bits}bit"
                    quantizer.save(
                        quantizer_dir / f"{quantizer_name}.npz",
                        {
                            "mode": "scalar",
                            "chunk_size": chunk_size,
                            "coeff_count": coeff_count,
                            "bits": bits,
                            "train_stats": train_stats,
                            "fit_stats": fit_stats,
                        },
                    )
                    for split in args.eval_splits:
                        result = evaluate_quantizer(
                            args,
                            split=split,
                            quantizer=quantizer,
                            quantizer_mode="scalar",
                            train_stats=train_stats,
                            fit_stats=fit_stats,
                            chunk_size=chunk_size,
                            coeff_count=coeff_count,
                        )
                        name = f"scalar_{split}_w{args.window_frames}_chunk{chunk_size}_k{coeff_count}_{bits}bit"
                        _write_result(out_dir, name, result)
                        results.append(result)
                        print(json.dumps(result, indent=2, sort_keys=True))
            if "product" in modes:
                for codebook_size in args.codebook_sizes:
                    quantizer, fit_stats = _fit_product_kmeans(
                        train_vectors,
                        coeff_count=coeff_count,
                        codebook_size=codebook_size,
                        normalize=args.normalize,
                        max_iters=args.kmeans_iters,
                        seed=args.seed + chunk_size * 1000 + coeff_count * 100 + codebook_size + 424242,
                    )
                    quantizer_name = f"product_chunk{chunk_size}_k{coeff_count}_vocab{quantizer.codebook_size}"
                    quantizer.save(
                        quantizer_dir / f"{quantizer_name}.npz",
                        {
                            "mode": "product",
                            "chunk_size": chunk_size,
                            "coeff_count": coeff_count,
                            "codebook_size": quantizer.codebook_size,
                            "product_groups": quantizer.groups,
                            "train_stats": train_stats,
                            "fit_stats": fit_stats,
                        },
                    )
                    for split in args.eval_splits:
                        result = evaluate_quantizer(
                            args,
                            split=split,
                            quantizer=quantizer,
                            quantizer_mode="product",
                            train_stats=train_stats,
                            fit_stats=fit_stats,
                            chunk_size=chunk_size,
                            coeff_count=coeff_count,
                        )
                        name = f"product_{split}_w{args.window_frames}_chunk{chunk_size}_k{coeff_count}_vocab{quantizer.codebook_size}"
                        _write_result(out_dir, name, result)
                        results.append(result)
                        print(json.dumps(result, indent=2, sort_keys=True))
            if "rvq" in modes:
                for codebook_size in args.codebook_sizes:
                    for depth in args.rvq_depths:
                        quantizer, fit_stats = _fit_rvq(
                            train_vectors,
                            codebook_size=codebook_size,
                            depth=depth,
                            normalize=args.normalize,
                            max_iters=args.kmeans_iters,
                            seed=args.seed + chunk_size * 1000 + coeff_count * 100 + codebook_size + depth * 17 + 777777,
                        )
                        quantizer_name = f"rvq_chunk{chunk_size}_k{coeff_count}_vocab{quantizer.codebook_size}_d{depth}"
                        quantizer.save(
                            quantizer_dir / f"{quantizer_name}.npz",
                            {
                                "mode": "rvq",
                                "chunk_size": chunk_size,
                                "coeff_count": coeff_count,
                                "codebook_size": quantizer.codebook_size,
                                "rvq_depth": quantizer.depth,
                                "train_stats": train_stats,
                                "fit_stats": fit_stats,
                            },
                        )
                        for split in args.eval_splits:
                            result = evaluate_quantizer(
                                args,
                                split=split,
                                quantizer=quantizer,
                                quantizer_mode="rvq",
                                train_stats=train_stats,
                                fit_stats=fit_stats,
                                chunk_size=chunk_size,
                                coeff_count=coeff_count,
                            )
                            name = f"rvq_{split}_w{args.window_frames}_chunk{chunk_size}_k{coeff_count}_vocab{quantizer.codebook_size}_d{depth}"
                            _write_result(out_dir, name, result)
                            results.append(result)
                            print(json.dumps(result, indent=2, sort_keys=True))

    results = sorted(
        results,
        key=lambda item: (
            item.get("split", ""),
            item.get("root_fast_tokens_per_window")
            if item.get("root_fast_tokens_per_window") is not None
            else 10_000,
            item.get("bits_per_window") or 0,
            item["mpjpe_mm"],
        ),
    )
    summary = {
        "cache_root": str(Path(args.cache_root).expanduser().resolve()),
        "train_split": args.train_split,
        "eval_splits": args.eval_splits,
        "window_frames": args.window_frames,
        "stride": args.stride,
        "chunk_sizes": args.chunk_sizes,
        "coeff_counts": args.coeff_counts,
        "mode": args.mode,
        "results": results,
    }
    summary_path = out_dir / f"w{args.window_frames}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantize Root-FAST DCT coefficients and evaluate reconstruction.")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--eval-splits", nargs="+", default=["val", "test"], choices=("train", "val", "test"))
    parser.add_argument("--window-frames", type=int, default=196)
    parser.add_argument("--stride", type=int, default=196)
    parser.add_argument("--include-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-window-frames", type=int, default=40)
    parser.add_argument("--max-train-sequences", type=int, default=0)
    parser.add_argument("--max-eval-sequences", type=int, default=0)
    parser.add_argument("--max-train-vectors", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--mode", default="vector", choices=("vector", "product", "rvq", "scalar", "both", "all"))
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[16, 32, 64, 98, 196])
    parser.add_argument("--coeff-counts", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--codebook-sizes", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--rvq-depths", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--scalar-bits", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--range-quantile", type=float, default=0.999)
    parser.add_argument("--kmeans-iters", type=int, default=50)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260531)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
