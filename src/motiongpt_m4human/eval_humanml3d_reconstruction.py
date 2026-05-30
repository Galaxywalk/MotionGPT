from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .features import recover_from_ric
from .vqvae import DEFAULT_CKPT, DEFAULT_MEAN, DEFAULT_STD, load_vqvae, resolve_device


DEFAULT_HUMANML_ROOT = "datasets/humanml3d"


def _read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _iter_rows(root: Path, split: str, max_sequences: int):
    ids = _read_ids(root / f"{split}.txt")
    yielded = 0
    for motion_id in ids:
        path = root / "new_joint_vecs" / f"{motion_id}.npy"
        if not path.exists():
            continue
        features = np.load(path).astype(np.float32, copy=False)
        if np.isfinite(features).all():
            yield motion_id, features
            yielded += 1
            if max_sequences and yielded >= max_sequences:
                break


def _window_starts(
    length: int,
    window_frames: int,
    stride: int,
    min_window_frames: int,
    include_tail: bool,
) -> list[tuple[int, int]]:
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


def _iter_windows(
    root: Path,
    split: str,
    window_frames: int,
    stride: int,
    min_window_frames: int,
    include_tail: bool,
    max_sequences: int,
    max_windows: int,
):
    yielded = 0
    for motion_id, features in _iter_rows(root, split, max_sequences):
        for start, frames in _window_starts(
            length=int(features.shape[0]),
            window_frames=window_frames,
            stride=stride,
            min_window_frames=min_window_frames,
            include_tail=include_tail,
        ):
            yield {
                "id": motion_id,
                "start": start,
                "end": start + frames,
                "features": features[start : start + frames],
            }
            yielded += 1
            if max_windows and yielded >= max_windows:
                return


def _mpjpe_sums(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, int]:
    err = torch.linalg.norm(pred - target, dim=-1)
    return float(err.sum().item()), int(err.numel())


def _flush_batch(
    batch: list[dict[str, Any]],
    model,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> dict[str, float | int | set[int]]:
    features = torch.as_tensor(np.stack([item["features"] for item in batch]), device=device, dtype=torch.float32)
    norm_features = (features - mean) / std
    with torch.no_grad():
        recon_norm, _, _ = model(norm_features)
        recon_features = recon_norm * std + mean
        ref_joints = recover_from_ric(features, 22)
        recon_joints = recover_from_ric(recon_features, 22)
        codes, _ = model.encode(norm_features)

    recon_sum, recon_count = _mpjpe_sums(recon_joints, ref_joints)
    ref_ra = ref_joints - ref_joints[..., :1, :]
    recon_ra = recon_joints - recon_joints[..., :1, :]
    root_sum, root_count = _mpjpe_sums(recon_ra, ref_ra)
    return {
        "recon_sum": recon_sum,
        "recon_count": recon_count,
        "root_sum": root_sum,
        "root_count": root_count,
        "feature_l1_sum": float(torch.abs(recon_features - features).sum().item()),
        "feature_l1_count": int(recon_features.numel()),
        "token_count": int(codes.numel()),
        "unique_tokens": set(int(v) for v in codes.detach().cpu().reshape(-1).tolist()),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.humanml_root).expanduser().resolve()
    device = resolve_device(args.device)
    mean = torch.from_numpy(np.load(args.mean).astype(np.float32)).to(device)
    std = torch.from_numpy(np.load(args.std).astype(np.float32)).to(device)
    model, ckpt_meta = load_vqvae(Path(args.checkpoint), device, args.calibration_domain)

    split_ids = _read_ids(root / f"{args.split}.txt")
    pending: dict[int, list[dict[str, Any]]] = {}
    sums = {
        "recon_sum": 0.0,
        "recon_count": 0,
        "root_sum": 0.0,
        "root_count": 0,
        "feature_l1_sum": 0.0,
        "feature_l1_count": 0,
        "token_count": 0,
    }
    unique_tokens: set[int] = set()
    example_windows: list[str] = []
    window_count = 0
    feature_frame_count = 0

    def flush(length: int) -> None:
        nonlocal unique_tokens
        batch = pending.get(length, [])
        if not batch:
            return
        stats = _flush_batch(batch, model, mean, std, device)
        for key in sums:
            sums[key] += stats[key]  # type: ignore[operator]
        unique_tokens.update(stats["unique_tokens"])  # type: ignore[arg-type]
        pending[length] = []

    iterator = _iter_windows(
        root=root,
        split=args.split,
        window_frames=args.window_frames,
        stride=args.stride,
        min_window_frames=args.min_window_frames,
        include_tail=args.include_tail,
        max_sequences=args.max_sequences,
        max_windows=args.max_windows,
    )
    for item in iterator:
        length = int(item["features"].shape[0])
        pending.setdefault(length, []).append(item)
        window_count += 1
        feature_frame_count += length
        if len(example_windows) < 5:
            example_windows.append(f"{item['id']}:{item['start']}-{item['end']}")
        if len(pending[length]) >= args.batch_size:
            flush(length)
        if args.log_every and window_count % args.log_every == 0:
            print(f"processed {window_count} windows, {feature_frame_count} feature frames")

    for length in list(pending):
        flush(length)

    if window_count == 0:
        raise RuntimeError("No evaluation windows were produced from HumanML3D")

    result = {
        **ckpt_meta,
        "source": "humanml3d",
        "humanml_root": str(root),
        "split": args.split,
        "split_sequence_count": len(split_ids),
        "window_frames": args.window_frames,
        "stride": args.stride,
        "include_tail": args.include_tail,
        "min_window_frames": args.min_window_frames,
        "window_count": window_count,
        "feature_frame_count": feature_frame_count,
        "device": str(device),
        "batch_size": args.batch_size,
        "vqvae_mpjpe_mm": sums["recon_sum"] / sums["recon_count"] * 1000.0,
        "vqvae_root_aligned_mpjpe_mm": sums["root_sum"] / sums["root_count"] * 1000.0,
        "feature_l1": sums["feature_l1_sum"] / sums["feature_l1_count"],
        "tokens_per_window_mean": sums["token_count"] / window_count,
        "unique_code_count": len(unique_tokens),
        "example_windows": example_windows,
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a MotionGPT VQVAE checkpoint on HumanML3D features.")
    parser.add_argument("--humanml-root", default=DEFAULT_HUMANML_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--mean", default=DEFAULT_MEAN)
    parser.add_argument("--std", default=DEFAULT_STD)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--window-frames", type=int, default=196, help="Use 0 to evaluate each sequence as one trimmed window.")
    parser.add_argument("--stride", type=int, default=196)
    parser.add_argument("--include-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-window-frames", type=int, default=40)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-domain", default="humanml3d")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--out-json", default="")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
