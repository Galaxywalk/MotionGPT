from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mGPT.archs.mgpt_vq import VQVae

from ..features import recover_from_ric
from .representation import (
    CONTACT_START,
    LOCAL_VEL_START,
    RIC_DIM,
    ROOT_DIM,
)


DEFAULT_CACHE_ROOT = "/cpfs01/liangbo/data/MotionGPT/factorized_cache/v1_m4human_xz-y_20hz"
DEFAULT_EXP_ROOT = "/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_v1"
LOCAL_JOINT_DIM = 21 * 3
LOCAL_VEL_DIM = 21 * 3
CONTACT_DIM = 4
LOCAL_DIM = LOCAL_JOINT_DIM + LOCAL_VEL_DIM + CONTACT_DIM


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    out = {f"p{q:02d}": float(np.percentile(arr, q)) for q in qs}
    out["mean"] = float(arr.mean())
    out["std"] = float(arr.std())
    return out


def factorized_arrays_to_local_vector(arrays: dict[str, np.ndarray]) -> np.ndarray:
    local_joints = arrays["local_joints"][:, 1:].reshape(-1, LOCAL_JOINT_DIM)
    local_vel = arrays["local_joint_vel"][:, 1:].reshape(-1, LOCAL_VEL_DIM)
    contacts = arrays["contacts"].reshape(-1, CONTACT_DIM)
    return np.concatenate([local_joints, local_vel, contacts], axis=-1).astype(np.float32, copy=False)


def local_vector_to_features(ref_features: np.ndarray, local_vector: np.ndarray) -> np.ndarray:
    pred = ref_features.copy()
    body_pos = local_vector[..., :LOCAL_JOINT_DIM]
    body_vel = local_vector[..., LOCAL_JOINT_DIM:LOCAL_JOINT_DIM + LOCAL_VEL_DIM]
    contacts = local_vector[..., -CONTACT_DIM:]

    pred[..., ROOT_DIM:ROOT_DIM + RIC_DIM] = body_pos
    pred[..., LOCAL_VEL_START + 3:LOCAL_VEL_START + 3 + LOCAL_VEL_DIM] = body_vel
    pred[..., CONTACT_START:CONTACT_START + CONTACT_DIM] = contacts
    return pred


class FactorizedLocalStore:
    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        max_sequences: int = 0,
        preload_features: bool = False,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        rows = [
            row for row in _load_jsonl(self.cache_root / "sequences.jsonl")
            if row.get("subset") == split
        ]
        if max_sequences:
            rows = rows[:max_sequences]
        if not rows:
            raise RuntimeError(f"No factorized rows for split={split} under {self.cache_root}")

        self.rows = rows
        self.sequences: list[dict[str, Any]] = []
        for row in rows:
            arrays = dict(np.load(self.cache_root / row["factorized"], allow_pickle=False))
            local = factorized_arrays_to_local_vector(arrays)
            item = {
                "id": row["id"],
                "local": local,
                "length": int(local.shape[0]),
                "row": row,
            }
            if preload_features:
                item["features_263"] = arrays["features_263"].astype(np.float32, copy=False)
            self.sequences.append(item)

        self.eligible: dict[int, list[int]] = {}

    def __len__(self) -> int:
        return len(self.sequences)

    def _eligible_indices(self, window_size: int) -> list[int]:
        if window_size not in self.eligible:
            self.eligible[window_size] = [
                idx for idx, seq in enumerate(self.sequences)
                if seq["length"] >= window_size
            ]
        if not self.eligible[window_size]:
            raise RuntimeError(f"No sequences with at least {window_size} frames")
        return self.eligible[window_size]

    def compute_stats(self) -> tuple[np.ndarray, np.ndarray]:
        local = np.concatenate([seq["local"] for seq in self.sequences], axis=0)
        mean = local.mean(axis=0).astype(np.float32)
        std = local.std(axis=0).astype(np.float32)
        std = np.maximum(std, 1e-4).astype(np.float32)
        return mean, std

    def sample_batch(
        self,
        batch_size: int,
        window_size: int,
        rng: random.Random,
    ) -> np.ndarray:
        eligible = self._eligible_indices(window_size)
        batch = []
        for _ in range(batch_size):
            seq = self.sequences[rng.choice(eligible)]
            start = rng.randint(0, seq["length"] - window_size)
            batch.append(seq["local"][start:start + window_size])
        return np.stack(batch).astype(np.float32, copy=False)

    def iter_windows(
        self,
        window_frames: int,
        stride: int,
        min_window_frames: int,
        include_tail: bool,
    ):
        for seq in self.sequences:
            length = seq["length"]
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
                yield {
                    "id": seq["id"],
                    "start": start,
                    "end": start + frames,
                    "local": seq["local"][start:start + frames],
                    "features_263": seq["features_263"][start:start + frames],
                }


def build_model(args: argparse.Namespace) -> VQVae:
    return VQVae(
        nfeats=LOCAL_DIM,
        quantizer=args.quantizer,
        code_num=args.code_num,
        code_dim=args.code_dim,
        output_emb_width=args.code_dim,
        down_t=args.down_t,
        stride_t=args.stride_t,
        width=args.width,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        norm=None,
        activation="relu",
    )


def _loss_terms(
    pred_norm: torch.Tensor,
    ref_norm: torch.Tensor,
    pred_raw: torch.Tensor,
    ref_raw: torch.Tensor,
    commit_loss: torch.Tensor,
    perplexity: torch.Tensor,
    fps: float,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    joint_pred = pred_raw[..., :LOCAL_JOINT_DIM]
    joint_ref = ref_raw[..., :LOCAL_JOINT_DIM]
    vel_pred = pred_raw[..., LOCAL_JOINT_DIM:LOCAL_JOINT_DIM + LOCAL_VEL_DIM] * fps
    vel_ref = ref_raw[..., LOCAL_JOINT_DIM:LOCAL_JOINT_DIM + LOCAL_VEL_DIM] * fps
    contact_pred = pred_raw[..., -CONTACT_DIM:]
    contact_ref = ref_raw[..., -CONTACT_DIM:]

    feature = F.smooth_l1_loss(pred_norm, ref_norm)
    joint = F.smooth_l1_loss(joint_pred, joint_ref)
    vel = F.smooth_l1_loss(vel_pred, vel_ref)
    contact = F.mse_loss(contact_pred, contact_ref)
    total = (
        args.lambda_feature * feature
        + args.lambda_joint * joint
        + args.lambda_velocity * vel
        + args.lambda_contact * contact
        + args.lambda_commit * commit_loss
    )
    return {
        "loss": total,
        "feature": feature.detach(),
        "joint_m": joint.detach(),
        "velocity_mps": vel.detach(),
        "contact": contact.detach(),
        "commit": commit_loss.detach(),
        "perplexity": perplexity.detach(),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    exp_root = Path(args.exp_root).expanduser().resolve()
    ckpt_dir = exp_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_store = FactorizedLocalStore(
        args.cache_root,
        split=args.train_split,
        max_sequences=args.max_train_sequences,
        preload_features=False,
    )
    mean_np, std_np = train_store.compute_stats()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs * args.steps_per_epoch, 1),
        eta_min=args.lr_min,
    )

    window_sizes = [int(item) for item in args.window_sizes]
    window_weights = [float(item) for item in args.window_weights]
    history: list[dict[str, float]] = []
    best_metric = math.inf
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    for epoch in range(args.epochs):
        model.train()
        accum: dict[str, float] = {}
        for _ in range(args.steps_per_epoch):
            window_size = rng.choices(window_sizes, weights=window_weights, k=1)[0]
            batch_np = train_store.sample_batch(args.batch_size, window_size, rng)
            batch = torch.from_numpy(batch_np).to(device)
            batch_norm = (batch - mean) / std
            pred_norm, commit_loss, perplexity = model(batch_norm)
            pred_raw = pred_norm * std + mean
            losses = _loss_terms(
                pred_norm,
                batch_norm,
                pred_raw,
                batch,
                commit_loss,
                perplexity,
                fps=args.fps,
                args=args,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            for key, value in losses.items():
                accum[key] = accum.get(key, 0.0) + float(value.item())

        summary = {
            key: value / args.steps_per_epoch
            for key, value in accum.items()
        }
        summary["epoch"] = epoch
        summary["lr"] = float(scheduler.get_last_lr()[0])
        history.append(summary)
        print(
            f"epoch {epoch}: loss={summary['loss']:.5f} "
            f"feature={summary['feature']:.5f} joint={summary['joint_m']:.5f} "
            f"vel={summary['velocity_mps']:.5f} contact={summary['contact']:.5f} "
            f"perplexity={summary['perplexity']:.2f}"
        )

        clean_args = {
            key: value for key, value in vars(args).items()
            if key != "func"
        }
        payload = {
            "model": model.state_dict(),
            "mean": mean_np,
            "std": std_np,
            "args": clean_args,
            "epoch": epoch,
            "history": history,
        }
        torch.save(payload, last_path)
        if summary["loss"] < best_metric:
            best_metric = summary["loss"]
            torch.save(payload, best_path)

    result = {
        "exp_root": str(exp_root),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "train_sequence_count": len(train_store),
        "history": history,
    }
    (exp_root / "train_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _load_checkpoint(path: Path, device: torch.device) -> tuple[VQVae, torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    ckpt_args = argparse.Namespace(**payload["args"])
    model = build_model(ckpt_args).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    mean = torch.as_tensor(payload["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(payload["std"], dtype=torch.float32, device=device)
    return model, mean, std, payload


def _flush_eval(
    batch: list[dict[str, Any]],
    model: VQVae,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    sums: dict[str, float],
    code_hist: torch.Tensor,
    fps: float,
) -> None:
    local_np = np.stack([item["local"] for item in batch]).astype(np.float32, copy=False)
    features_np = np.stack([item["features_263"] for item in batch]).astype(np.float32, copy=False)
    local = torch.from_numpy(local_np).to(device)
    local_norm = (local - mean) / std
    with torch.no_grad():
        pred_norm, _, perplexity = model(local_norm)
        pred = pred_norm * std + mean
        codes, _ = model.encode(local_norm)
        code_hist += torch.bincount(codes.reshape(-1).cpu(), minlength=code_hist.numel())

    pred_np = pred.detach().cpu().numpy()
    pred_features_np = local_vector_to_features(features_np, pred_np)
    ref_features = torch.from_numpy(features_np).to(device)
    pred_features = torch.from_numpy(pred_features_np).to(device)
    with torch.no_grad():
        ref_joints = recover_from_ric(ref_features, 22)
        pred_joints = recover_from_ric(pred_features, 22)

    joint_err = torch.linalg.norm(pred_joints - ref_joints, dim=-1)
    ref_ra = ref_joints - ref_joints[..., :1, :]
    pred_ra = pred_joints - pred_joints[..., :1, :]
    ra_err = torch.linalg.norm(pred_ra - ref_ra, dim=-1)
    body_err = torch.linalg.norm(
        pred[..., :LOCAL_JOINT_DIM].reshape(pred.shape[0], pred.shape[1], 21, 3)
        - local[..., :LOCAL_JOINT_DIM].reshape(local.shape[0], local.shape[1], 21, 3),
        dim=-1,
    )
    vel_err = torch.linalg.norm(
        (
            pred[..., LOCAL_JOINT_DIM:LOCAL_JOINT_DIM + LOCAL_VEL_DIM]
            - local[..., LOCAL_JOINT_DIM:LOCAL_JOINT_DIM + LOCAL_VEL_DIM]
        ).reshape(local.shape[0], local.shape[1], 21, 3),
        dim=-1,
    ) * fps

    contact_ref = local[..., -CONTACT_DIM:]
    contact_pred = pred[..., -CONTACT_DIM:]
    contact_bin = (contact_pred > 0.5).float()
    tp = ((contact_bin == 1) & (contact_ref > 0.5)).float().sum()
    fp = ((contact_bin == 1) & (contact_ref <= 0.5)).float().sum()
    fn = ((contact_bin == 0) & (contact_ref > 0.5)).float().sum()
    acc = ((contact_bin > 0.5) == (contact_ref > 0.5)).float().sum()

    sums["mpjpe_sum"] += float(joint_err.sum().item())
    sums["mpjpe_count"] += int(joint_err.numel())
    sums["ra_sum"] += float(ra_err.sum().item())
    sums["ra_count"] += int(ra_err.numel())
    sums["body_sum"] += float(body_err.sum().item())
    sums["body_count"] += int(body_err.numel())
    sums["vel_sum"] += float(vel_err.sum().item())
    sums["vel_count"] += int(vel_err.numel())
    sums["feature_l1_sum"] += float(torch.abs(pred - local).sum().item())
    sums["feature_l1_count"] += int(pred.numel())
    sums["contact_acc_sum"] += float(acc.item())
    sums["contact_count"] += int(contact_ref.numel())
    sums["contact_tp"] += float(tp.item())
    sums["contact_fp"] += float(fp.item())
    sums["contact_fn"] += float(fn.item())
    sums["perplexity_sum"] += float(perplexity.item()) * len(batch)
    sums["window_count"] += len(batch)
    sums["frame_count"] += int(local.shape[0] * local.shape[1])


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, mean, std, payload = _load_checkpoint(Path(args.checkpoint), device)
    store = FactorizedLocalStore(
        args.cache_root,
        split=args.split,
        max_sequences=args.max_sequences,
        preload_features=True,
    )
    code_num = int(payload["args"]["code_num"])
    code_hist = torch.zeros(code_num, dtype=torch.long)
    sums = {
        "mpjpe_sum": 0.0,
        "mpjpe_count": 0,
        "ra_sum": 0.0,
        "ra_count": 0,
        "body_sum": 0.0,
        "body_count": 0,
        "vel_sum": 0.0,
        "vel_count": 0,
        "feature_l1_sum": 0.0,
        "feature_l1_count": 0,
        "contact_acc_sum": 0.0,
        "contact_count": 0,
        "contact_tp": 0.0,
        "contact_fp": 0.0,
        "contact_fn": 0.0,
        "perplexity_sum": 0.0,
        "window_count": 0,
        "frame_count": 0,
    }
    pending: dict[int, list[dict[str, Any]]] = {}
    example_windows: list[str] = []
    for item in store.iter_windows(
        window_frames=args.window_frames,
        stride=args.stride,
        min_window_frames=args.min_window_frames,
        include_tail=args.include_tail,
    ):
        length = int(item["local"].shape[0])
        pending.setdefault(length, []).append(item)
        if len(example_windows) < 5:
            example_windows.append(f"{item['id']}:{item['start']}-{item['end']}")
        if len(pending[length]) >= args.batch_size:
            _flush_eval(pending[length], model, mean, std, device, sums, code_hist, args.fps)
            pending[length] = []
    for length, batch in list(pending.items()):
        if batch:
            _flush_eval(batch, model, mean, std, device, sums, code_hist, args.fps)

    precision = sums["contact_tp"] / max(sums["contact_tp"] + sums["contact_fp"], 1.0)
    recall = sums["contact_tp"] / max(sums["contact_tp"] + sums["contact_fn"], 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    used_codes = int((code_hist > 0).sum().item())
    probs = code_hist.float() / max(int(code_hist.sum().item()), 1)
    entropy = float((-(probs[probs > 0] * torch.log(probs[probs > 0])).sum()).item())
    effective_codes = float(math.exp(entropy))

    result = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "cache_root": str(Path(args.cache_root).expanduser().resolve()),
        "split": args.split,
        "window_frames": args.window_frames,
        "stride": args.stride,
        "include_tail": args.include_tail,
        "example_windows": example_windows,
        "window_count": int(sums["window_count"]),
        "frame_count": int(sums["frame_count"]),
        "tokens_per_window_mean": (
            float(code_hist.sum().item()) / max(sums["window_count"], 1)),
        "code_num": code_num,
        "unique_code_count": used_codes,
        "effective_code_count": effective_codes,
        "perplexity_mean": sums["perplexity_sum"] / max(sums["window_count"], 1),
        "feature_l1": sums["feature_l1_sum"] / max(sums["feature_l1_count"], 1),
        "mpjpe_mm": sums["mpjpe_sum"] / max(sums["mpjpe_count"], 1) * 1000.0,
        "root_aligned_mpjpe_mm": sums["ra_sum"] / max(sums["ra_count"], 1) * 1000.0,
        "local_body_mpjpe_mm": sums["body_sum"] / max(sums["body_count"], 1) * 1000.0,
        "local_velocity_error_mm_per_s": sums["vel_sum"] / max(sums["vel_count"], 1) * 1000.0,
        "contact_accuracy": sums["contact_acc_sum"] / max(sums["contact_count"], 1),
        "contact_precision": precision,
        "contact_recall": recall,
        "contact_f1": f1,
    }
    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate factorized local-only VQ.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    train_p.add_argument("--exp-root", default=DEFAULT_EXP_ROOT)
    train_p.add_argument("--train-split", default="train")
    train_p.add_argument("--max-train-sequences", type=int, default=0)
    train_p.add_argument("--device", default="auto")
    train_p.add_argument("--seed", type=int, default=1234)
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--steps-per-epoch", type=int, default=100)
    train_p.add_argument("--batch-size", type=int, default=256)
    train_p.add_argument("--window-sizes", nargs="+", type=int, default=[64, 128, 196])
    train_p.add_argument("--window-weights", nargs="+", type=float, default=[0.25, 0.25, 0.5])
    train_p.add_argument("--fps", type=float, default=20.0)
    train_p.add_argument("--lr", type=float, default=2e-4)
    train_p.add_argument("--lr-min", type=float, default=1e-6)
    train_p.add_argument("--weight-decay", type=float, default=0.0)
    train_p.add_argument("--grad-clip", type=float, default=1.0)
    train_p.add_argument("--lambda-feature", type=float, default=1.0)
    train_p.add_argument("--lambda-joint", type=float, default=5.0)
    train_p.add_argument("--lambda-velocity", type=float, default=0.5)
    train_p.add_argument("--lambda-contact", type=float, default=0.5)
    train_p.add_argument("--lambda-commit", type=float, default=0.02)
    train_p.add_argument("--quantizer", default="ema_reset")
    train_p.add_argument("--code-num", type=int, default=512)
    train_p.add_argument("--code-dim", type=int, default=512)
    train_p.add_argument("--width", type=int, default=512)
    train_p.add_argument("--depth", type=int, default=3)
    train_p.add_argument("--down-t", type=int, default=2)
    train_p.add_argument("--stride-t", type=int, default=2)
    train_p.add_argument("--dilation-growth-rate", type=int, default=3)
    train_p.set_defaults(func=train)

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    eval_p.add_argument("--split", default="test", choices=("train", "val", "test"))
    eval_p.add_argument("--device", default="auto")
    eval_p.add_argument("--batch-size", type=int, default=512)
    eval_p.add_argument("--window-frames", type=int, default=196)
    eval_p.add_argument("--stride", type=int, default=196)
    eval_p.add_argument("--include-tail", action=argparse.BooleanOptionalAction, default=True)
    eval_p.add_argument("--min-window-frames", type=int, default=40)
    eval_p.add_argument("--max-sequences", type=int, default=0)
    eval_p.add_argument("--fps", type=float, default=20.0)
    eval_p.add_argument("--out-json", default="")
    eval_p.set_defaults(func=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
