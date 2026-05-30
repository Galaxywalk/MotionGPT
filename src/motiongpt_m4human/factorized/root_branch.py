from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features import motion_process, recover_from_ric
from .local_vq import (
    DEFAULT_CACHE_ROOT,
    FactorizedLocalStore,
    LOCAL_DIM,
    LOCAL_JOINT_DIM,
    LOCAL_VEL_DIM,
    CONTACT_DIM,
    _load_checkpoint as load_local_vq_checkpoint,
    local_vector_to_features,
)


DEFAULT_EXP_ROOT = "/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_branch_m4human_v1"
DEFAULT_LOCAL_CKPT = "/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_v1/checkpoints/best.pt"
ROOT_DIM = 4


def _root_controls_from_features(features: np.ndarray, fps: float) -> np.ndarray:
    root = np.zeros((features.shape[0], ROOT_DIM), dtype=np.float32)
    root[:, 0:1] = features[:, 0:1] * fps
    root[:, 1:3] = features[:, 1:3] * fps
    root[:, 3:4] = features[:, 3:4]
    return root


def _features_with_root_controls(ref_features: np.ndarray, root_controls: np.ndarray, fps: float) -> np.ndarray:
    out = ref_features.copy()
    out[..., 0:1] = root_controls[..., 0:1] / fps
    out[..., 1:3] = root_controls[..., 1:3] / fps
    out[..., 3:4] = root_controls[..., 3:4]
    return out


def _compute_stats(store: FactorizedLocalStore, fps: float) -> tuple[np.ndarray, np.ndarray]:
    roots = []
    for seq in store.sequences:
        features = seq.get("features_263")
        if features is None:
            arrays = np.load(store.cache_root / seq["row"]["factorized"], allow_pickle=False)
            features = arrays["features_263"]
        roots.append(_root_controls_from_features(features, fps))
    root = np.concatenate(roots, axis=0)
    mean = root.mean(axis=0).astype(np.float32)
    std = np.maximum(root.std(axis=0), 1e-4).astype(np.float32)
    return mean, std


class RootBranch(nn.Module):
    def __init__(self, local_dim: int = LOCAL_DIM, root_dim: int = ROOT_DIM, width: int = 128, latent_width: int = 128):
        super().__init__()
        self.local_proj = nn.Sequential(
            nn.Conv1d(local_dim, width, 1),
            nn.GELU(),
            nn.Conv1d(width, width, 1),
        )
        self.enc1 = nn.Sequential(
            nn.Conv1d(root_dim + width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.down1 = nn.Conv1d(width, width, 4, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(width, latent_width, 3, padding=1),
            nn.GELU(),
        )
        self.down2 = nn.Conv1d(latent_width, latent_width, 4, stride=2, padding=1)
        self.mid = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(latent_width, latent_width, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(latent_width, latent_width, 3, padding=1),
        )
        self.up2 = nn.Sequential(
            nn.Conv1d(latent_width + width, width, 3, padding=1),
            nn.GELU(),
        )
        self.up1 = nn.Sequential(
            nn.Conv1d(width + width, width, 3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Conv1d(width, root_dim, 1)

    def forward(self, root_norm: torch.Tensor, local_norm: torch.Tensor) -> torch.Tensor:
        root = root_norm.permute(0, 2, 1)
        local = local_norm.permute(0, 2, 1)
        cond = self.local_proj(local)
        h1 = self.enc1(torch.cat([root, cond], dim=1))
        h2 = self.enc2(self.down1(h1))
        z = self.mid(self.down2(h2))
        y = F.interpolate(z, size=h2.shape[-1], mode="nearest")
        y = self.up2(torch.cat([y, h2], dim=1))
        y = F.interpolate(y, size=h1.shape[-1], mode="nearest")
        y = self.up1(torch.cat([y, h1], dim=1))
        return self.out(y).permute(0, 2, 1)


class BottleneckRootBranch(nn.Module):
    def __init__(self, local_dim: int = LOCAL_DIM, root_dim: int = ROOT_DIM, width: int = 128, latent_width: int = 128):
        super().__init__()
        self.local_proj = nn.Sequential(
            nn.Conv1d(local_dim, width, 1),
            nn.GELU(),
            nn.Conv1d(width, width, 1),
        )
        self.enc = nn.Sequential(
            nn.Conv1d(root_dim + width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(width, width, 4, stride=2, padding=1),
            nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(width, latent_width, 4, stride=2, padding=1),
            nn.GELU(),
        )
        self.mid = nn.Sequential(
            nn.Conv1d(latent_width, latent_width, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(latent_width, latent_width, 3, padding=1),
            nn.GELU(),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(latent_width + width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Conv1d(width, root_dim, 1)

    def encode(self, root_norm: torch.Tensor, local_norm: torch.Tensor) -> torch.Tensor:
        root = root_norm.permute(0, 2, 1)
        local = local_norm.permute(0, 2, 1)
        cond = self.local_proj(local)
        h = self.enc(torch.cat([root, cond], dim=1))
        return self.mid(self.down2(self.down1(h)))

    def decode(self, latent: torch.Tensor, local_norm: torch.Tensor) -> torch.Tensor:
        local = local_norm.permute(0, 2, 1)
        cond = self.local_proj(local)
        y = F.interpolate(latent, size=cond.shape[-1], mode="nearest")
        y = self.dec(torch.cat([y, cond], dim=1))
        return self.out(y).permute(0, 2, 1)

    def forward(self, root_norm: torch.Tensor, local_norm: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(root_norm, local_norm), local_norm)


def build_root_model(args: argparse.Namespace) -> nn.Module:
    architecture = getattr(args, "architecture", "unet")
    if architecture == "unet":
        return RootBranch(width=args.width, latent_width=args.latent_width)
    if architecture == "bottleneck":
        return BottleneckRootBranch(width=args.width, latent_width=args.latent_width)
    raise ValueError(f"Unsupported root branch architecture: {architecture}")


def _recover_root_xz(features: torch.Tensor) -> torch.Tensor:
    _, root = motion_process.recover_root_rot_pos(features)
    return root[..., [0, 2]]


def _root_losses(
    pred_norm: torch.Tensor,
    ref_norm: torch.Tensor,
    pred_root: torch.Tensor,
    ref_root: torch.Tensor,
    pred_features: torch.Tensor,
    ref_features: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    control = F.smooth_l1_loss(pred_norm, ref_norm)
    yaw = F.smooth_l1_loss(pred_root[..., 0:1], ref_root[..., 0:1])
    vel = F.smooth_l1_loss(pred_root[..., 1:3], ref_root[..., 1:3])
    height = F.smooth_l1_loss(pred_root[..., 3:4], ref_root[..., 3:4])

    pred_xz = _recover_root_xz(pred_features)
    ref_xz = _recover_root_xz(ref_features)
    global_pos = F.smooth_l1_loss(pred_xz, ref_xz)
    pred_steps_vec = pred_xz[:, 1:] - pred_xz[:, :-1]
    ref_steps_vec = ref_xz[:, 1:] - ref_xz[:, :-1]
    global_vel = F.smooth_l1_loss(pred_steps_vec, ref_steps_vec)
    final = F.smooth_l1_loss(pred_xz[:, -1] - pred_xz[:, 0], ref_xz[:, -1] - ref_xz[:, 0])
    path = F.smooth_l1_loss(
        torch.linalg.norm(pred_steps_vec, dim=-1).sum(dim=-1),
        torch.linalg.norm(ref_steps_vec, dim=-1).sum(dim=-1),
    )
    accel = pred_steps_vec[:, 1:] - pred_steps_vec[:, :-1]
    smooth = (accel ** 2).mean()
    total = (
        args.lambda_control * control
        + args.lambda_yaw * yaw
        + args.lambda_velocity * vel
        + args.lambda_height * height
        + args.lambda_global_pos * global_pos
        + args.lambda_global_vel * global_vel
        + args.lambda_final * final
        + args.lambda_path * path
        + args.lambda_smooth * smooth
    )
    return {
        "loss": total,
        "control": control.detach(),
        "yaw": yaw.detach(),
        "velocity": vel.detach(),
        "height": height.detach(),
        "global_pos": global_pos.detach(),
        "global_vel": global_vel.detach(),
        "final": final.detach(),
        "path": path.detach(),
        "smooth": smooth.detach(),
    }


def _local_vq_reconstruct(
    local_vq,
    local_mean: torch.Tensor,
    local_std: torch.Tensor,
    local_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_norm = (local_raw - local_mean) / local_std
    with torch.no_grad():
        pred_norm, _, _ = local_vq(local_norm)
    pred_raw = pred_norm * local_std + local_mean
    return pred_raw, pred_norm


def _sample_root_batch(
    store: FactorizedLocalStore,
    batch_size: int,
    window_size: int,
    rng: random.Random,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible = store._eligible_indices(window_size)
    local_items = []
    root_items = []
    feature_items = []
    for _ in range(batch_size):
        seq = store.sequences[rng.choice(eligible)]
        features = seq["features_263"]
        start = rng.randint(0, seq["length"] - window_size)
        local_items.append(seq["local"][start:start + window_size])
        feature = features[start:start + window_size]
        feature_items.append(feature)
        root_items.append(_root_controls_from_features(feature, fps))
    return (
        np.stack(local_items).astype(np.float32, copy=False),
        np.stack(root_items).astype(np.float32, copy=False),
        np.stack(feature_items).astype(np.float32, copy=False),
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    exp_root = Path(args.exp_root).expanduser().resolve()
    ckpt_dir = exp_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    store = FactorizedLocalStore(
        args.cache_root,
        split=args.train_split,
        max_sequences=args.max_train_sequences,
        preload_features=True,
    )
    root_mean_np, root_std_np = _compute_stats(store, args.fps)
    root_mean = torch.from_numpy(root_mean_np).to(device)
    root_std = torch.from_numpy(root_std_np).to(device)
    local_vq, local_mean, local_std, local_payload = load_local_vq_checkpoint(Path(args.local_vq_checkpoint), device)
    local_vq.eval()
    model = build_root_model(args).to(device)
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
            local_np, root_np, features_np = _sample_root_batch(
                store,
                args.batch_size,
                window_size,
                rng,
                args.fps,
            )
            local_raw = torch.from_numpy(local_np).to(device)
            root_raw = torch.from_numpy(root_np).to(device)
            ref_features = torch.from_numpy(features_np).to(device)
            local_pred_raw, local_pred_norm = _local_vq_reconstruct(
                local_vq,
                local_mean,
                local_std,
                local_raw,
            )
            root_norm = (root_raw - root_mean) / root_std
            pred_norm = model(root_norm, local_pred_norm)
            pred_root = pred_norm * root_std + root_mean
            pred_features_np = _features_with_root_controls(features_np, pred_root.detach().cpu().numpy(), args.fps)
            pred_features = torch.from_numpy(pred_features_np).to(device)
            pred_features[..., 0:4] = torch.cat([
                pred_root[..., 0:1] / args.fps,
                pred_root[..., 1:3] / args.fps,
                pred_root[..., 3:4],
            ], dim=-1)
            losses = _root_losses(
                pred_norm,
                root_norm,
                pred_root,
                root_raw,
                pred_features,
                ref_features,
                args,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            for key, value in losses.items():
                accum[key] = accum.get(key, 0.0) + float(value.item())
        summary = {key: value / args.steps_per_epoch for key, value in accum.items()}
        summary["epoch"] = epoch
        summary["lr"] = float(scheduler.get_last_lr()[0])
        history.append(summary)
        print(
            f"epoch {epoch}: loss={summary['loss']:.6f} control={summary['control']:.6f} "
            f"vel={summary['velocity']:.6f} global={summary['global_pos']:.6f} "
            f"final={summary['final']:.6f} path={summary['path']:.6f}"
        )
        clean_args = {key: value for key, value in vars(args).items() if key != "func"}
        payload = {
            "model": model.state_dict(),
            "root_mean": root_mean_np,
            "root_std": root_std_np,
            "args": clean_args,
            "epoch": epoch,
            "history": history,
            "local_vq_epoch": local_payload.get("epoch"),
        }
        torch.save(payload, last_path)
        if summary["loss"] < best_metric:
            best_metric = summary["loss"]
            torch.save(payload, best_path)

    result = {
        "exp_root": str(exp_root),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history": history,
    }
    (exp_root / "train_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _load_root_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    ckpt_args = argparse.Namespace(**payload["args"])
    if not hasattr(ckpt_args, "architecture"):
        ckpt_args.architecture = "unet"
    model = build_root_model(ckpt_args).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    root_mean = torch.as_tensor(payload["root_mean"], dtype=torch.float32, device=device)
    root_std = torch.as_tensor(payload["root_std"], dtype=torch.float32, device=device)
    return model, root_mean, root_std, payload


def _flush_eval(
    batch: list[dict[str, Any]],
    root_model: RootBranch,
    root_mean: torch.Tensor,
    root_std: torch.Tensor,
    local_vq,
    local_mean: torch.Tensor,
    local_std: torch.Tensor,
    device: torch.device,
    sums: dict[str, float],
    fps: float,
) -> None:
    local_np = np.stack([item["local"] for item in batch]).astype(np.float32, copy=False)
    features_np = np.stack([item["features_263"] for item in batch]).astype(np.float32, copy=False)
    root_np = np.stack([
        _root_controls_from_features(item["features_263"], fps)
        for item in batch
    ]).astype(np.float32, copy=False)
    local_raw = torch.from_numpy(local_np).to(device)
    root_raw = torch.from_numpy(root_np).to(device)
    ref_features = torch.from_numpy(features_np).to(device)
    with torch.no_grad():
        local_pred_raw, local_pred_norm = _local_vq_reconstruct(local_vq, local_mean, local_std, local_raw)
        root_norm = (root_raw - root_mean) / root_std
        pred_root = root_model(root_norm, local_pred_norm) * root_std + root_mean

    pred_local_features_np = local_vector_to_features(features_np, local_pred_raw.detach().cpu().numpy())
    pred_features_np = _features_with_root_controls(pred_local_features_np, pred_root.detach().cpu().numpy(), fps)
    pred_features = torch.from_numpy(pred_features_np).to(device)
    with torch.no_grad():
        ref_joints = recover_from_ric(ref_features, 22)
        pred_joints = recover_from_ric(pred_features, 22)
        joint_err = torch.linalg.norm(pred_joints - ref_joints, dim=-1)
        ref_ra = ref_joints - ref_joints[..., :1, :]
        pred_ra = pred_joints - pred_joints[..., :1, :]
        ra_err = torch.linalg.norm(pred_ra - ref_ra, dim=-1)
        _, ref_root = motion_process.recover_root_rot_pos(ref_features)
        _, pred_root_pos = motion_process.recover_root_rot_pos(pred_features)
        xz_err = torch.linalg.norm(pred_root_pos[..., [0, 2]] - ref_root[..., [0, 2]], dim=-1)
        ref_steps = torch.linalg.norm(ref_root[..., 1:, [0, 2]] - ref_root[..., :-1, [0, 2]], dim=-1)
        pred_steps = torch.linalg.norm(pred_root_pos[..., 1:, [0, 2]] - pred_root_pos[..., :-1, [0, 2]], dim=-1)

    speed_bias = (pred_steps - ref_steps) * fps * 1000.0
    sums["mpjpe_sum"] += float(joint_err.sum().item())
    sums["mpjpe_count"] += int(joint_err.numel())
    sums["ra_sum"] += float(ra_err.sum().item())
    sums["ra_count"] += int(ra_err.numel())
    sums["root_xz_sum"] += float(xz_err.sum().item())
    sums["root_xz_count"] += int(xz_err.numel())
    sums["final_xz_sum"] += float(xz_err[:, -1].sum().item())
    sums["path_error_sum"] += float((pred_steps.sum(dim=-1) - ref_steps.sum(dim=-1)).sum().item())
    sums["speed_bias_sum"] += float(speed_bias.sum().item())
    sums["speed_bias_count"] += int(speed_bias.numel())
    sums["window_count"] += len(batch)
    sums["frame_count"] += int(ref_features.shape[0] * ref_features.shape[1])


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    root_model, root_mean, root_std, payload = _load_root_checkpoint(Path(args.checkpoint), device)
    local_vq, local_mean, local_std, _ = load_local_vq_checkpoint(Path(payload["args"]["local_vq_checkpoint"]), device)
    local_vq.eval()
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
        "final_xz_sum": 0.0,
        "path_error_sum": 0.0,
        "speed_bias_sum": 0.0,
        "speed_bias_count": 0,
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
        length = int(item["local"].shape[0])
        pending.setdefault(length, []).append(item)
        if len(examples) < 5:
            examples.append(f"{item['id']}:{item['start']}-{item['end']}")
        if len(pending[length]) >= args.batch_size:
            _flush_eval(pending[length], root_model, root_mean, root_std, local_vq, local_mean, local_std, device, sums, args.fps)
            pending[length] = []
    for batch in list(pending.values()):
        if batch:
            _flush_eval(batch, root_model, root_mean, root_std, local_vq, local_mean, local_std, device, sums, args.fps)

    result = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "local_vq_checkpoint": payload["args"]["local_vq_checkpoint"],
        "architecture": payload["args"].get("architecture", "unet"),
        "cache_root": str(Path(args.cache_root).expanduser().resolve()),
        "split": args.split,
        "window_frames": args.window_frames,
        "stride": args.stride,
        "include_tail": args.include_tail,
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
        "final_xz_error_mm": sums["final_xz_sum"] / max(sums["window_count"], 1) * 1000.0,
        "path_error_m": sums["path_error_sum"] / max(sums["window_count"], 1),
        "speed_bias_mm_per_s": sums["speed_bias_sum"] / max(sums["speed_bias_count"], 1),
    }
    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate a continuous root branch conditioned on local VQ.")
    sub = parser.add_subparsers(dest="command", required=True)
    train_p = sub.add_parser("train")
    train_p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    train_p.add_argument("--local-vq-checkpoint", default=DEFAULT_LOCAL_CKPT)
    train_p.add_argument("--exp-root", default=DEFAULT_EXP_ROOT)
    train_p.add_argument("--train-split", default="train")
    train_p.add_argument("--max-train-sequences", type=int, default=0)
    train_p.add_argument("--device", default="auto")
    train_p.add_argument("--seed", type=int, default=1234)
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--steps-per-epoch", type=int, default=100)
    train_p.add_argument("--batch-size", type=int, default=256)
    train_p.add_argument("--window-sizes", nargs="+", type=int, default=[64, 128, 196])
    train_p.add_argument("--window-weights", nargs="+", type=float, default=[0.25, 0.25, 0.5])
    train_p.add_argument("--fps", type=float, default=20.0)
    train_p.add_argument("--lr", type=float, default=2e-4)
    train_p.add_argument("--lr-min", type=float, default=1e-6)
    train_p.add_argument("--weight-decay", type=float, default=0.0)
    train_p.add_argument("--grad-clip", type=float, default=1.0)
    train_p.add_argument("--width", type=int, default=128)
    train_p.add_argument("--latent-width", type=int, default=128)
    train_p.add_argument("--architecture", choices=("unet", "bottleneck"), default="unet")
    train_p.add_argument("--lambda-control", type=float, default=1.0)
    train_p.add_argument("--lambda-yaw", type=float, default=0.1)
    train_p.add_argument("--lambda-velocity", type=float, default=0.5)
    train_p.add_argument("--lambda-height", type=float, default=0.5)
    train_p.add_argument("--lambda-global-pos", type=float, default=10.0)
    train_p.add_argument("--lambda-global-vel", type=float, default=10.0)
    train_p.add_argument("--lambda-final", type=float, default=20.0)
    train_p.add_argument("--lambda-path", type=float, default=10.0)
    train_p.add_argument("--lambda-smooth", type=float, default=0.1)
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
