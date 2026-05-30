from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.core.multiarray import _reconstruct as numpy_reconstruct
import torch
from torch.utils.data import DataLoader

from mGPT.archs.mgpt_vq import VQVae

from tools.m4human_motiongpt.dataloader import (
    M4HumanMotionFeatureConfig,
    M4HumanMotionFeatureDataset,
    recover_from_ric,
)


def _numpy_core_reconstruct(*args, **kwargs):
    return numpy_reconstruct(*args, **kwargs)


_numpy_core_reconstruct.__name__ = "_reconstruct"
_numpy_core_reconstruct.__qualname__ = "_reconstruct"
_numpy_core_reconstruct.__module__ = "numpy.core.multiarray"


class _CheckpointState:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.__dict__.update(kwargs)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


WordVectorizer = type(
    "WordVectorizer",
    (_CheckpointState,),
    {"__module__": "mGPT.data.humanml.utils.word_vectorizer"},
)


DEFAULT_CKPT = (
    "experiments/mgpt/VQVAE_HumanML3D_H200_2000e_bs256_eval100/"
    "checkpoints/min-MPJPEep=0.ckpt"
)
DEFAULT_MEAN = "deps/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy"
DEFAULT_STD = "deps/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy"


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_arg}, but CUDA is not available")
    return device


def _install_omegaconf_checkpoint_shim() -> list[Any]:
    """Provide enough OmegaConf symbols to unpickle trusted Lightning checkpoints."""
    if importlib.util.find_spec("omegaconf") is not None:
        from omegaconf.dictconfig import DictConfig
        from omegaconf.listconfig import ListConfig

        symbols: list[Any] = [
            Any,
            bool,
            int,
            float,
            str,
            type(None),
            list,
            dict,
            tuple,
            set,
            defaultdict,
            OrderedDict,
            np.dtype,
            type(np.dtype("bool")),
            type(np.dtype("float32")),
            type(np.dtype("float64")),
            type(np.dtype("int32")),
            type(np.dtype("int64")),
            np.ndarray,
            _numpy_core_reconstruct,
            WordVectorizer,
            DictConfig,
            ListConfig,
        ]
        for module_name in ("omegaconf.base", "omegaconf.nodes"):
            module = __import__(module_name, fromlist=["_"])
            for name in (
                "AnyNode",
                "BooleanNode",
                "BytesNode",
                "ContainerMetadata",
                "EnumNode",
                "FloatNode",
                "IntegerNode",
                "Metadata",
                "StringNode",
                "ValueKind",
                "ValueNode",
            ):
                value = getattr(module, name, None)
                if isinstance(value, type):
                    symbols.append(value)
        return symbols

    omega = types.ModuleType("omegaconf")
    listconfig = types.ModuleType("omegaconf.listconfig")
    dictconfig = types.ModuleType("omegaconf.dictconfig")
    base = types.ModuleType("omegaconf.base")
    nodes = types.ModuleType("omegaconf.nodes")

    class ListConfig(list):
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

    class DictConfig(dict):
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

    class _OmegaConfState:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.__dict__.update(kwargs)

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.state = state

    for cls, module_name in (
        (ListConfig, "omegaconf.listconfig"),
        (DictConfig, "omegaconf.dictconfig"),
        (_OmegaConfState, "omegaconf.base"),
    ):
        cls.__module__ = module_name

    listconfig.ListConfig = ListConfig
    dictconfig.DictConfig = DictConfig
    symbols: list[Any] = [
        Any,
        bool,
        int,
        float,
        str,
        type(None),
        list,
        dict,
        tuple,
        set,
        defaultdict,
        OrderedDict,
        np.dtype,
        type(np.dtype("bool")),
        type(np.dtype("float32")),
        type(np.dtype("float64")),
        type(np.dtype("int32")),
        type(np.dtype("int64")),
        np.ndarray,
        _numpy_core_reconstruct,
        WordVectorizer,
        ListConfig,
        DictConfig,
    ]
    for name in ("ContainerMetadata", "Metadata", "ValueKind"):
        child = type(name, (_OmegaConfState,), {"__module__": "omegaconf.base"})
        setattr(base, name, child)
        symbols.append(child)
    for name in (
        "AnyNode",
        "BooleanNode",
        "BytesNode",
        "EnumNode",
        "FloatNode",
        "IntegerNode",
        "StringNode",
        "ValueNode",
    ):
        child = type(name, (_OmegaConfState,), {"__module__": "omegaconf.nodes"})
        setattr(nodes, name, child)
        symbols.append(child)

    omega.ListConfig = ListConfig
    omega.DictConfig = DictConfig
    omega.listconfig = listconfig
    omega.dictconfig = dictconfig
    omega.base = base
    omega.nodes = nodes
    sys.modules.setdefault("omegaconf", omega)
    sys.modules.setdefault("omegaconf.listconfig", listconfig)
    sys.modules.setdefault("omegaconf.dictconfig", dictconfig)
    sys.modules.setdefault("omegaconf.base", base)
    sys.modules.setdefault("omegaconf.nodes", nodes)
    return symbols


def _load_vae(checkpoint_path: Path, device: torch.device) -> tuple[VQVae, dict[str, Any]]:
    model = VQVae(
        nfeats=263,
        quantizer="ema_reset",
        code_num=512,
        code_dim=512,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        norm=None,
        activation="relu",
    )

    safe_symbols = _install_omegaconf_checkpoint_shim()
    torch.serialization.add_safe_globals(safe_symbols)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    prefixes = ("vae.", "motion_vae.", "model.vae.")
    selected = None
    for prefix in prefixes:
        candidate = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        if candidate:
            selected = candidate
            break
    if selected is None:
        selected = state

    missing, unexpected = model.load_state_dict(selected, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match VQVae architecture: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    model.to(device)
    model.eval()
    meta = {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
    }
    return model, meta


def _mpjpe_sums(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, int]:
    err = torch.linalg.norm(pred - target, dim=-1)
    return float(err.sum().item()), int(err.numel())


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = _resolve_device(args.device)
    dataset = M4HumanMotionFeatureDataset(
        M4HumanMotionFeatureConfig(
            root=Path(args.m4human_root),
            protocol=args.protocol,
            split_id=args.split_id,
            subset=args.subset,
            feature_frames=args.feature_frames,
            stride=args.stride,
            max_windows=None if args.max_windows <= 0 else args.max_windows,
            axis_mode=args.axis_mode,
            pose_source=args.pose_source,
            smplx_model_root=Path(args.smplx_model_root) if args.smplx_model_root else None,
            normalize_z=args.normalize_z,
            reference_joints=Path(args.reference_joints) if args.reference_joints else None,
            foot_threshold=args.foot_threshold,
        )
    )
    if len(dataset) == 0:
        raise RuntimeError("No M4Human motion windows were built; lower feature_frames or check the split")

    mean = torch.from_numpy(np.load(args.mean).astype(np.float32)).to(device)
    std = torch.from_numpy(np.load(args.std).astype(np.float32)).to(device)
    if mean.shape[-1] != 263 or std.shape[-1] != 263:
        raise ValueError(f"Expected mean/std with 263 features, got {tuple(mean.shape)} and {tuple(std.shape)}")

    model, ckpt_meta = _load_vae(Path(args.checkpoint), device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    recon_sum = 0.0
    recon_count = 0
    root_aligned_sum = 0.0
    root_aligned_count = 0
    converter_sum = 0.0
    converter_count = 0
    feature_l1_sum = 0.0
    feature_l1_count = 0
    token_count = 0
    unique_tokens: set[int] = set()
    first_names: list[str] = []

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            features = batch["features"].to(device=device, dtype=torch.float32)
            canonical = batch["canonical_joints"].to(device=device, dtype=torch.float32)
            norm_features = (features - mean) / std
            recon_norm, _, _ = model(norm_features)
            recon_features = recon_norm * std + mean

            ref_joints = recover_from_ric(features, 22)
            recon_joints = recover_from_ric(recon_features, 22)
            s, c = _mpjpe_sums(recon_joints, ref_joints)
            recon_sum += s
            recon_count += c

            ref_ra = ref_joints - ref_joints[..., :1, :]
            recon_ra = recon_joints - recon_joints[..., :1, :]
            s, c = _mpjpe_sums(recon_ra, ref_ra)
            root_aligned_sum += s
            root_aligned_count += c

            s, c = _mpjpe_sums(ref_joints, canonical)
            converter_sum += s
            converter_count += c

            feature_l1_sum += float(torch.abs(recon_features - features).sum().item())
            feature_l1_count += int(recon_features.numel())

            codes, _ = model.encode(norm_features)
            token_count += int(codes.numel())
            unique_tokens.update(int(v) for v in codes.detach().cpu().reshape(-1).tolist())
            if len(first_names) < 5:
                first_names.extend(batch["name"][: 5 - len(first_names)])

            if args.log_every and step % args.log_every == 0:
                done = min(step * args.batch_size, len(dataset))
                print(f"processed {done}/{len(dataset)} windows")

    result = {
        **ckpt_meta,
        "m4human_root": str(Path(args.m4human_root).resolve()),
        "protocol": args.protocol,
        "split_id": args.split_id,
        "subset": args.subset,
        "axis_mode": args.axis_mode,
        "pose_source": args.pose_source,
        "smplx_model_root": str(Path(args.smplx_model_root).resolve()) if args.smplx_model_root else "",
        "feature_frames": args.feature_frames,
        "windows": len(dataset),
        "batch_size": args.batch_size,
        "device": str(device),
        "vqvae_mpjpe_mm": recon_sum / recon_count * 1000.0,
        "vqvae_root_aligned_mpjpe_mm": root_aligned_sum / root_aligned_count * 1000.0,
        "feature_converter_mpjpe_mm": converter_sum / converter_count * 1000.0,
        "feature_l1": feature_l1_sum / feature_l1_count,
        "tokens_per_window": token_count / len(dataset),
        "unique_code_count": len(unique_tokens),
        "example_windows": first_names,
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MotionGPT VQVAE reconstruction on M4Human LMDB motions.")
    parser.add_argument("--m4human-root", default="/cpfs01/liangbo/widouble_workspace")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--mean", default=DEFAULT_MEAN)
    parser.add_argument("--std", default=DEFAULT_STD)
    parser.add_argument("--protocol", default="p1", choices=("p1", "p2", "p3"))
    parser.add_argument("--split-id", default="s2", choices=("s1", "s2", "s3"))
    parser.add_argument("--subset", default="test", choices=("train", "val", "test"))
    parser.add_argument("--feature-frames", type=int, default=196)
    parser.add_argument("--stride", type=int, default=196)
    parser.add_argument("--max-windows", type=int, default=16, help="Use 0 or a negative value to evaluate all windows.")
    parser.add_argument("--axis-mode", default="xz-y", choices=("xzy", "xz-y", "-xzy", "x-zy"))
    parser.add_argument("--pose-source", default="param_joints", choices=("param_joints", "smplx"))
    parser.add_argument("--smplx-model-root", default="/cpfs01/liangbo/widouble_workspace/models")
    parser.add_argument("--normalize-z", action="store_true")
    parser.add_argument("--reference-joints", default="")
    parser.add_argument("--foot-threshold", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--out-json", default="")
    return parser


def main() -> None:
    result = evaluate(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
