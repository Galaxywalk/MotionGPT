from __future__ import annotations

import importlib.util
import sys
import types
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.core.multiarray import _reconstruct as numpy_reconstruct
import torch

from mGPT.archs.mgpt_vq import VQVae


DEFAULT_CKPT = (
    "experiments/mgpt/VQVAE_HumanML3D_H200_2000e_bs256_eval100/"
    "checkpoints/min-MPJPEep=0.ckpt"
)
DEFAULT_MEAN = "deps/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy"
DEFAULT_STD = "deps/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy"


class RootVelocityCalibratedVQVae(torch.nn.Module):
    def __init__(
        self,
        vae: VQVae,
        scale_logit: torch.Tensor,
        bounds: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.register_buffer("root_velocity_scale_logit", scale_logit.detach().clone().float())
        self.register_buffer("root_velocity_scale_bounds", bounds.detach().clone().float())
        if bias is not None:
            self.register_buffer("root_velocity_bias", bias.detach().clone().float())

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        bounds = self.root_velocity_scale_bounds.to(device=x.device, dtype=x.dtype)
        logits = self.root_velocity_scale_logit.to(device=x.device, dtype=x.dtype)
        return bounds[0] + (bounds[1] - bounds[0]) * torch.sigmoid(logits)

    def _apply_calibration(self, feats: torch.Tensor) -> torch.Tensor:
        scale = self._scale(feats)
        bias = getattr(self, "root_velocity_bias", None)
        if bias is None:
            bias = torch.zeros_like(scale)
        else:
            bias = bias.to(device=feats.device, dtype=feats.dtype)
        feats = feats.clone()
        feats[..., 1:3] = feats[..., 1:3] * scale + bias
        return feats

    def forward(self, *args, **kwargs):
        feats, loss_commit, perplexity = self.vae(*args, **kwargs)
        return self._apply_calibration(feats), loss_commit, perplexity

    def encode(self, *args, **kwargs):
        return self.vae.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._apply_calibration(self.vae.decode(*args, **kwargs))


def numpy_core_reconstruct(*args, **kwargs):
    return numpy_reconstruct(*args, **kwargs)


numpy_core_reconstruct.__name__ = "_reconstruct"
numpy_core_reconstruct.__qualname__ = "_reconstruct"
numpy_core_reconstruct.__module__ = "numpy.core.multiarray"


class CheckpointState:
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
    (CheckpointState,),
    {"__module__": "mGPT.data.humanml.utils.word_vectorizer"},
)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_arg}, but CUDA is not available")
    return device


def install_omegaconf_checkpoint_shim() -> list[Any]:
    common_symbols: list[Any] = [
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
        numpy_core_reconstruct,
        WordVectorizer,
    ]

    if importlib.util.find_spec("omegaconf") is not None:
        from omegaconf.dictconfig import DictConfig
        from omegaconf.listconfig import ListConfig

        symbols = [*common_symbols, DictConfig, ListConfig]
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

    class OmegaConfState:
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
        (OmegaConfState, "omegaconf.base"),
    ):
        cls.__module__ = module_name

    listconfig.ListConfig = ListConfig
    dictconfig.DictConfig = DictConfig
    symbols = [*common_symbols, ListConfig, DictConfig]
    for name in ("ContainerMetadata", "Metadata", "ValueKind"):
        child = type(name, (OmegaConfState,), {"__module__": "omegaconf.base"})
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
        child = type(name, (OmegaConfState,), {"__module__": "omegaconf.nodes"})
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


def load_vqvae(
    checkpoint_path: Path,
    device: torch.device,
    calibration_domain: str = "none",
) -> tuple[torch.nn.Module, dict[str, Any]]:
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

    safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if safe_globals is not None:
        safe_globals(install_omegaconf_checkpoint_shim())
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
    calibration_meta: dict[str, Any] = {
        "root_velocity_calibration_domain": calibration_domain,
        "root_velocity_calibration_applied": False,
    }
    scale_logit = state.get("root_velocity_scale_logit")
    if scale_logit is not None:
        bounds = state.get(
            "root_velocity_scale_bounds",
            torch.tensor([0.8, 1.3], dtype=torch.float32),
        )
        scale = bounds[0] + (bounds[1] - bounds[0]) * torch.sigmoid(scale_logit)
        calibration_meta.update({
            "root_velocity_scale": [float(v) for v in scale.detach().cpu().tolist()],
            "root_velocity_scale_bounds": [float(v) for v in bounds.detach().cpu().tolist()],
        })
        if calibration_domain in ("m4human", "all", "*"):
            model = RootVelocityCalibratedVQVae(
                model,
                scale_logit=scale_logit,
                bounds=bounds,
                bias=state.get("root_velocity_bias"),
            ).to(device)
            model.eval()
            calibration_meta["root_velocity_calibration_applied"] = True
    return model, {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        **calibration_meta,
    }
