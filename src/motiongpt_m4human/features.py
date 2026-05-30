from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


_HUMANML_PACKAGE = "_motiongpt_src_humanml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_humanml_package() -> Path:
    root = _repo_root() / "mGPT" / "data" / "humanml"
    packages = {
        _HUMANML_PACKAGE: root,
        f"{_HUMANML_PACKAGE}.common": root / "common",
        f"{_HUMANML_PACKAGE}.scripts": root / "scripts",
        f"{_HUMANML_PACKAGE}.utils": root / "utils",
    }
    for name, path in packages.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
    return root


def _load_humanml_module(relative_name: str):
    root = _ensure_humanml_package()
    module_name = f"{_HUMANML_PACKAGE}.{relative_name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = root / Path(*relative_name.split(".")).with_suffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MotionGPT HumanML module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


motion_process = _load_humanml_module("scripts.motion_process")
Skeleton = _load_humanml_module("common.skeleton").Skeleton
_param_util = _load_humanml_module("utils.paramUtil")
t2m_kinematic_chain = _param_util.t2m_kinematic_chain
t2m_raw_offsets = _param_util.t2m_raw_offsets
recover_from_ric = motion_process.recover_from_ric


@dataclass(frozen=True)
class FeatureConversionResult:
    features: np.ndarray
    canonical_joints: np.ndarray


def load_reference_pose(path: Path, num_joints: int = 22) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim == 3:
        pose = arr[0, :num_joints]
    elif arr.ndim == 2 and arr.shape[-1] == 3:
        pose = arr[:num_joints]
    elif arr.ndim == 2 and arr.shape[-1] == num_joints * 3:
        pose = arr[0].reshape(num_joints, 3)
    else:
        raise ValueError(f"Unsupported reference joints shape at {path}: {arr.shape}")
    return torch.from_numpy(np.asarray(pose, dtype=np.float32))


class HumanMLFeatureConverter:
    """Convert y-up 22-joint sequences into MotionGPT/HumanML3D 263-D features."""

    def __init__(
        self,
        reference_joints: Path | None = None,
        foot_threshold: float = 0.002,
        num_joints: int = 22,
    ) -> None:
        self.reference_joints = reference_joints
        self.foot_threshold = float(foot_threshold)
        self.num_joints = int(num_joints)
        self._configured = False

    def _configure(self, joints_hml: np.ndarray) -> None:
        motion_process.l_idx1, motion_process.l_idx2 = 5, 8
        motion_process.fid_r, motion_process.fid_l = [8, 11], [7, 10]
        motion_process.face_joint_indx = [2, 1, 17, 16]
        motion_process.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
        motion_process.kinematic_chain = t2m_kinematic_chain

        if self.reference_joints is not None:
            reference = load_reference_pose(self.reference_joints, self.num_joints)
        else:
            reference = torch.from_numpy(joints_hml[0].astype(np.float32))

        target_skeleton = Skeleton(motion_process.n_raw_offsets, motion_process.kinematic_chain, "cpu")
        motion_process.tgt_offsets = target_skeleton.get_offsets_joints(reference)
        self._configured = True

    def convert(self, joints_hml: np.ndarray) -> FeatureConversionResult:
        if joints_hml.ndim != 3 or joints_hml.shape[1:] != (self.num_joints, 3):
            raise ValueError(f"Expected joints with shape [T,{self.num_joints},3], got {joints_hml.shape}")
        if joints_hml.shape[0] < 2:
            raise ValueError("At least two frames are required to build MotionGPT features")
        if not self._configured:
            self._configure(joints_hml)

        features, canonical_joints, _, _ = motion_process.process_file(
            joints_hml.astype(np.float32, copy=True),
            self.foot_threshold,
        )
        return FeatureConversionResult(
            features=features.astype(np.float32, copy=False),
            canonical_joints=canonical_joints[:-1].astype(np.float32, copy=False),
        )
