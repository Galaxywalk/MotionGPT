from __future__ import annotations

import gzip
import importlib.util
import os
import pickle
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lmdb
import msgpack
import numpy as np
import torch
from torch.utils.data import Dataset


_HUMANML_PACKAGE = "_motiongpt_humanml"


def _ensure_humanml_package() -> Path:
    root = Path(__file__).resolve().parents[2] / "mGPT" / "data" / "humanml"
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


VALID_PROTOCOLS = ("p1", "p2", "p3")
VALID_SPLIT_IDS = ("s1", "s2", "s3")
VALID_SUBSETS = ("train", "val", "test")
GENDER_INFO = {
    "P1": 0,
    "P3": 0,
    "P4": 0,
    "P7": 0,
    "P8": 0,
    "P10": 0,
    "P13": 0,
    "P15": 0,
    "P2": 1,
    "P5": 1,
    "P6": 1,
    "P9": 1,
    "P11": 1,
    "P12": 1,
    "P14": 1,
    "P16": 1,
    "P17": 1,
    "P18": 1,
    "P19": 1,
    "P20": 1,
}
NON_VALID_INDICATOR_SET = {
    (1, 12, 58),
    (1, 13, 491),
    (1, 31, 324),
    (1, 36, 284),
    (1, 37, 43),
    (5, 43, 551),
    (5, 43, 552),
    (5, 47, 136),
    (5, 47, 140),
}


@dataclass(frozen=True)
class M4HumanMotionFeatureConfig:
    root: Path
    protocol: str = "p1"
    split_id: str = "s2"
    subset: str = "test"
    feature_frames: int = 196
    stride: int = 196
    max_windows: int | None = 16
    axis_mode: str = "xz-y"
    num_joints: int = 22
    pose_source: str = "param_joints"
    smplx_model_root: Path | None = None
    normalize_z: bool = False
    reference_joints: Path | None = None
    foot_threshold: float = 0.002


def _resolve_cache_path(root: Path) -> Path:
    root = root.expanduser().resolve()
    if (root / "rf3dpose_all").exists():
        return root / "rf3dpose_all"
    return root


def _resolve_smplx_model_root(root: Path, cache_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    root = root.expanduser().resolve()
    if (root / "models").exists():
        return root / "models"
    if cache_path.name == "rf3dpose_all":
        return cache_path.parent / "models"
    return root / "models"


def _decode_np(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get("__nd__") is True:
        arr = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
        return arr.reshape(tuple(obj["shape"]))
    if isinstance(obj, dict):
        return {k: _decode_np(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_np(v) for v in obj]
    return obj


def _unpack_dict_np(blob: bytes) -> dict[str, Any]:
    return _decode_np(msgpack.unpackb(blob, raw=False))


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    rvec = np.asarray(rvec, dtype=np.float64)
    theta = np.linalg.norm(rvec)
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    r = rvec / theta
    k = np.array(
        [
            [0.0, -r[2], r[1]],
            [r[2], 0.0, -r[0]],
            [-r[1], r[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def _inv_rodrigues(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64)
    theta = np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * np.sin(theta))
    return (axis * theta).astype(np.float32)


def _calibrate_smplx_root_to_radar(params: dict[str, Any], calib: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    root_rot = _rodrigues(np.asarray(params["root_orient"], dtype=np.float32))
    vicon_to_cam_rot = np.asarray(calib["vicon_to_cam_rotmatrix"], dtype=np.float64)
    vicon_to_cam_t = np.asarray(calib["vicon_to_cam_tvec"], dtype=np.float64) / 1000.0
    radar_to_cam_rot = np.asarray(calib["radar_to_cam_rotmatrix"], dtype=np.float64)
    radar_to_cam_t = np.asarray(calib["radar_to_cam_tvec"], dtype=np.float64)

    cam_rot = vicon_to_cam_rot @ root_rot
    radar_rot = np.linalg.inv(radar_to_cam_rot) @ cam_rot

    joints = np.asarray(params["joints"], dtype=np.float64)
    trans = np.asarray(params["trans"], dtype=np.float64)
    cam_trans = vicon_to_cam_rot @ joints[0] + vicon_to_cam_t
    radar_trans = np.linalg.inv(radar_to_cam_rot) @ (cam_trans - radar_to_cam_t)
    transl = radar_trans + (trans - joints[0])
    return _inv_rodrigues(radar_rot), transl.astype(np.float32)


def _axis_to_humanml(joints_radar: np.ndarray, mode: str) -> np.ndarray:
    """Map M4Human radar coordinates to MotionGPT's y-up convention."""
    x = joints_radar[..., 0]
    y = joints_radar[..., 1]
    z = joints_radar[..., 2]
    if mode == "xzy":
        out = np.stack([x, z, y], axis=-1)
    elif mode == "xz-y":
        out = np.stack([x, z, -y], axis=-1)
    elif mode == "-xzy":
        out = np.stack([-x, z, y], axis=-1)
    elif mode == "x-zy":
        out = np.stack([x, -z, y], axis=-1)
    else:
        raise ValueError(f"Unknown axis_mode={mode!r}; use xzy, xz-y, -xzy, or x-zy")
    return out.astype(np.float32, copy=False)


def _load_reference_pose(path: Path, num_joints: int) -> torch.Tensor:
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


class M4HumanMotionFeatureDataset(Dataset):
    """Build MotionGPT HumanML3D-style feature windows from M4Human LMDB joints.

    The dataset returns absolute M4Human pose windows converted to MotionGPT's
    263-D feature format. Each item contains `feature_frames` feature rows and
    therefore reads `feature_frames + 1` original M4Human frames.
    """

    def __init__(self, config: M4HumanMotionFeatureConfig):
        super().__init__()
        self.config = config
        if config.protocol not in VALID_PROTOCOLS:
            raise ValueError(f"protocol must be one of {VALID_PROTOCOLS}, got {config.protocol}")
        if config.split_id not in VALID_SPLIT_IDS:
            raise ValueError(f"split_id must be one of {VALID_SPLIT_IDS}, got {config.split_id}")
        if config.subset not in VALID_SUBSETS:
            raise ValueError(f"subset must be one of {VALID_SUBSETS}, got {config.subset}")
        if config.feature_frames <= 0 or config.feature_frames % 4 != 0:
            raise ValueError("feature_frames must be positive and divisible by 4")
        if config.stride <= 0:
            raise ValueError("stride must be positive")
        if config.pose_source not in {"param_joints", "smplx"}:
            raise ValueError("pose_source must be one of {'param_joints', 'smplx'}")

        self.cache_path = _resolve_cache_path(config.root)
        self.smplx_model_root = _resolve_smplx_model_root(config.root, self.cache_path, config.smplx_model_root)
        self.env_paths = {
            "params": self.cache_path / "params.lmdb",
            "calib": self.cache_path / "calib.lmdb",
        }
        missing = [str(path) for path in self.env_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing M4Human LMDB files: {missing}")

        self.lmdb_envs: dict[str, lmdb.Environment] | None = None
        self._lmdb_owner_pid: int | None = None
        self.smplx_models: dict[str, torch.nn.Module] | None = None
        self._smplx_owner_pid: int | None = None
        self._motion_process_ready = False
        self.windows = self._build_windows()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["lmdb_envs"] = None
        state["_lmdb_owner_pid"] = None
        state["smplx_models"] = None
        state["_smplx_owner_pid"] = None
        state["_motion_process_ready"] = False
        return state

    def close(self) -> None:
        if self.lmdb_envs is not None:
            for env in self.lmdb_envs.values():
                env.close()
        self.lmdb_envs = None
        self._lmdb_owner_pid = None
        self.smplx_models = None
        self._smplx_owner_pid = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _open_lmdb_envs(self) -> None:
        cur_pid = os.getpid()
        if self.lmdb_envs is not None and self._lmdb_owner_pid == cur_pid:
            return
        if self.lmdb_envs is not None:
            for env in self.lmdb_envs.values():
                env.close()
        self.lmdb_envs = {
            name: lmdb.open(str(path), readonly=True, subdir=False, lock=False, readahead=False)
            for name, path in self.env_paths.items()
        }
        self._lmdb_owner_pid = cur_pid

    def _ensure_smplx_models(self) -> None:
        cur_pid = os.getpid()
        if self.smplx_models is not None and self._smplx_owner_pid == cur_pid:
            return
        if not self.smplx_model_root.exists():
            raise FileNotFoundError(f"Missing SMPL-X model root: {self.smplx_model_root}")

        import smplx

        models: dict[str, torch.nn.Module] = {}
        for gender in ("male", "female"):
            model = smplx.create(
                model_path=str(self.smplx_model_root),
                model_type="smplx",
                gender=gender,
            )
            model.eval()
            model.requires_grad_(False)
            models[gender] = model
        self.smplx_models = models
        self._smplx_owner_pid = cur_pid

    def _load_split_indices(self) -> list[tuple[int, int, int]]:
        indices_path = self.cache_path / "indeces.pkl.gz"
        if not indices_path.exists():
            raise FileNotFoundError(f"Missing split index file: {indices_path}")
        with gzip.open(indices_path, "rb") as f:
            split_indices = pickle.load(f)
        indicators = split_indices[self.config.protocol][self.config.split_id][self.config.subset]
        return [
            tuple(int(v) for v in indicator)
            for indicator in indicators
            if tuple(indicator) not in NON_VALID_INDICATOR_SET
        ]

    def _build_windows(self) -> list[list[tuple[int, int, int]]]:
        indicators = self._load_split_indices()
        grouped: dict[tuple[int, int], list[tuple[int, tuple[int, int, int]]]] = defaultdict(list)
        for indicator in indicators:
            subject, action, frame = indicator
            grouped[(subject, action)].append((frame, indicator))

        joint_frames = self.config.feature_frames + 1
        windows: list[list[tuple[int, int, int]]] = []

        def emit_segment(segment: list[tuple[int, int, int]]) -> bool:
            if len(segment) < joint_frames:
                return False
            limit = len(segment) - joint_frames + 1
            for start in range(0, limit, self.config.stride):
                windows.append(segment[start : start + joint_frames])
                if self.config.max_windows is not None and len(windows) >= self.config.max_windows:
                    return True
            return False

        for key in sorted(grouped):
            frames = sorted(grouped[key], key=lambda x: x[0])
            segment: list[tuple[int, int, int]] = []
            prev_frame: int | None = None
            for frame, indicator in frames:
                if prev_frame is not None and frame != prev_frame + 1:
                    if emit_segment(segment):
                        return windows
                    segment = []
                segment.append(indicator)
                prev_frame = frame
            if emit_segment(segment):
                return windows
        return windows

    def _read_params_and_calib(self, indicator: tuple[int, int, int]) -> tuple[dict[str, Any], dict[str, Any]]:
        self._open_lmdb_envs()
        assert self.lmdb_envs is not None
        key = str(list(indicator)).encode()
        with self.lmdb_envs["params"].begin() as txn_param, self.lmdb_envs["calib"].begin() as txn_calib:
            param_blob = txn_param.get(key)
            calib_blob = txn_calib.get(key)
        if param_blob is None or calib_blob is None:
            raise KeyError(f"Missing params/calib for indicator {indicator}")
        return _unpack_dict_np(param_blob), _unpack_dict_np(calib_blob)

    def _load_param_joints_radar(self, indicator: tuple[int, int, int]) -> np.ndarray:
        params, calib = self._read_params_and_calib(indicator)
        joints = np.asarray(params["joints"], dtype=np.float32)
        if joints.ndim != 2 or joints.shape[0] < self.config.num_joints or joints.shape[1] < 3:
            raise ValueError(f"Invalid params.joints for indicator {indicator}: {joints.shape}")

        vicon_to_cam_rot = np.asarray(calib["vicon_to_cam_rotmatrix"], dtype=np.float32)
        vicon_to_cam_t = np.asarray(calib["vicon_to_cam_tvec"], dtype=np.float32) / 1000.0
        radar_to_cam_rot = np.asarray(calib["radar_to_cam_rotmatrix"], dtype=np.float32)
        radar_to_cam_t = np.asarray(calib["radar_to_cam_tvec"], dtype=np.float32)
        joints_cam = (vicon_to_cam_rot @ joints[:, :3].T).T + vicon_to_cam_t
        joints_radar = (np.linalg.inv(radar_to_cam_rot) @ (joints_cam - radar_to_cam_t).T).T
        if self.config.normalize_z:
            joints_radar = joints_radar.copy()
            joints_radar[:, 2] -= 1.5
        return joints_radar[: self.config.num_joints].astype(np.float32, copy=False)

    def _load_smplx_window_radar(self, indicators: list[tuple[int, int, int]]) -> np.ndarray:
        self._ensure_smplx_models()
        assert self.smplx_models is not None

        batches: dict[str, list[tuple[int, dict[str, Any], np.ndarray, np.ndarray]]] = {
            "male": [],
            "female": [],
        }
        for frame_idx, indicator in enumerate(indicators):
            params, calib = self._read_params_and_calib(indicator)
            subject_key = f"P{indicator[0]}"
            if subject_key not in GENDER_INFO:
                raise KeyError(f"Unknown M4Human subject gender for {subject_key}")
            gender = "male" if float(GENDER_INFO[subject_key]) > 0.5 else "female"
            root_orient, transl = _calibrate_smplx_root_to_radar(params, calib)
            if self.config.normalize_z:
                transl = transl.copy()
                transl[2] -= 1.5
            batches[gender].append((frame_idx, params, root_orient, transl))

        joints_radar = np.zeros((len(indicators), self.config.num_joints, 3), dtype=np.float32)
        for gender, entries in batches.items():
            if not entries:
                continue
            model = self.smplx_models[gender]
            betas = torch.as_tensor(np.stack([e[1]["betas"] for e in entries]), dtype=torch.float32)
            body_pose = torch.as_tensor(np.stack([e[1]["pose_body"] for e in entries]), dtype=torch.float32)
            global_orient = torch.as_tensor(np.stack([e[2] for e in entries]), dtype=torch.float32)
            transl = torch.as_tensor(np.stack([e[3] for e in entries]), dtype=torch.float32)

            with torch.no_grad():
                batch_size = int(betas.shape[0])
                hand_pose = torch.zeros((batch_size, int(model.num_pca_comps)), dtype=torch.float32)
                face_pose = torch.zeros((batch_size, 3), dtype=torch.float32)
                expression = torch.zeros((batch_size, int(model.num_expression_coeffs)), dtype=torch.float32)
                output = model(
                    betas=betas,
                    body_pose=body_pose,
                    global_orient=global_orient,
                    transl=transl,
                    left_hand_pose=hand_pose,
                    right_hand_pose=hand_pose,
                    jaw_pose=face_pose,
                    leye_pose=face_pose,
                    reye_pose=face_pose,
                    expression=expression,
                )
                vertices_mm = output.vertices.detach() * 1000.0
                regressor = model.J_regressor[: self.config.num_joints].to(vertices_mm.device)
                joints_m = torch.einsum("ij,bjc->bic", regressor, vertices_mm) / 1000.0
            group_joints = joints_m.cpu().numpy().astype(np.float32, copy=False)
            for local_idx, (frame_idx, _, _, _) in enumerate(entries):
                joints_radar[frame_idx] = group_joints[local_idx]
        return joints_radar

    def _load_window_joints_radar(self, indicators: list[tuple[int, int, int]]) -> np.ndarray:
        if self.config.pose_source == "param_joints":
            return np.stack([self._load_param_joints_radar(indicator) for indicator in indicators], axis=0)
        return self._load_smplx_window_radar(indicators)

    def _configure_motion_process(self, joints_hml: np.ndarray) -> None:
        motion_process.l_idx1, motion_process.l_idx2 = 5, 8
        motion_process.fid_r, motion_process.fid_l = [8, 11], [7, 10]
        motion_process.face_joint_indx = [2, 1, 17, 16]
        motion_process.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
        motion_process.kinematic_chain = t2m_kinematic_chain

        if self.config.reference_joints is not None:
            reference = _load_reference_pose(self.config.reference_joints, self.config.num_joints)
        else:
            reference = torch.from_numpy(joints_hml[0].astype(np.float32))
        target_skeleton = Skeleton(motion_process.n_raw_offsets, motion_process.kinematic_chain, "cpu")
        motion_process.tgt_offsets = target_skeleton.get_offsets_joints(reference)
        self._motion_process_ready = True

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        indicators = self.windows[int(idx)]
        joints_radar = self._load_window_joints_radar(indicators)
        joints_hml = _axis_to_humanml(joints_radar, self.config.axis_mode)
        if not self._motion_process_ready:
            self._configure_motion_process(joints_hml)

        features, canonical_joints, _, _ = motion_process.process_file(
            joints_hml.copy(),
            self.config.foot_threshold,
        )
        features = features.astype(np.float32, copy=False)
        canonical_joints = canonical_joints[:-1].astype(np.float32, copy=False)
        if features.shape != (self.config.feature_frames, 263):
            raise ValueError(f"Unexpected feature shape {features.shape}; expected {(self.config.feature_frames, 263)}")

        start = indicators[0]
        end = indicators[-1]
        return {
            "name": f"P{start[0]}_A{start[1]}_{start[2]}-{end[2]}",
            "indicators": np.asarray(indicators, dtype=np.int64),
            "joints_radar": joints_radar,
            "features": features,
            "canonical_joints": canonical_joints,
        }
