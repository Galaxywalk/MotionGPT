from __future__ import annotations

import gzip
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lmdb
import msgpack
import numpy as np
import torch


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
class MotionSegment:
    protocol: str
    split_id: str
    subset: str
    subject: int
    action: int
    start_frame: int
    end_frame: int
    indicators: tuple[tuple[int, int, int], ...]

    @property
    def raw_frames(self) -> int:
        return len(self.indicators)

    @property
    def sequence_id(self) -> str:
        return f"{self.subset}_P{self.subject}_A{self.action}_{self.start_frame:06d}_{self.end_frame:06d}"


def resolve_cache_path(root: Path) -> Path:
    root = root.expanduser().resolve()
    if (root / "rf3dpose_all").exists():
        return root / "rf3dpose_all"
    return root


def resolve_smplx_model_root(root: Path, cache_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    root = root.expanduser().resolve()
    if (root / "models").exists():
        return root / "models"
    if cache_path.name == "rf3dpose_all":
        return cache_path.parent / "models"
    return root / "models"


def decode_np(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get("__nd__") is True:
        arr = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
        return arr.reshape(tuple(obj["shape"]))
    if isinstance(obj, dict):
        return {k: decode_np(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_np(v) for v in obj]
    return obj


def unpack_dict_np(blob: bytes) -> dict[str, Any]:
    return decode_np(msgpack.unpackb(blob, raw=False))


def axis_to_humanml(joints_radar: np.ndarray, mode: str) -> np.ndarray:
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


def resample_joints_linear(joints: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(f"source_fps and target_fps must be positive, got {source_fps}, {target_fps}")
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected joints with shape [T,J,3], got {joints.shape}")
    if joints.shape[0] < 2 or abs(source_fps - target_fps) < 1e-8:
        return joints.astype(np.float32, copy=True)

    source_times = np.arange(joints.shape[0], dtype=np.float64) / float(source_fps)
    last_time = source_times[-1]
    target_len = int(np.floor(last_time * float(target_fps) + 1e-6)) + 1
    target_times = np.arange(target_len, dtype=np.float64) / float(target_fps)
    target_times[-1] = min(target_times[-1], last_time)

    flat = joints.reshape(joints.shape[0], -1)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float32)
    for dim in range(flat.shape[1]):
        out[:, dim] = np.interp(target_times, source_times, flat[:, dim]).astype(np.float32)
    return out.reshape(target_len, joints.shape[1], 3)


def rodrigues(rvec: np.ndarray) -> np.ndarray:
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


def inv_rodrigues(rot: np.ndarray) -> np.ndarray:
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


def calibrate_smplx_root_to_radar(params: dict[str, Any], calib: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    root_rot = rodrigues(np.asarray(params["root_orient"], dtype=np.float32))
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
    return inv_rodrigues(radar_rot), transl.astype(np.float32)


class M4HumanReader:
    def __init__(
        self,
        root: Path,
        pose_source: str = "param_joints",
        smplx_model_root: Path | None = None,
        num_joints: int = 22,
        normalize_z: bool = False,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.cache_path = resolve_cache_path(self.root)
        self.pose_source = pose_source
        self.num_joints = int(num_joints)
        self.normalize_z = bool(normalize_z)
        self.smplx_model_root = resolve_smplx_model_root(self.root, self.cache_path, smplx_model_root)
        if pose_source not in {"param_joints", "smplx"}:
            raise ValueError("pose_source must be one of {'param_joints', 'smplx'}")

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

    def close(self) -> None:
        if self.lmdb_envs is not None:
            for env in self.lmdb_envs.values():
                env.close()
        self.lmdb_envs = None
        self._lmdb_owner_pid = None
        self.smplx_models = None
        self._smplx_owner_pid = None

    def __enter__(self) -> "M4HumanReader":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

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

    def load_split_indices(self, protocol: str, split_id: str, subset: str) -> list[tuple[int, int, int]]:
        if protocol not in VALID_PROTOCOLS:
            raise ValueError(f"protocol must be one of {VALID_PROTOCOLS}, got {protocol}")
        if split_id not in VALID_SPLIT_IDS:
            raise ValueError(f"split_id must be one of {VALID_SPLIT_IDS}, got {split_id}")
        if subset not in VALID_SUBSETS:
            raise ValueError(f"subset must be one of {VALID_SUBSETS}, got {subset}")

        indices_path = self.cache_path / "indeces.pkl.gz"
        if not indices_path.exists():
            raise FileNotFoundError(f"Missing split index file: {indices_path}")
        with gzip.open(indices_path, "rb") as f:
            split_indices = pickle.load(f)
        indicators = split_indices[protocol][split_id][subset]
        return [
            tuple(int(v) for v in indicator)
            for indicator in indicators
            if tuple(indicator) not in NON_VALID_INDICATOR_SET
        ]

    def build_segments(
        self,
        protocol: str,
        split_id: str,
        subset: str,
        min_raw_frames: int = 2,
    ) -> list[MotionSegment]:
        indicators = self.load_split_indices(protocol, split_id, subset)
        grouped: dict[tuple[int, int], list[tuple[int, tuple[int, int, int]]]] = defaultdict(list)
        for indicator in indicators:
            subject, action, frame = indicator
            grouped[(subject, action)].append((frame, indicator))

        segments: list[MotionSegment] = []

        def emit(segment: list[tuple[int, int, int]]) -> None:
            if len(segment) < min_raw_frames:
                return
            start = segment[0]
            end = segment[-1]
            segments.append(
                MotionSegment(
                    protocol=protocol,
                    split_id=split_id,
                    subset=subset,
                    subject=start[0],
                    action=start[1],
                    start_frame=start[2],
                    end_frame=end[2],
                    indicators=tuple(segment),
                )
            )

        for key in sorted(grouped):
            frames = sorted(grouped[key], key=lambda x: x[0])
            segment: list[tuple[int, int, int]] = []
            prev_frame: int | None = None
            for frame, indicator in frames:
                if prev_frame is not None and frame != prev_frame + 1:
                    emit(segment)
                    segment = []
                segment.append(indicator)
                prev_frame = frame
            emit(segment)
        return segments

    def read_params_and_calib(self, indicator: tuple[int, int, int]) -> tuple[dict[str, Any], dict[str, Any]]:
        self._open_lmdb_envs()
        assert self.lmdb_envs is not None
        key = str(list(indicator)).encode()
        with self.lmdb_envs["params"].begin() as txn_param, self.lmdb_envs["calib"].begin() as txn_calib:
            param_blob = txn_param.get(key)
            calib_blob = txn_calib.get(key)
        if param_blob is None or calib_blob is None:
            raise KeyError(f"Missing params/calib for indicator {indicator}")
        return unpack_dict_np(param_blob), unpack_dict_np(calib_blob)

    def load_param_joints_radar(self, indicator: tuple[int, int, int]) -> np.ndarray:
        params, calib = self.read_params_and_calib(indicator)
        joints = np.asarray(params["joints"], dtype=np.float32)
        if joints.ndim != 2 or joints.shape[0] < self.num_joints or joints.shape[1] < 3:
            raise ValueError(f"Invalid params.joints for indicator {indicator}: {joints.shape}")

        vicon_to_cam_rot = np.asarray(calib["vicon_to_cam_rotmatrix"], dtype=np.float32)
        vicon_to_cam_t = np.asarray(calib["vicon_to_cam_tvec"], dtype=np.float32) / 1000.0
        radar_to_cam_rot = np.asarray(calib["radar_to_cam_rotmatrix"], dtype=np.float32)
        radar_to_cam_t = np.asarray(calib["radar_to_cam_tvec"], dtype=np.float32)
        joints_cam = (vicon_to_cam_rot @ joints[:, :3].T).T + vicon_to_cam_t
        joints_radar = (np.linalg.inv(radar_to_cam_rot) @ (joints_cam - radar_to_cam_t).T).T
        if self.normalize_z:
            joints_radar = joints_radar.copy()
            joints_radar[:, 2] -= 1.5
        return joints_radar[: self.num_joints].astype(np.float32, copy=False)

    def load_smplx_segment_radar(self, indicators: tuple[tuple[int, int, int], ...]) -> np.ndarray:
        self._ensure_smplx_models()
        assert self.smplx_models is not None

        batches: dict[str, list[tuple[int, dict[str, Any], np.ndarray, np.ndarray]]] = {
            "male": [],
            "female": [],
        }
        for frame_idx, indicator in enumerate(indicators):
            params, calib = self.read_params_and_calib(indicator)
            subject_key = f"P{indicator[0]}"
            if subject_key not in GENDER_INFO:
                raise KeyError(f"Unknown M4Human subject gender for {subject_key}")
            gender = "male" if float(GENDER_INFO[subject_key]) > 0.5 else "female"
            root_orient, transl = calibrate_smplx_root_to_radar(params, calib)
            if self.normalize_z:
                transl = transl.copy()
                transl[2] -= 1.5
            batches[gender].append((frame_idx, params, root_orient, transl))

        joints_radar = np.zeros((len(indicators), self.num_joints, 3), dtype=np.float32)
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
                regressor = model.J_regressor[: self.num_joints].to(vertices_mm.device)
                joints_m = torch.einsum("ij,bjc->bic", regressor, vertices_mm) / 1000.0
            group_joints = joints_m.cpu().numpy().astype(np.float32, copy=False)
            for local_idx, (frame_idx, _, _, _) in enumerate(entries):
                joints_radar[frame_idx] = group_joints[local_idx]
        return joints_radar

    def load_segment_joints_radar(self, segment: MotionSegment) -> np.ndarray:
        if self.pose_source == "param_joints":
            return np.stack([self.load_param_joints_radar(indicator) for indicator in segment.indicators], axis=0)
        return self.load_smplx_segment_radar(segment.indicators)
