from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..features import motion_process


NFEATS = 263
JOINTS_NUM = 22
ROOT_DIM = 4
RIC_DIM = (JOINTS_NUM - 1) * 3
ROT6D_DIM = (JOINTS_NUM - 1) * 6
LOCAL_VEL_DIM = JOINTS_NUM * 3
ROT6D_START = ROOT_DIM + RIC_DIM
LOCAL_VEL_START = ROT6D_START + ROT6D_DIM
CONTACT_START = LOCAL_VEL_START + LOCAL_VEL_DIM


def recover_root_np(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    root_quat, root_pos = motion_process.recover_root_rot_pos(tensor)
    yaw = torch.atan2(root_quat[..., 2], root_quat[..., 0])
    return yaw.cpu().numpy().astype(np.float32), root_pos.cpu().numpy().astype(np.float32)


def features_to_factorized(
    features: np.ndarray,
    fps: float,
    source_domain: str,
) -> dict[str, np.ndarray | str | float]:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[-1] != NFEATS:
        raise ValueError(f"Expected HumanML3D features [T,{NFEATS}], got {features.shape}")
    if features.shape[0] < 2:
        raise ValueError("At least two feature frames are required")

    dt = 1.0 / float(fps)
    root_yaw, root_pos = recover_root_np(features)
    root_xy = root_pos[:, [0, 2]]
    root_height = features[:, 3:4].copy()

    local_joints = np.zeros((features.shape[0], JOINTS_NUM, 3), dtype=np.float32)
    local_joints[:, 0, 1] = features[:, 3]
    local_joints[:, 1:] = features[:, ROOT_DIM:ROOT_DIM + RIC_DIM].reshape(
        features.shape[0],
        JOINTS_NUM - 1,
        3,
    )

    root_vel_local_mps = features[:, 1:3].copy() / dt
    root_vel_global_mps = np.zeros_like(root_vel_local_mps)
    root_vel_global_mps[:-1] = np.diff(root_xy, axis=0) / dt
    if len(root_vel_global_mps) > 1:
        root_vel_global_mps[-1] = root_vel_global_mps[-2]
    root_yaw_vel_radps = features[:, 0:1].copy() / dt

    return {
        "local_joints": local_joints,
        "local_joint_vel": features[:, LOCAL_VEL_START:LOCAL_VEL_START + LOCAL_VEL_DIM].reshape(
            features.shape[0],
            JOINTS_NUM,
            3,
        ).copy(),
        "local_rot6d": features[:, ROT6D_START:ROT6D_START + ROT6D_DIM].reshape(
            features.shape[0],
            JOINTS_NUM - 1,
            6,
        ).copy(),
        "contacts": features[:, CONTACT_START:CONTACT_START + 4].copy(),
        "root_xy": root_xy.copy(),
        "root_yaw": root_yaw[:, None].copy(),
        "root_height": root_height,
        "root_vel_local_mps": root_vel_local_mps,
        "root_vel_global_mps": root_vel_global_mps,
        "root_yaw_vel_radps": root_yaw_vel_radps,
        "dt": np.asarray(dt, dtype=np.float32),
        "source_domain": source_domain,
        "valid_mask": np.ones((features.shape[0],), dtype=np.bool_),
        "features_263": features.copy(),
    }


def factorized_to_minimal_features(arrays: dict[str, Any]) -> np.ndarray:
    local_joints = np.asarray(arrays["local_joints"], dtype=np.float32)
    root_vel_local = np.asarray(arrays["root_vel_local_mps"], dtype=np.float32)
    root_yaw_vel = np.asarray(arrays["root_yaw_vel_radps"], dtype=np.float32)
    root_height = np.asarray(arrays["root_height"], dtype=np.float32)
    contacts = np.asarray(arrays.get("contacts", np.zeros((local_joints.shape[0], 4))), dtype=np.float32)
    dt = float(np.asarray(arrays["dt"]).reshape(()))

    features = np.zeros((local_joints.shape[0], NFEATS), dtype=np.float32)
    features[:, 0:1] = root_yaw_vel * dt
    features[:, 1:3] = root_vel_local * dt
    features[:, 3:4] = root_height
    features[:, ROOT_DIM:ROOT_DIM + RIC_DIM] = local_joints[:, 1:].reshape(
        local_joints.shape[0],
        RIC_DIM,
    )

    if "local_rot6d" in arrays:
        features[:, ROT6D_START:ROT6D_START + ROT6D_DIM] = np.asarray(
            arrays["local_rot6d"],
            dtype=np.float32,
        ).reshape(local_joints.shape[0], ROT6D_DIM)
    if "local_joint_vel" in arrays:
        features[:, LOCAL_VEL_START:LOCAL_VEL_START + LOCAL_VEL_DIM] = np.asarray(
            arrays["local_joint_vel"],
            dtype=np.float32,
        ).reshape(local_joints.shape[0], LOCAL_VEL_DIM)
    features[:, CONTACT_START:CONTACT_START + 4] = contacts
    return features
