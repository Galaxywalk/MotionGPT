from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..features import recover_from_ric
from .representation import factorized_to_minimal_features


def recover_joints_from_factorized(arrays: dict[str, Any]) -> np.ndarray:
    features = factorized_to_minimal_features(arrays)
    joints = recover_from_ric(torch.from_numpy(features), 22)
    return joints.cpu().numpy().astype(np.float32, copy=False)


def roundtrip_mpjpe_mm(features: np.ndarray, arrays: dict[str, Any]) -> float:
    ref = recover_from_ric(torch.from_numpy(np.asarray(features, dtype=np.float32)), 22)
    rec = torch.from_numpy(recover_joints_from_factorized(arrays))
    return float(torch.linalg.norm(rec - ref, dim=-1).mean().item() * 1000.0)
