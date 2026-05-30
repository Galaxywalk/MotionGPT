"""M4Human feature conversion utilities for MotionGPT.

Keep package imports light so cached-feature tools do not require LMDB/SMPL-X
dependencies unless the raw M4Human reader is explicitly imported.
"""

from .features import HumanMLFeatureConverter

__all__ = ["HumanMLFeatureConverter"]
