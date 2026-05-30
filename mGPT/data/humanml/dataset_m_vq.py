import random
import codecs as cs
import numpy as np
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from .dataset_m import MotionDataset
from .dataset_t2m import Text2MotionDataset


class MotionDatasetVQ(Text2MotionDataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length,
        min_motion_length,
        win_size,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        window_sizes=None,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        # Filter out the motions that are too short
        self.window_size = win_size
        if window_sizes is None:
            self.window_sizes = [int(win_size)]
        else:
            self.window_sizes = sorted({int(size) for size in window_sizes})
            if not self.window_sizes:
                raise ValueError("window_sizes must not be empty")
        min_window_size = min(self.window_sizes)
        name_list = list(self.name_list)
        for name in self.name_list:
            motion = self.data_dict[name]["motion"]
            if motion.shape[0] < min_window_size:
                name_list.remove(name)
                self.data_dict.pop(name)
        self.name_list = name_list
        self.eligible_names_by_window = {
            size: [
                name for name in self.name_list
                if self.data_dict[name]["motion"].shape[0] >= size
            ]
            for size in self.window_sizes
        }
        for size, names in self.eligible_names_by_window.items():
            if not names:
                raise RuntimeError(
                    f"No HumanML3D motions are at least {size} frames long")

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        window_size = self.window_size
        if isinstance(item, (tuple, list)):
            item, window_size = item
            window_size = int(window_size)

        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, length = data["motion"], data["length"]
        if motion.shape[0] < window_size:
            name = random.choice(self.eligible_names_by_window[window_size])
            data = self.data_dict[name]
            motion, length = data["motion"], data["length"]

        idx = random.randint(0, motion.shape[0] - window_size)
        motion = motion[idx:idx + window_size]
        motion = (motion - self.mean) / self.std

        return None, motion, window_size, None, None, None, None,
