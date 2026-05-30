import json
import random
from pathlib import Path

import numpy as np
from torch.utils import data
from torch.utils.data import RandomSampler, SequentialSampler

from .HumanML3D import HumanML3DDataModule
from .humanml import MotionDatasetVQ


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _window_sizes_from_cfg(cfg, default_size):
    sizes = cfg.get("WINDOW_SIZES", None)
    if sizes is None:
        sizes = [cfg.get("WINDOW_SIZE", default_size)]
    sizes = sorted({int(size) for size in sizes})
    if not sizes:
        raise ValueError("DATASET.M4HUMAN.WINDOW_SIZES must not be empty")
    return sizes


def _window_weights_from_cfg(cfg, window_sizes):
    weights = cfg.get("WINDOW_WEIGHTS", None)
    if weights is None:
        return None
    weights = [float(weight) for weight in weights]
    if len(weights) != len(window_sizes):
        raise ValueError(
            "DATASET.M4HUMAN.WINDOW_WEIGHTS must have the same length as "
            "DATASET.M4HUMAN.WINDOW_SIZES")
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("DATASET.M4HUMAN.WINDOW_WEIGHTS must be non-negative and non-zero")
    return weights


def _split_index_and_window(item, default_window_size):
    if isinstance(item, (tuple, list)):
        item, window_size = item
        return int(item), int(window_size)
    return int(item), int(default_window_size)


class RandomWindowBatchSampler(data.Sampler):
    """Batch sampler that attaches one window size to each yielded batch.

    The first argument intentionally follows torch BatchSampler's ``sampler``
    API. This lets PyTorch Lightning replace it with a DistributedSampler in
    DDP while we keep the random window-size behavior.
    """

    def __init__(
        self,
        sampler,
        batch_size,
        drop_last,
        shuffle=False,
        window_sizes=None,
        window_weights=None,
        seed=1234,
    ):
        self.sampler = sampler
        self.dataset = getattr(sampler, "data_source", getattr(sampler, "dataset", sampler))
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        if window_sizes is None:
            window_sizes = getattr(self.dataset, "window_sizes", None)
        if window_sizes is None:
            window_sizes = [getattr(self.dataset, "window_size", 64)]
        self.window_sizes = [int(size) for size in window_sizes]
        if window_weights is None:
            window_weights = getattr(self.dataset, "window_weights", None)
        self.window_weights = window_weights
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.window_sizes:
            raise ValueError("window_sizes must not be empty")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        if isinstance(self.sampler, data.Dataset):
            indices = list(range(len(self.sampler)))
            if self.shuffle:
                rng.shuffle(indices)
        else:
            indices = list(iter(self.sampler))
            if self.shuffle and isinstance(self.sampler, SequentialSampler):
                rng.shuffle(indices)
        if self.shuffle and isinstance(self.sampler, RandomSampler):
            rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                break
            window_size = rng.choices(
                self.window_sizes,
                weights=self.window_weights,
                k=1,
            )[0]
            yield [(idx, window_size) for idx in batch]

    def __len__(self):
        size = len(self.sampler)
        if self.drop_last:
            return size // self.batch_size
        return (size + self.batch_size - 1) // self.batch_size


class M4HumanFeatureDatasetVQ(data.Dataset):
    """MotionDatasetVQ-compatible reader for cached M4Human 263-D features."""

    def __init__(
        self,
        cache_root,
        split,
        mean,
        std,
        win_size,
        min_motion_length=40,
        tiny=False,
        debug=False,
        max_sequences=0,
        window_sizes=None,
        **kwargs,
    ):
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.mean = mean
        self.std = std
        self.window_size = int(win_size)
        self.window_sizes = (
            sorted({int(size) for size in window_sizes})
            if window_sizes is not None
            else [self.window_size]
        )
        if not self.window_sizes:
            raise ValueError("window_sizes must not be empty")
        min_window_size = min(self.window_sizes)

        if not self.cache_root.exists():
            raise FileNotFoundError(f"M4Human cache root does not exist: {self.cache_root}")
        rows = _read_jsonl(self.cache_root / "sequences.jsonl")
        rows = [
            row for row in rows
            if row.get("subset") == split
            and int(row.get("feature_frames", 0)) >= max(min_window_size, min_motion_length)
        ]
        if max_sequences:
            rows = rows[:int(max_sequences)]
        if tiny or debug:
            rows = rows[: min(len(rows), 100)]
        if not rows:
            raise RuntimeError(
                f"No M4Human cache sequences found for split={split} "
                f"with window_size={self.window_size} under {self.cache_root}"
            )

        self.rows = rows
        self.name_list = [row["id"] for row in rows]
        self.nfeats = int(np.load(self.cache_root / rows[0]["features"], mmap_mode="r").shape[-1])
        self.eligible_rows_by_window = {
            size: [
                row for row in self.rows
                if int(row.get("feature_frames", 0)) >= size
            ]
            for size in self.window_sizes
        }
        for size, eligible_rows in self.eligible_rows_by_window.items():
            if not eligible_rows:
                raise RuntimeError(
                    f"No M4Human cache sequences are at least {size} frames long")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        item, window_size = _split_index_and_window(item, self.window_size)
        row = self.rows[item % len(self.rows)]
        if int(row.get("feature_frames", 0)) < window_size:
            row = random.choice(self.eligible_rows_by_window[window_size])
        motion = np.load(self.cache_root / row["features"], mmap_mode="r")
        if motion.shape[0] < window_size:
            raise RuntimeError(f"M4Human sequence {row['id']} is shorter than {window_size} frames")

        start = random.randint(0, motion.shape[0] - window_size)
        motion = np.asarray(motion[start:start + window_size], dtype=np.float32)
        motion = (motion - self.mean) / self.std

        return None, motion, window_size, None, None, None, None, "m4human"


class MixedMotionDatasetVQ(data.Dataset):
    """Probability-based mixture that preserves the base epoch length."""

    def __init__(self, humanml_dataset, m4human_dataset, m4human_ratio):
        self.humanml_dataset = humanml_dataset
        self.m4human_dataset = m4human_dataset
        self.m4human_ratio = float(m4human_ratio)
        self.window_sizes = getattr(humanml_dataset, "window_sizes", [getattr(humanml_dataset, "window_size", 64)])
        self.window_weights = getattr(humanml_dataset, "window_weights", None)
        if not 0.0 <= self.m4human_ratio <= 1.0:
            raise ValueError(f"m4human_ratio must be in [0, 1], got {self.m4human_ratio}")
        self.name_list = getattr(humanml_dataset, "name_list", [])
        self.nfeats = getattr(humanml_dataset, "nfeats", getattr(m4human_dataset, "nfeats", 263))

    def __len__(self):
        return len(self.humanml_dataset)

    def __getitem__(self, item):
        item, window_size = _split_index_and_window(
            item, getattr(self.humanml_dataset, "window_size", 64))
        if self.m4human_ratio > 0.0 and random.random() < self.m4human_ratio:
            m4_item = random.randrange(len(self.m4human_dataset))
            return self.m4human_dataset[(m4_item, window_size)]
        sample = self.humanml_dataset[(item % len(self.humanml_dataset), window_size)]
        if sample[0] is None:
            return (*sample, "humanml3d")
        return sample


class MultiWindowTrainMixin:
    def _m4_cfg(self):
        return self.cfg.DATASET.get("M4HUMAN", {})

    def _train_window_sizes(self):
        return _window_sizes_from_cfg(self._m4_cfg(), self.hparams.win_size)

    def _train_window_weights(self, window_sizes):
        return _window_weights_from_cfg(self._m4_cfg(), window_sizes)

    def _make_train_dataloader(self, dataset):
        window_sizes = self._train_window_sizes()
        if len(window_sizes) <= 1:
            return self._make_dataloader(
                dataset,
                self.cfg.TRAIN,
                shuffle=self.cfg.TRAIN.get("SHUFFLE", True),
            )

        split_cfg = self.cfg.TRAIN
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["num_workers"] = split_cfg.NUM_WORKERS
        dataloader_options["pin_memory"] = split_cfg.get("PIN_MEMORY", False)
        num_workers = dataloader_options["num_workers"]
        if num_workers > 0:
            dataloader_options["persistent_workers"] = split_cfg.get(
                "PERSISTENT_WORKERS", True)
            prefetch_factor = split_cfg.get("PREFETCH_FACTOR", None)
            if prefetch_factor is not None:
                dataloader_options["prefetch_factor"] = prefetch_factor
        else:
            dataloader_options["persistent_workers"] = False

        dataloader_options["batch_sampler"] = RandomWindowBatchSampler(
            sampler=dataset,
            batch_size=split_cfg.BATCH_SIZE,
            drop_last=split_cfg.get("DROP_LAST", False),
            shuffle=split_cfg.get("SHUFFLE", True),
            window_sizes=window_sizes,
            window_weights=self._train_window_weights(window_sizes),
            seed=self.cfg.SEED_VALUE,
        )
        return data.DataLoader(dataset, **dataloader_options)


class MixedHumanML3DM4HumanDataModule(MultiWindowTrainMixin, HumanML3DDataModule):
    """HumanML3D VQ training mixed with cached M4Human features."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        if cfg.TRAIN.STAGE != "vae":
            raise ValueError("MixedHumanML3DM4HumanDataModule is only intended for TRAIN.STAGE=vae")
        if self.Dataset is not MotionDatasetVQ:
            raise ValueError("MixedHumanML3DM4HumanDataModule requires the VQVAE MotionDatasetVQ path")

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            params = self.hparams.copy()
            m4_cfg = self.cfg.DATASET.get("M4HUMAN", {})
            window_sizes = _window_sizes_from_cfg(m4_cfg, self.hparams.win_size)
            params["window_sizes"] = window_sizes
            humanml_dataset = MotionDatasetVQ(split=self.cfg.TRAIN.SPLIT, **params)

            cache_root = m4_cfg.get("CACHE_ROOT")
            if not cache_root:
                raise ValueError("DATASET.M4HUMAN.CACHE_ROOT must be set for mixed training")
            m4human_dataset = M4HumanFeatureDatasetVQ(
                cache_root=cache_root,
                split=m4_cfg.get("TRAIN_SUBSET", "train"),
                mean=self.hparams.mean,
                std=self.hparams.std,
                win_size=m4_cfg.get("WINDOW_SIZE", self.hparams.win_size),
                window_sizes=window_sizes,
                min_motion_length=m4_cfg.get("MIN_MOTION_LEN", self.cfg.DATASET.HUMANML3D.MIN_MOTION_LEN),
                tiny=self.hparams.debug,
                debug=self.hparams.debug,
                max_sequences=m4_cfg.get("MAX_SEQUENCES", 0),
            )
            self._train_dataset = MixedMotionDatasetVQ(
                humanml_dataset=humanml_dataset,
                m4human_dataset=m4human_dataset,
                m4human_ratio=m4_cfg.get("MIX_RATIO", 0.3),
            )
            self._train_dataset.window_sizes = window_sizes
            self._train_dataset.window_weights = _window_weights_from_cfg(
                m4_cfg, window_sizes)
            print(
                "Mixed VQ dataset: "
                f"HumanML3D={len(humanml_dataset)} sequences, "
                f"M4Human={len(m4human_dataset)} sequences, "
                f"m4human_ratio={self._train_dataset.m4human_ratio:.2f}"
            )
        return self._train_dataset

    def train_dataloader(self):
        return self._make_train_dataloader(self.train_dataset)


class M4HumanOnlyDataModule(MultiWindowTrainMixin, HumanML3DDataModule):
    """HumanML3D-compatible VQ training module that trains only on M4Human."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        if cfg.TRAIN.STAGE != "vae":
            raise ValueError("M4HumanOnlyDataModule is only intended for TRAIN.STAGE=vae")

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            m4_cfg = self.cfg.DATASET.get("M4HUMAN", {})
            cache_root = m4_cfg.get("CACHE_ROOT")
            if not cache_root:
                raise ValueError("DATASET.M4HUMAN.CACHE_ROOT must be set for M4Human training")
            window_sizes = _window_sizes_from_cfg(m4_cfg, self.hparams.win_size)
            self._train_dataset = M4HumanFeatureDatasetVQ(
                cache_root=cache_root,
                split=m4_cfg.get("TRAIN_SUBSET", "train"),
                mean=self.hparams.mean,
                std=self.hparams.std,
                win_size=m4_cfg.get("WINDOW_SIZE", self.hparams.win_size),
                window_sizes=window_sizes,
                min_motion_length=m4_cfg.get("MIN_MOTION_LEN", self.cfg.DATASET.HUMANML3D.MIN_MOTION_LEN),
                tiny=self.hparams.debug,
                debug=self.hparams.debug,
                max_sequences=m4_cfg.get("MAX_SEQUENCES", 0),
            )
            self._train_dataset.window_weights = _window_weights_from_cfg(
                m4_cfg, window_sizes)
            print(
                "M4Human-only VQ dataset: "
                f"M4Human={len(self._train_dataset)} sequences, "
                f"window_sizes={window_sizes}"
            )
        return self._train_dataset

    def train_dataloader(self):
        return self._make_train_dataloader(self.train_dataset)
