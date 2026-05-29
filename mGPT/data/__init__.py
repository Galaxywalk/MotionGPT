import pytorch_lightning as pl
from torch.utils.data import DataLoader


class BASEDataModule(pl.LightningDataModule):
    def __init__(self, collate_fn):
        super().__init__()

        self.dataloader_options = {"collate_fn": collate_fn}
        self.persistent_workers = True
        self.is_mm = False

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def _make_dataloader(self, dataset, split_cfg, shuffle=False, batch_size=None):
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["batch_size"] = batch_size or split_cfg.BATCH_SIZE
        dataloader_options["num_workers"] = split_cfg.NUM_WORKERS
        dataloader_options["shuffle"] = shuffle
        dataloader_options["pin_memory"] = split_cfg.get("PIN_MEMORY", False)
        dataloader_options["drop_last"] = split_cfg.get("DROP_LAST", False)

        num_workers = dataloader_options["num_workers"]
        if num_workers > 0:
            dataloader_options["persistent_workers"] = split_cfg.get(
                "PERSISTENT_WORKERS", True)
            prefetch_factor = split_cfg.get("PREFETCH_FACTOR", None)
            if prefetch_factor is not None:
                dataloader_options["prefetch_factor"] = prefetch_factor
        else:
            dataloader_options["persistent_workers"] = False

        return DataLoader(dataset, **dataloader_options)

    def get_sample_set(self, overrides={}):
        sample_params = self.hparams.copy()
        sample_params.update(overrides)
        return self.DatasetEval(**sample_params)

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = self.Dataset(split=self.cfg.TRAIN.SPLIT,
                                               **self.hparams)
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            params = self.hparams.copy()
            params['code_path'] = None
            params['split'] = self.cfg.EVAL.SPLIT
            self._val_dataset = self.DatasetEval(**params)
        return self._val_dataset

    @property
    def test_dataset(self):
        if self._test_dataset is None:
            # self._test_dataset = self.DatasetEval(split=self.cfg.TEST.SPLIT,
            #                                       **self.hparams)
            params = self.hparams.copy()
            params['code_path'] = None
            params['split'] = self.cfg.TEST.SPLIT
            self._test_dataset = self.DatasetEval( **params)
        return self._test_dataset

    def setup(self, stage=None):
        # Use the getter the first time to load the data
        if stage in (None, "fit"):
            _ = self.train_dataset
            _ = self.val_dataset
        if stage in (None, "test"):
            _ = self.test_dataset

    def train_dataloader(self):
        return self._make_dataloader(
            self.train_dataset,
            self.cfg.TRAIN,
            shuffle=self.cfg.TRAIN.get("SHUFFLE", True),
        )

    def predict_dataloader(self):
        return self._make_dataloader(
            self.test_dataset,
            self.cfg.TEST,
            shuffle=False,
            batch_size=1 if self.is_mm else self.cfg.TEST.BATCH_SIZE,
        )

    def val_dataloader(self):
        # overrides batch_size and num_workers
        return self._make_dataloader(
            self.val_dataset,
            self.cfg.EVAL,
            shuffle=False,
        )

    def test_dataloader(self):
        # overrides batch_size and num_workers
        return self._make_dataloader(
            self.test_dataset,
            self.cfg.TEST,
            shuffle=False,
            batch_size=1 if self.is_mm else self.cfg.TEST.BATCH_SIZE,
        )
