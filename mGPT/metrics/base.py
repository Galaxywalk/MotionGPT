from torch import nn


class BaseMetrics(nn.Module):
    def __init__(self, cfg, datamodule, debug, **kwargs) -> None:
        super().__init__()

        njoints = datamodule.njoints
        metric_types = set(cfg.METRIC.TYPE)

        data_name = datamodule.name
        if data_name in ["humanml3d", "kit"]:
            if "TM2TMetrics" in metric_types:
                from .t2m import TM2TMetrics
                self.TM2TMetrics = TM2TMetrics(
                    cfg=cfg,
                    dataname=data_name,
                    diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            if "M2TMetrics" in metric_types or cfg.model.params.task == "m2t":
                from .m2t import M2TMetrics
                self.M2TMetrics = M2TMetrics(
                    cfg=cfg,
                    w_vectorizer=datamodule.hparams.w_vectorizer,
                    diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP)
            if "MMMetrics" in metric_types or "TM2TMetrics" in metric_types:
                from .mm import MMMetrics
                self.MMMetrics = MMMetrics(
                    cfg=cfg,
                    dataname=data_name,
                    mm_num_times=cfg.METRIC.MM_NUM_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                )

        if "MRMetrics" in metric_types:
            from .mr import MRMetrics
            self.MRMetrics = MRMetrics(
                njoints=njoints,
                jointstype=cfg.DATASET.JOINT_TYPE,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )
        if "PredMetrics" in metric_types:
            from .m2m import PredMetrics
            self.PredMetrics = PredMetrics(
                cfg=cfg,
                njoints=njoints,
                jointstype=cfg.DATASET.JOINT_TYPE,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                task=cfg.model.params.task,
            )
