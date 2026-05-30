from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from mGPT.config import get_module_config
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model


def _load_cfg(args: argparse.Namespace):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_assets = OmegaConf.load(args.cfg_assets)
    cfg_base = OmegaConf.load(Path(cfg_assets.CONFIG_FOLDER) / "default.yaml")
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(args.cfg))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)

    cfg.DEBUG = False
    cfg.DEVICE = [int(args.device_index)]
    cfg.TEST.CHECKPOINTS = args.checkpoint
    cfg.TEST.SPLIT = args.split
    cfg.TEST.BATCH_SIZE = args.batch_size
    cfg.TEST.NUM_WORKERS = args.num_workers
    cfg.TEST.REPLICATION_TIMES = args.replication_times
    cfg.TEST.SAVE_PREDICTIONS = False
    cfg.FOLDER_EXP = str(Path(args.out_json).expanduser().resolve().parent / "official_eval_tmp")
    cfg.TIME = "official_eval"
    return cfg


def _load_state_dict(checkpoint: str) -> dict[str, Any]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    return payload["state_dict"]


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = _load_cfg(args)
    pl.seed_everything(cfg.SEED_VALUE)

    datamodule = build_data(cfg)
    model = build_model(cfg, datamodule)
    model.load_state_dict(_load_state_dict(args.checkpoint), strict=False)

    trainer = pl.Trainer(
        benchmark=False,
        accelerator=cfg.ACCELERATOR,
        devices=[int(args.device_index)],
        default_root_dir=cfg.FOLDER_EXP,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=not args.no_progress,
        logger=None,
        callbacks=[],
    )

    rep_metrics: list[dict[str, float]] = []
    for _ in range(int(args.replication_times)):
        metrics = trainer.test(model, datamodule=datamodule, verbose=False)[0]
        rep_metrics.append({key: _to_float(value) for key, value in metrics.items()})

    keys = sorted(rep_metrics[0])
    mean_metrics = {
        key: float(np.mean([metrics[key] for metrics in rep_metrics]))
        for key in keys
    }
    payload = {
        "checkpoint": args.checkpoint,
        "cfg": args.cfg,
        "split": args.split,
        "batch_size": args.batch_size,
        "replication_times": args.replication_times,
        "metrics": mean_metrics,
        "replication_metrics": rep_metrics,
    }

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MotionGPT's official HumanML3D VQ-VAE test metrics.")
    parser.add_argument("--cfg", default="configs/config_h3d_stage1.yaml")
    parser.add_argument("--cfg-assets", default="configs/assets.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--replication-times", type=int, default=1)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
