import os
import numpy as np
import pytorch_lightning as pl
import torch
from pathlib import Path
from tqdm import tqdm
from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained_vae

def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    datasets = build_data(cfg, phase='token')
    print("datasets module initialized")
    output_dir = os.path.join(datasets.hparams.data_root, cfg.DATASET.CODE_PATH)

    os.makedirs(output_dir, exist_ok=True)

    # create model
    model = build_model(cfg, datasets)
    if hasattr(model, "motion_vae"):
        model.vae = model.motion_vae
    print("model loaded")

    # Strict load vae model
    assert cfg.TRAIN.PRETRAINED_VAE
    load_pretrained_vae(cfg, model)

    device = torch.device("cuda" if cfg.ACCELERATOR == "gpu"
                          and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(datasets.train_dataloader(),
                          desc=f'motion tokenize'):
            names = batch['text']
            poses = batch['motion'].to(device).float()
            lengths = batch['length']

            for i, name in enumerate(names):
                if lengths[i] == 0:
                    continue

                pose = poses[i:i + 1, :lengths[i]]
                target, _ = model.vae.encode(pose)
                target = target.to('cpu').numpy()

                target_path = os.path.join(output_dir, name + '.npy')
                Path(target_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(target_path, target)

    print(
        f'Motion tokenization done, the motion tokens are saved to {output_dir}'
    )


if __name__ == "__main__":
    main()
