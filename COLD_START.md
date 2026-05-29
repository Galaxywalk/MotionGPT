# MotionGPT Cold Start Checklist

This checklist records the setup that was used to prepare this fork for
reproducing MotionGPT training. It assumes the repository is checked out on a
machine with a large shared filesystem such as `/cpfs01`.

## 1. Clone The Fork

```bash
git clone git@github.com:Galaxywalk/MotionGPT.git
cd MotionGPT
git remote add upstream https://github.com/OpenMotionLab/MotionGPT
```

The H200-oriented training changes are on `main`. They were introduced by:

```bash
git checkout main
git log -1
```

```text
af197b0 Optimize MotionGPT training for H200
```

## 2. Choose A Data Root

Keep datasets, dependencies, checkpoints, and package caches off the home
filesystem.

```bash
export DATA_ROOT=/cpfs01/liangbo/data/MotionGPT
mkdir -p "$DATA_ROOT"/{datasets,deps,checkpoints}
```

Create repository symlinks:

```bash
ln -sfn "$DATA_ROOT/datasets" datasets
ln -sfn "$DATA_ROOT/deps" deps
ln -sfn "$DATA_ROOT/checkpoints" checkpoints
```

Expected local links:

```text
datasets   -> /cpfs01/liangbo/data/MotionGPT/datasets
deps       -> /cpfs01/liangbo/data/MotionGPT/deps
checkpoints -> /cpfs01/liangbo/data/MotionGPT/checkpoints
```

Do not commit these symlinks or downloaded data files.

## 3. Create The Conda Environment

Use a path-based environment on the large filesystem:

```bash
export CONDA_PKGS_DIRS=/cpfs01/liangbo/data/conda_pkgs
export PIP_CACHE_DIR=/cpfs01/liangbo/data/pip_cache
export TMPDIR=/cpfs01/liangbo/data/tmp

conda create -p /cpfs01/liangbo/data/conda_envs/mgpt python=3.10 -y
conda activate /cpfs01/liangbo/data/conda_envs/mgpt
pip install -e .
```

The verified environment used these key versions:

```text
torch==2.0.0+cu118
pytorch-lightning==2.0.9
transformers==4.31.0
diffusers==0.20.2
numpy==1.23.5
spacy==3.6.1
```

Install `en_core_web_sm` matching spaCy 3.6.x. If the normal spaCy downloader
is slow or blocked, download the wheel directly and install it with pip.

## 4. Prepare HumanML3D

Target layout:

```text
$DATA_ROOT/datasets/humanml3d/
  train.txt
  val.txt
  test.txt
  texts/
  new_joint_vecs/
  template_pretrain.json
  template_instructions.json
```

The prepared copy had:

```text
texts: 29232 txt files
new_joint_vecs: 29228 npy files
split ids: 29228, missing text/motion: 0
```

Google Drive downloads may fail through `gdown` when confirmation pages or
quota checks are involved. Practical fallback:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
wget --no-proxy --continue "<resolved-google-drive-download-url>"
```

After extraction, keep the dataset under:

```text
/cpfs01/liangbo/data/MotionGPT/datasets/humanml3d
```

## 5. Prepare T2M Evaluator, Mean/Std, And GloVe

Target layout:

```text
$DATA_ROOT/deps/t2m/
$DATA_ROOT/deps/glove -> t2m/glove
```

Required files:

```bash
test -f deps/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy
test -f deps/t2m/t2m/Comp_v6_KLD01/meta/mean.npy
test -f deps/t2m/t2m/text_mot_match/model/finest.tar
test -f deps/glove/our_vab_data.npy
```

Some archives unpack with one extra nested `t2m` directory. In that case, add
symlinks so the default config paths resolve:

```bash
cd "$DATA_ROOT/deps/t2m/t2m"
ln -sfn t2m/VQVAEV3_CB1024_CMT_H1024_NRES3 VQVAEV3_CB1024_CMT_H1024_NRES3
ln -sfn t2m/Comp_v6_KLD01 Comp_v6_KLD01
ln -sfn t2m/text_mot_match text_mot_match
cd -
ln -sfn t2m/glove "$DATA_ROOT/deps/glove"
```

## 6. Download Hugging Face Assets

Install or log in to the Hugging Face CLI as needed, then download:

```bash
hf download google/flan-t5-base \
  --local-dir "$DATA_ROOT/deps/flan-t5-base"

hf download OpenMotionLab/MotionGPT-base \
  motiongpt_s3_h3d.tar \
  --local-dir "$DATA_ROOT/checkpoints/MotionGPT-base"
```

Required files:

```bash
test -f deps/flan-t5-base/config.json
test -f checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar
```

## 7. Prepare SMPL And SMPLH

Target layout:

```text
$DATA_ROOT/deps/smpl_models/
  smpl/SMPL_NEUTRAL.pkl
  smpl/SMPL_MALE.pkl
  smpl/SMPL_FEMALE.pkl
  smplh/SMPLH_NEUTRAL.npz
  smplh/SMPLH_MALE.npz
  smplh/SMPLH_FEMALE.npz
```

Compatibility symlink for the default render config:

```bash
mkdir -p "$DATA_ROOT/deps/smpl"
ln -sfn ../smpl_models "$DATA_ROOT/deps/smpl/smpl_models"
```

## 8. Validate The Setup

Basic import check:

```bash
python - <<'PY'
import torch
import pytorch_lightning as pl
import spacy

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("lightning", pl.__version__)
spacy.load("en_core_web_sm")
print("imports ok")
PY
```

Compile key files:

```bash
python -m py_compile \
  mGPT/config.py \
  train.py \
  mGPT/data/__init__.py \
  mGPT/models/mgpt.py \
  mGPT/models/base.py \
  mGPT/metrics/base.py \
  scripts/get_motion_code.py
```

CPU-only smoke check for the stage-1 tokenizer path:

```bash
python - <<'PY'
import sys
sys.argv = ["check", "--cfg", "configs/config_h3d_stage1.yaml", "--nodebug"]

from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model

cfg = parse_args(phase="train")
cfg.ACCELERATOR = "cpu"
cfg.DEVICE = [1]
cfg.TRAIN.NUM_WORKERS = 0
cfg.EVAL.NUM_WORKERS = 0
cfg.TEST.NUM_WORKERS = 0
cfg.TRAIN.BATCH_SIZE = 2
cfg.EVAL.BATCH_SIZE = 2
cfg.METRIC.TYPE = ["MRMetrics"]
cfg.LOGGER.WANDB.params.project = None

datamodule = build_data(cfg)
datamodule.setup("fit")
batch = next(iter(datamodule.train_dataloader()))
model = build_model(cfg, datamodule)
loss = model.training_step(batch, 0)
print("loss_finite", bool(loss.isfinite()))
print("stage1_lm_is_none", model.lm is None)
PY
```

## 9. Generate Motion Tokens For Stage 2

Stage 1 trains the motion tokenizer, which is the VQ-VAE. Stage 2 trains the
motion-language model using motion token codes. After the tokenizer checkpoint
is available, generate codes with:

```bash
python scripts/get_motion_code.py \
  --cfg configs/config_h3d_stage2.yaml \
  --nodebug
```

Expected output:

```text
datasets/humanml3d/TOKENS/
```

## 10. H200 Training Notes

Always pass `--nodebug` for full training runs. Without it, the default config
uses debug behavior and tiny subsets.

Recommended H200 multi-GPU overrides:

```yaml
DEVICE: [0,1,2,3,4,5,6,7]
TRAIN:
  PRECISION: bf16-mixed
  MATMUL_PRECISION: high
  DDP_FIND_UNUSED_PARAMETERS: False
  BENCHMARK: True
```

Stage summary:

```text
Stage 1: VQ-VAE motion tokenizer
Stage 2: motion-language pretraining on generated motion codes
Stage 3: instruction tuning
```

The fork includes code changes that avoid loading FLAN/T5 during stage 1,
initialize metrics lazily, expose precision/DDP/DataLoader knobs, and make
motion token generation work without hard-coded CUDA.
