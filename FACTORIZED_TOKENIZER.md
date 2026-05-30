# Factorized Motion Tokenizer

This document summarizes the current best M4Human motion tokenizer in this
repository. It is not the original single-stream MotionGPT 263-D VQ-VAE. The
current best system factorizes motion into:

```text
motion = local discrete tokens + continuous root latent
```

The main motivation is that local body pose and global root trajectory have
different error behavior. Local pose reconstruction error usually stays local,
while root velocity and yaw errors integrate over time and become 196-frame
drift.

## Current Best Checkpoint

The current best factorized setup is the full from-scratch R3 recipe:

```text
local branch: frozen local-only VQ
root branch:  R3 no-skip bottleneck TCN, 2x root latent rate, multiscale loss
```

Artifacts:

- Local VQ:
  `/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_scratch_full_v1/checkpoints/best.pt`
- Best root branch R3:
  `/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_branch_m4human_scratch_full_r3_v1/checkpoints/best.pt`
- Factorized cache:
  `/cpfs01/liangbo/data/MotionGPT/factorized_cache/v1_m4human_xz-y_20hz`

Code:

- `src/motiongpt_m4human/factorized/local_vq.py`
- `src/motiongpt_m4human/factorized/root_branch.py`
- `src/motiongpt_m4human/factorized/cache.py`
- `src/motiongpt_m4human/factorized/representation.py`
- `scripts/train_factorized_scratch_m4human.sh`

## Representation

The original HumanML3D-style 263-D feature mixes root motion and local body
motion in one stream:

```text
root yaw velocity
root local x/z velocity
root height
local joints
local rotations
local velocities
contacts
```

The factorized tokenizer separates these fields.

### Local Branch Input

The local VQ input is 130-D:

```text
root-relative body local joints:       21 * 3
root-frame body local joint velocity:  21 * 3
foot contacts:                         4
total:                                 130
```

It intentionally excludes:

```text
root yaw velocity
root x/z velocity
root height
```

This prevents local tokens from leaking global trajectory.

### Root Branch Input

The root branch uses 4-D root controls:

```text
yaw velocity
local root velocity x
local root velocity z
root height
```

Root velocity is handled in physical units during loss computation.

## Network Structure

### Local VQ

The local branch reuses MotionGPT's VQ-VAE implementation:

```text
local 130-D feature
  -> temporal encoder
  -> VQ codebook
  -> temporal decoder
  -> reconstructed local 130-D feature
```

Important settings:

```text
code_num: 512
code_dim: 512
down_t:   2
stride_t: 2
```

A 196-frame clip produces about 47 local tokens.

### Root Branch R3

The best root branch is a no-skip bottleneck TCN:

```text
root controls + decoded local feature
  -> dilated residual TCN encoder
  -> continuous root latent
  -> dilated residual TCN decoder + local condition
  -> reconstructed root controls
```

Important settings:

```text
architecture:          bottleneck_tcn
width:                 256
latent_width:          256
root_downsample_layers: 1
effective root rate:   2x temporal downsample
tcn_depth:             4
```

This is a strict no-skip bottleneck. The decoder does not receive root encoder
skip features, so it cannot directly copy high-resolution root information from
the input. This is why R3 is a meaningful tokenizer-style result, unlike the
U-Net skip variant, which should be treated as an optimistic upper bound.

## Training Process

Training is staged.

### Stage 1: Build Factorized Cache

The cache stores local and root fields separately:

```text
local_joints
local_joint_vel
local_rot6d
contacts
root_xy
root_yaw
root_height
root_vel_local_mps
root_vel_global_mps
root_yaw_vel_radps
dt
source_domain
valid_mask
features_263
```

Current cache:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_cache/v1_m4human_xz-y_20hz
```

The cache keeps the validated M4Human axis mapping:

```text
axis_mode = xz-y
fps       = 20 Hz
```

### Stage 2: Train Local-Only VQ

The local VQ is trained on M4Human local features only.

Full from-scratch command:

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=5 \
/cpfs01/liangbo/data/conda_envs/mgpt/bin/python \
  -m motiongpt_m4human.factorized.local_vq train \
  --cache-root /cpfs01/liangbo/data/MotionGPT/factorized_cache/v1_m4human_xz-y_20hz \
  --exp-root /cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_scratch_full_v1 \
  --seed 20260531 \
  --epochs 200 \
  --steps-per-epoch 200 \
  --batch-size 256 \
  --window-sizes 64 128 196 \
  --window-weights 0.25 0.25 0.5 \
  --lr 2e-4 \
  --device cuda:0
```

Training losses include:

```text
normalized feature reconstruction
local joint reconstruction
local joint velocity reconstruction
contact reconstruction
VQ commit loss
```

### Stage 3: Train Root Branch

The local VQ is frozen. Only the root branch is trained.

Best full from-scratch R3 command:

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=6 \
/cpfs01/liangbo/data/conda_envs/mgpt/bin/python \
  -m motiongpt_m4human.factorized.root_branch train \
  --architecture bottleneck_tcn \
  --width 256 \
  --latent-width 256 \
  --root-downsample-layers 1 \
  --tcn-depth 4 \
  --lambda-multiscale 20.0 \
  --cache-root /cpfs01/liangbo/data/MotionGPT/factorized_cache/v1_m4human_xz-y_20hz \
  --local-vq-checkpoint /cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_scratch_full_v1/checkpoints/best.pt \
  --exp-root /cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_branch_m4human_scratch_full_r3_v1 \
  --seed 20260531 \
  --epochs 120 \
  --steps-per-epoch 200 \
  --batch-size 256 \
  --window-sizes 64 128 196 \
  --window-weights 0.25 0.25 0.5 \
  --lr 2e-4 \
  --device cuda:0
```

Root losses include:

```text
root control reconstruction
yaw velocity
local x/z root velocity
root height
global root position
global root step
final displacement
path length
multiscale displacement at 25%, 50%, 75%, 100%
smoothness
```

The most important added loss is multiscale displacement. It directly supervises
low-frequency endpoint and trajectory direction error, which was the remaining
root drift mode.

## Results

### Local VQ

Full from-scratch local VQ, M4Human test196:

```text
MPJPE / root-aligned MPJPE: 53.562 / 53.562 mm
unique codes:               511 / 512
effective code count:       359
tokens per 196-frame clip:  46.99
contact F1:                 0.994
```

The earlier shorter local VQ run reached a better local-only test196 MPJPE
(`51.168 mm`). The full scratch recipe is still the current best end-to-end
factorized reconstruction because its root branch is trained longer and reduces
global trajectory error more strongly.

### Root Branch

M4Human test196 comparison:

| experiment | MPJPE / root-align / gap |
| --- | ---: |
| Exp3 single-stream 263-D VQ-VAE | 101.077 / 52.429 / 48.648 mm |
| old no-skip bottleneck root branch | 77.232 / 48.804 / 28.428 mm |
| R1 stronger TCN | 63.412 / 48.570 / 14.843 mm |
| R2 TCN + multiscale displacement | 56.663 / 48.304 / 8.360 mm |
| R3 TCN + multiscale + 2x root latent rate | 55.171 / 48.243 / 6.928 mm |
| full from-scratch R3 | 54.696 / 49.605 / 5.091 mm |

Full from-scratch R3 root metrics on M4Human test196:

```text
root_xz_mean_error: 2.908 mm
final_xz_error:     4.347 mm
path_error:        -0.0004 m
speed_bias:        -0.039 mm/s
```

Length breakdown:

| split/window | MPJPE / root-align / gap | root xz mean | final xz | speed bias |
| --- | ---: | ---: | ---: | ---: |
| test 64 | 54.822 / 49.986 / 4.836 mm | 1.398 mm | 2.086 mm | 0.007 mm/s |
| test 128 | 54.665 / 49.719 / 4.945 mm | 2.110 mm | 3.297 mm | -0.019 mm/s |
| test 196 | 54.696 / 49.605 / 5.091 mm | 2.908 mm | 4.347 mm | -0.039 mm/s |
| val 196 | 45.708 / 42.821 / 2.887 mm | 2.145 mm | 3.512 mm | 0.015 mm/s |

## Interpretation

The main conclusion is that the M4Human failure mode was not simply a loss
weight problem in the original 263-D single-stream tokenizer. It was structural:
root trajectory should not be forced through the same discrete bottleneck as
local articulated pose.

The factorized tokenizer works because:

```text
local body motion -> discrete VQ tokens
global root motion -> continuous root latent
```

The local tokens capture body pose well. The continuous root branch prevents
small root velocity/yaw errors from accumulating into large 196-frame drift.

The full from-scratch R3 is currently the best M4Human reconstruction path in
this repo. Its root gap is about `5 mm` on 196-frame test windows, so remaining
error is dominated by local reconstruction quality rather than accumulated root
drift.

## Limitations

The current tokenizer is not yet the final mixed-domain training system.

Known limitations:

- The factorized experiments are currently M4Human-focused.
- HumanML3D still needs a matching factorized cache and mixed-domain eval.
- The root branch is continuous, not discrete. This is intentional for quality,
  but downstream language modeling needs a clear interface for continuous root
  controls.
- The current root branch reconstructs root controls from root controls plus
  local condition. For generation, we still need to decide whether root controls
  are predicted from text/sensor features, supplied as conditioning, or later
  tokenized separately.

## Recommended Next Steps

1. Build a HumanML3D factorized cache with the same representation.
2. Evaluate R3-style factorized reconstruction on HumanML3D.
3. Train a mixed M4Human/HumanML3D factorized tokenizer.
4. Keep R3 continuous root branch as the quality reference before attempting
   root discretization.
5. Only consider root discrete tokens after the continuous root branch is stable
   across both domains.
