# Experiment Index

This file indexes the major experiment families and artifact locations for the
M4Human tokenizer work. Large data, checkpoints, caches, and JSON summaries are
stored outside Git under:

```text
/cpfs01/liangbo/data/MotionGPT
```

Use this index with `M4HUMAN_EVAL_20260530.md` for chronological context and
`PROJECT_STATUS.md` for the current stage conclusion.

## Core Data and Caches

```text
M4Human cache, final factorized format:
/cpfs01/liangbo/data/MotionGPT/factorized_cache/v1_m4human_xz-y_20hz

M4Human SMPL-X/feature cache used during earlier eval:
/cpfs01/liangbo/data/MotionGPT/m4human_cache/v2_xz-y_param_joints_m4ref_20hz

Factorized experiment root:
/cpfs01/liangbo/data/MotionGPT/factorized_experiments
```

Current code entry points:

```text
src/motiongpt_m4human/factorized/cache.py
src/motiongpt_m4human/factorized/dataset.py
src/motiongpt_m4human/factorized/representation.py
src/motiongpt_m4human/factorized/recover.py
```

## Single-Stream M4Human Diagnostics

Purpose: test whether the original MotionGPT 263-D VQ-VAE can reconstruct
M4Human and identify the long-window drift source.

Important artifact families:

```text
/cpfs01/liangbo/data/MotionGPT/length_drift_analysis
/cpfs01/liangbo/data/MotionGPT/root_distribution_analysis
/cpfs01/liangbo/data/MotionGPT/root_reconstruction_analysis
/cpfs01/liangbo/data/MotionGPT/m4human_axis_check
```

Key conclusions:

```text
Exp3 mixed multi-length, M4Human test196:
MPJPE / root-align / gap = 101.077 / 52.429 / 48.648 mm

Root oracle:
GT local velocity reduces MPJPE by about 30.6 mm.
GT yaw reduces MPJPE by about 9.6 mm.
```

The root oracle script is:

```text
src/motiongpt_m4human/eval_root_oracle.py
```

## Root Correction and Loss Experiments

Purpose: check whether losses or light correction heads can fix single-stream
root drift without changing the tokenizer structure.

Experiment families include:

```text
mixed multi-length finetuning
recover-space final/path losses
root velocity scale calibration
speed statistic losses
root step displacement losses
decoded-263D root correction head
```

Conclusion: these were useful diagnostics but not the final path. The best
improvement came from training window length and factorization, not from adding
more scalar root losses to the single-stream VQ-VAE.

## Factorized Local VQ

Purpose: learn discrete tokens for local body pose without root trajectory.

Main checkpoints:

```text
shorter local VQ:
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_v1/checkpoints/best.pt

full scratch local VQ:
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_scratch_full_v1/checkpoints/best.pt
```

Important summaries:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_v1/train_summary.json
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_v1/eval/best_test196.json
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_scratch_full_v1/train_summary.json
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_scratch_full_v1/eval/best_test196.json
```

Current preferred local checkpoint for the full Root-FAST eval:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_v1/checkpoints/best.pt
```

M4Human test196:

```text
local-only MPJPE = 51.168 mm
tokens/window    ~= 46.99
vocab            = 512
```

Code:

```text
src/motiongpt_m4human/factorized/local_vq.py
```

## R3 Continuous Root Branch

Purpose: establish a high-quality root trajectory reference after factorizing
motion into local body and root trajectory.

Main checkpoint:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_branch_m4human_scratch_full_r3_v1/checkpoints/best.pt
```

Reproduction script:

```text
scripts/train_factorized_scratch_m4human.sh
```

M4Human test196:

```text
MPJPE / root-align / gap = 54.696 / 49.605 / 5.091 mm
root_xz_mean_error       = 2.908 mm
```

Conclusion: R3 proves the root/local split works. It is not the final compact
tokenizer because the root latent is `98 x 256` continuous values per 196-frame
clip.

Code:

```text
src/motiongpt_m4human/factorized/root_branch.py
```

## Root-FAST Continuous DCT Codec

Purpose: test whether root controls can be represented compactly without any
neural training.

Experiment root:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_dct_v1
```

Representative root-only M4Human test196 results:

| config | values/window | MPJPE | root xz mean | final xz |
| --- | ---: | ---: | ---: | ---: |
| chunk=16, K=4 | 208 | 1.20 mm | 1.12 mm | 0.42 mm |
| chunk=16, K=2 | 104 | 5.02 mm | 4.80 mm | 3.64 mm |
| chunk=32, K=4 | 112 | 4.16 mm | 3.96 mm | 2.94 mm |
| chunk=32, K=2 | 56 | 18.06 mm | 17.61 mm | 11.43 mm |

Conclusion: root trajectory is highly compressible in the local-command DCT
space.

Code:

```text
src/motiongpt_m4human/factorized/root_fast_codec.py
```

## Root-FAST Quantization

Purpose: discretize DCT root coefficients into token-like codes.

Experiment roots:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_scalar_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_product_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_vector_fixed_v2
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_rvq_v1
```

Important summary:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_rvq_v1/combined_summary.json
```

Best root-only RVQ frontier:

| root tokens | config | bits/window | MPJPE | root xz mean | final xz |
| ---: | --- | ---: | ---: | ---: | ---: |
| 28 | chunk32,K2,vocab1024,d4 | 280 | 35.12 mm | 34.11 mm | 45.47 mm |
| 52 | chunk16,K2,vocab1024,d4 | 520 | 23.23 mm | 22.52 mm | 34.07 mm |
| 56 | chunk32,K2,vocab512,d8 | 504 | 20.15 mm | 19.64 mm | 16.55 mm |
| 104 | chunk16,K2,vocab1024,d8 | 1040 | 7.03 mm | 6.76 mm | 7.47 mm |

Conclusion: plain vector VQ is too lossy, product VQ is better, and RVQ is the
best root-token path so far.

Code:

```text
src/motiongpt_m4human/factorized/root_fast_quantize.py
```

## Full Local VQ + Root-FAST RVQ Eval

Purpose: combine local body tokens and discrete root tokens in one full motion
reconstruction path.

Experiment root:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_full_eval_v1
```

Important summary:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_full_eval_v1/combined_summary.json
```

Recommended full-tokenizer results on M4Human test196:

| local checkpoint | root setting | root tokens | total tokens | MPJPE | root-align | gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| local_v1 | balanced56 | 56 | 102.99 | 58.53 mm | 53.79 mm | 4.74 mm |
| local_v1 | high104_vocab1024 | 104 | 150.99 | 52.79 mm | 52.14 mm | 0.65 mm |

Conclusion: the high104 setting nearly reaches the local VQ upper bound. The
next reconstruction bottleneck is local body tokenization, not root trajectory.

Code:

```text
src/motiongpt_m4human/factorized/root_fast_full_eval.py
```

## Historical Documentation

Use these documents for detailed reasoning:

```text
M4HUMAN_EVAL_20260530.md       chronological experiment log
FACTORIZED_TOKENIZER.md        model and training details
ROOT_LATENT_COMPRESSION.md     root latent size analysis
ROOT_FAST_TOKENIZER_TODO.md    Root-FAST frontier and next TODO
PROJECT_STATUS.md              current stage summary
MMWAVE_MERGE_PLAN.md           future merge interface
```
