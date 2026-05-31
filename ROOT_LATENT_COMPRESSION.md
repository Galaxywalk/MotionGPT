# Root Latent Compression Notes

This document records the current status after the full from-scratch R3
factorized tokenizer experiment, and reframes the next problem: root trajectory
quality is now good, but the R3 continuous root latent is too large to be a
practical upstream/downstream representation.

## Current Stage Result

The current best M4Human reconstruction path is:

```text
local branch: local-only VQ, 512-code codebook
root branch:  full-scratch R3 no-skip bottleneck TCN
```

Artifacts:

- Local VQ:
  `/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_scratch_full_v1/checkpoints/best.pt`
- Root R3:
  `/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_branch_m4human_scratch_full_r3_v1/checkpoints/best.pt`
- Reproduction script:
  `scripts/train_factorized_scratch_m4human.sh`

M4Human test196:

```text
MPJPE / root-align / gap: 54.696 / 49.605 / 5.091 mm
root_xz_mean_error:       2.908 mm
final_xz_error:           4.347 mm
speed_bias:              -0.039 mm/s
```

This proves the root/local split and trajectory losses are effective. It does
not prove that the current root latent is a useful compact token representation.

## Size Accounting

For a 196-frame clip:

| component | representation | size |
| --- | --- | ---: |
| local VQ | 512-code discrete tokens | 49 tokens |
| R3 root latent | continuous tensor | 98 x 256 floats |
| R3 root controls input | yaw vel, local vx/vz, height | 196 x 4 floats |

The R3 root latent has:

```text
98 * 256 = 25,088 continuous values
```

The original root-control sequence has:

```text
196 * 4 = 784 continuous values
```

So R3 expands the root controls by:

```text
25,088 / 784 = 32x
```

This is the central issue. R3 is a strong no-skip autoencoder, but it is still
overcomplete relative to the root signal. It is not a compact root tokenizer.

## What R3 Actually Proves

R3 proves:

- The M4Human root drift is solvable when root trajectory is separated from
  local body pose.
- Multiscale displacement loss is aligned with the root error mode.
- A TCN decoder can reconstruct global root trajectory accurately without U-Net
  skip connections.
- The remaining MPJPE is dominated by local pose reconstruction, not root drift.

R3 does not prove:

- that a downstream model can predict the root representation easily;
- that the root representation is compressed;
- that the root branch is suitable for language-model-style tokenization;
- that root latent can be stored or generated efficiently.

The phrase "no-skip bottleneck" only means there is no high-resolution root skip
connection. It does not imply the latent is smaller than the input.

## Practical Target

The next root representation should be much smaller than `98 x 256`.

Reasonable targets for a 196-frame clip:

| target type | size target | comment |
| --- | ---: | --- |
| compact continuous root latent | 16-32 steps x 16-64 dims | first target |
| spline/control-point root | 8-16 knots x 4-8 dims | physically interpretable |
| discrete root tokens | 8-32 tokens | only after continuous compression works |

A good near-term goal is:

```text
root representation <= 512-1024 continuous values per 196 frames
test196 MPJPE <= 60 mm
test196 root gap <= 10 mm
```

This would preserve most of R3's quality while making the root representation
small enough for upstream models or downstream applications.

## Next Experiments

### C1: Latent Width / Rate Compression Sweep

Keep the R3 architecture and losses, but reduce the latent size.

Suggested sweep:

| experiment | root downsample | latent steps @196 | latent width | total continuous values |
| --- | ---: | ---: | ---: | ---: |
| C1a | 4x | 49 | 64 | 3,136 |
| C1b | 4x | 49 | 32 | 1,568 |
| C1c | 4x | 49 | 16 | 784 |
| C1d | 8x | 25 | 64 | 1,600 |
| C1e | 8x | 25 | 32 | 800 |
| C1f | 8x | 25 | 16 | 400 |
| C1g | 16x | 13 | 32 | 416 |
| C1h | 16x | 13 | 16 | 208 |

Use the same M4Human-only recipe:

```text
local VQ:        use current full-scratch local VQ checkpoint
root branch:     train from scratch
loss:            R3 losses, including multiscale displacement
eval:            test64/test128/test196/val196
selection:       root gap and full MPJPE, not train loss only
```

This is the fastest way to identify the quality/compression frontier.

### C2: Bottleneck Regularization

If C1 degrades too sharply, add regularization to make the latent learn a
smoother low-frequency code:

```text
latent L2 penalty
latent temporal smoothness
latent dropout
latent noise injection
```

The goal is to prevent the root branch from storing frame-level details in an
overly rich continuous tensor.

### C3: Spline / Control-Point Root Branch

Instead of learning an unconstrained latent tensor, predict low-frequency root
trajectory controls:

```text
root_xy control points
root_yaw control points
root_height control points
optional residual velocity
```

For example:

```text
16 knots * (x, z, yaw, height) = 64 continuous values
```

This is much smaller and more physically meaningful than `98 x 256`. It may
also be easier for a sensor or language-conditioned upstream network to learn.

### C4: Root Discretization

Only after compact continuous root works should we discretize root.

Possible forms:

```text
VQ over compact latent sequence
RVQ over root velocity residuals
discrete spline/control-point tokens
```

Do not discretize the current R3 `98 x 256` latent. That would tokenize an
already oversized representation and likely make the downstream problem harder.

## Recommended Immediate Plan

The next concrete step should be C1, a compression sweep. The most informative
first three runs are:

```text
C1b: 4x downsample, width 32  -> 1,568 values
C1e: 8x downsample, width 32  ->   800 values
C1h: 16x downsample, width 16 ->   208 values
```

Interpretation:

- If C1e stays near `<=60 mm` full MPJPE and `<=10 mm` gap, root compression is
  mostly solved.
- If only C1b works, the representation is compressible but still moderately
  large.
- If all compact variants fail, the next direction should be spline/control
  points rather than making the unconstrained latent wider again.

## Current Decision

Use R3 as the quality reference, not as the final representation.

The next objective is to find the smallest root representation that keeps most
of R3's reconstruction quality.
