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
- FAST-like DCT root codec:
  `src/motiongpt_m4human/factorized/root_fast_codec.py`

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

## Root-FAST Continuous DCT Codec

Following the FAST-like idea, we implemented a no-training root command codec:

```text
root command u[t] = [yaw_rate, local_vx, local_vz, root_height]
u[t] -> chunked DCT -> keep K low-frequency coefficients -> inverse DCT
```

This is intentionally evaluated before any learned quantization. The purpose is
to answer whether root commands are compressible in a physically meaningful
frequency-domain representation.

Important detail: this eval uses GT local pose and only replaces root commands
with DCT-reconstructed root commands. Therefore the MPJPE values below are
root-codec error, not full local-token reconstruction error.

Output paths:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_dct_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_dct_v1_val
```

### Test196 Sweep

| chunk | K | values | raw/root compression | MPJPE | root-align | root xz mean | final xz | path error | speed bias |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2 | 56 | 14.00x | 18.056 mm | 9.613 mm | 17.609 mm | 11.428 mm | -0.1760 m | -18.828 mm/s |
| 16 | 2 | 104 | 7.54x | 5.020 mm | 4.944 mm | 4.795 mm | 3.641 mm | -0.0575 m | -6.153 mm/s |
| 32 | 4 | 112 | 7.00x | 4.164 mm | 4.308 mm | 3.961 mm | 2.943 mm | -0.0408 m | -4.366 mm/s |
| 32 | 6 | 168 | 4.67x | 1.863 mm | 2.322 mm | 1.738 mm | 1.156 mm | -0.0260 m | -2.781 mm/s |
| 8 | 2 | 200 | 3.92x | 1.375 mm | 2.068 mm | 1.266 mm | 0.762 mm | -0.0204 m | -2.185 mm/s |
| 16 | 4 | 208 | 3.77x | 1.204 mm | 1.518 mm | 1.117 mm | 0.418 mm | -0.0179 m | -1.911 mm/s |
| 32 | 8 | 224 | 3.50x | 1.113 mm | 1.127 mm | 1.041 mm | 0.295 mm | -0.0165 m | -1.765 mm/s |
| 16 | 6 | 312 | 2.51x | 0.620 mm | 0.659 mm | 0.575 mm | 0.390 mm | -0.0094 m | -1.011 mm/s |
| 8 | 4 | 400 | 1.96x | 0.401 mm | 0.522 mm | 0.367 mm | 0.235 mm | -0.0055 m | -0.588 mm/s |
| 16 | 8 | 416 | 1.88x | 0.442 mm | 0.451 mm | 0.410 mm | 0.214 mm | -0.0063 m | -0.670 mm/s |
| 8 | 6 | 600 | 1.31x | 0.308 mm | 0.294 mm | 0.291 mm | 0.281 mm | -0.0038 m | -0.406 mm/s |
| 8 | 8 | 800 | 0.98x | ~0 mm | ~0 mm | ~0 mm | ~0 mm | ~0 m | ~0 mm/s |

### Val196 Check

| chunk | K | values | raw/root compression | MPJPE | root-align | root xz mean | final xz | path error | speed bias |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2 | 104 | 7.54x | 4.359 mm | 4.286 mm | 4.170 mm | 2.620 mm | -0.0599 m | -6.224 mm/s |
| 16 | 4 | 208 | 3.77x | 0.966 mm | 1.330 mm | 0.892 mm | 0.269 mm | -0.0143 m | -1.489 mm/s |
| 32 | 2 | 56 | 14.00x | 16.001 mm | 9.408 mm | 15.612 mm | 8.248 mm | -0.1807 m | -18.774 mm/s |
| 32 | 4 | 112 | 7.00x | 3.522 mm | 3.397 mm | 3.368 mm | 2.248 mm | -0.0418 m | -4.345 mm/s |

### Interpretation

This is a very strong positive result.

- Root commands are highly compressible with a simple deterministic DCT codec.
- `chunk=16, K=4` uses only `208` continuous values for a 196-frame clip and
  gets `1.20 mm` test MPJPE / `1.12 mm` root xz mean error.
- `chunk=16, K=2` uses only `104` values and still keeps test MPJPE around
  `5.02 mm`, although it has noticeable path/speed underestimation.
- Even `chunk=32, K=4` uses only `112` values and stays around `4.16 mm` test
  MPJPE.

Compared with R3:

```text
R3 root latent:        25,088 continuous values
Root-FAST 16x4 coeffs:    208 continuous values
Root-FAST 16x2 coeffs:    104 continuous values
```

This means the compact root representation problem is likely much easier than
the R3 latent suggested. R3 should remain the neural quality reference, but the
Root-FAST DCT coefficients are a better interface for upstream prediction or
future discretization.

## Recommended Immediate Plan

The original C1 learned-latent compression sweep is now lower priority because
the no-training DCT codec is already far more compact than R3. The next concrete
step should be DCT coefficient quantization.

Recommended first quantization baselines:

```text
Q1: chunk=16, K=4, scalar quantization per coefficient dimension
Q2: chunk=16, K=4, k-means vector quantization over flattened 16-D coeffs
Q3: chunk=32, K=4, k-means vector quantization over flattened 16-D coeffs
```

Interpretation:

- If Q2 works, a 196-frame clip can be represented by about `13` root action
  tokens.
- If Q3 works, a 196-frame clip can be represented by about `7` root action
  tokens.
- If scalar quantization works, we can keep the representation continuous-ish
  but compact and easy for regressors/projectors.

## Current Decision

Use R3 as the neural quality reference, not as the final representation.

The next objective is to turn Root-FAST DCT coefficients into discrete root
action tokens.

## Root-FAST Quantization Status

We added a Root-FAST coefficient quantization evaluator:

```text
src/motiongpt_m4human/factorized/root_fast_quantize.py
```

It supports three quantization modes:

```text
vector:  one k-means token for the full flattened DCT chunk
product: one k-means token per root-command dimension per chunk
scalar:  uniform scalar quantization per DCT coefficient
```

Artifacts:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_quantized_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_scalar_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_product_v1
```

Important caveat: the first full vector sweep in `root_fast_quantized_v1` used
an early-stop bug in k-means and should not be treated as canonical. The bug is
fixed in code. A small post-fix check still showed that full-chunk vector VQ is
weak, but the full fixed vector sweep should be rerun when compute is available.

### Product VQ Test196

Product VQ quantizes each DCT chunk as four tokens, one token for each root
command dimension. For 196 frames:

```text
chunk=16 -> 13 chunks -> 52 root tokens
chunk=32 ->  7 chunks -> 28 root tokens
chunk=64 ->  4 chunks -> 16 root tokens
```

Best product-VQ results from the completed sweep:

| root tokens | chunk | K | vocab/group | bits/window | MPJPE | root xz mean | final xz |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 2 | 256 | 128 | 82.61 mm | 80.09 mm | 106.65 mm |
| 28 | 32 | 2 | 256 | 224 | 53.67 mm | 51.57 mm | 80.14 mm |
| 52 | 16 | 2 | 256 | 416 | 41.20 mm | 39.68 mm | 63.61 mm |

This is much better than one full-chunk VQ token, but still much worse than
continuous DCT coefficients. It suggests that the DCT coefficients are compact,
but a single-level product VQ is still too coarse.

### Scalar Quantization Test196

Scalar quantization is not a clean motion-token interface, but it is a useful
upper-bound check for whether DCT coefficients are discretizable.

| config | scalar codes | bits/window | MPJPE | root xz mean | final xz |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk=16, K=2, 6-bit | 104 | 624 | 27.97 mm | 26.94 mm | 43.06 mm |
| chunk=16, K=2, 8-bit | 104 | 832 | 9.94 mm | 9.48 mm | 11.98 mm |
| chunk=32, K=4, 6-bit | 112 | 672 | 27.88 mm | 26.87 mm | 42.59 mm |
| chunk=32, K=4, 8-bit | 112 | 896 | 9.36 mm | 8.91 mm | 12.32 mm |
| chunk=64, K=4, 8-bit | 64 | 512 | 19.08 mm | 18.53 mm | 18.64 mm |

This is the strongest signal so far: coefficient discretization itself is not
the problem. The hard part is representing several continuous coefficients with
a small number of language-model-style tokens.

### Interpretation

Root-FAST should remain the leading compact root representation, but the first
discrete version should not be plain one-token-per-chunk vector VQ.

The current ranking is:

```text
continuous DCT coefficients: excellent quality, compact continuous interface
8-bit scalar quantization:   good quality, too many scalar codes
product VQ:                  token-like, moderate quality
plain vector VQ:             too lossy at vocab <= 1024
```

The next promising tokenizer is residual vector quantization over DCT chunks:

```text
chunk coefficients -> RVQ code_1, code_2, ..., code_R
```

For `chunk=16`, this would use:

```text
13 chunks * R residual tokens
```

For example, `R=4` gives `52` root tokens, matching the product-VQ token count
but preserving cross-dimension structure better.
