# mmWave Merge Plan

The next stage should happen after this MotionGPT fork is merged with the
mmWave repository. This file defines the current tokenizer interface and the
minimal integration checklist so the merge does not re-open solved MotionGPT
questions.

## Integration Goal

The mmWave side should predict a factorized motion representation:

```text
mmWave features
  -> local body VQ tokens
  -> Root-FAST RVQ root tokens
  -> MotionGPT/M4Human factorized decoder
  -> recovered global motion
```

The current goal is not to train another single-stream 263-D VQ-VAE. The root
trajectory and local articulated pose should remain separate.

## Token Interface

### Local Body Tokens

Recommended checkpoint:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_v1/checkpoints/best.pt
```

Token properties:

```text
vocab size:     512
window:         196 frames
tokens/window:  ~= 47 on current eval windows
input feature:  130-D local body feature
```

The local branch excludes root yaw velocity, root x/z velocity, and root height.

### Root Tokens: High-Quality Setting

Recommended when reconstruction quality is the first priority:

```text
quantizer:
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_rvq_v1/chunk16_k2/quantizers/rvq_chunk16_k2_vocab1024_d8.npz

chunk size:        16 frames
DCT coeffs:        K=2 per root-control dimension
RVQ vocab:         1024 per stage
RVQ depth:         8
chunks/window:     13
root tokens/window: 104
```

Current full reconstruction on M4Human test196:

```text
MPJPE / root-align / gap = 52.790 / 52.139 / 0.651 mm
total tokens/window      ~= 151
```

### Root Tokens: Compact Balanced Setting

Recommended when token budget is more important:

```text
quantizer:
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_rvq_v1/chunk32_k2/quantizers/rvq_chunk32_k2_vocab512_d8.npz

chunk size:        32 frames
DCT coeffs:        K=2 per root-control dimension
RVQ vocab:         512 per stage
RVQ depth:         8
chunks/window:     7
root tokens/window: 56
```

Current full reconstruction on M4Human test196:

```text
MPJPE / root-align / gap = 58.534 / 53.793 / 4.742 mm
total tokens/window      ~= 103
```

## Root Command Convention

Root-FAST operates on local root commands:

```text
u[t] = [yaw_rate, local_vx, local_vz, root_height]
```

Do not tokenize absolute world `root_xy` directly. The local-command convention
is more stable and matches the root oracle diagnosis.

For a 196-frame clip:

```text
root_controls -> chunked DCT -> keep K coeffs -> RVQ codes
RVQ codes shape should be preserved as [num_chunks, rvq_depth]
```

For sequence models, flattening can be done later, but the merge should keep the
2-D structure in saved datasets so we can choose serialization explicitly:

```text
chunk-major: [chunk0_stage0, chunk0_stage1, ..., chunk1_stage0, ...]
stage-major: [stage0_chunk0, stage0_chunk1, ..., stage1_chunk0, ...]
```

The default recommendation is chunk-major order because each chunk corresponds
to one local time span.

## Dataset Contract

For each 196-frame motion window, the merged dataset should be able to store:

```text
sample_id
source_sequence
start_frame
end_frame
fps = 20
axis_mode = xz-y
features_263
local_features_130
root_controls_4d
local_tokens
root_rvq_codes
optional mmWave features/tokens
```

The exact mmWave feature names can remain repository-specific, but the motion
side should keep these fields stable.

## Minimum Merge Checklist

1. Import or vendor the factorized tokenizer code:

   ```text
   src/motiongpt_m4human/factorized
   ```

2. Keep the current M4Human preprocessing convention:

   ```text
   axis_mode=xz-y
   fps=20 Hz
   window=196 frames
   ```

3. Add a dataset export path that writes local VQ tokens and root RVQ codes next
   to each mmWave sample.

4. Add a reconstruction sanity check:

   ```text
   stored local tokens + stored root tokens -> recovered motion
   ```

   This should match the current MotionGPT-side full eval within numerical
   tolerance before any mmWave predictor is trained.

5. Train proxy predictors before full mmWave training:

   ```text
   local continuous features -> local VQ tokens
   local continuous features -> Root-FAST RVQ tokens
   optional: local VQ tokens -> Root-FAST RVQ tokens
   ```

6. Then train the real mmWave predictor:

   ```text
   mmWave features/tokens -> local VQ tokens + Root-FAST RVQ tokens
   ```

## Metrics to Keep

Do not only track token accuracy. Always reconstruct motion and report:

```text
full MPJPE
root-aligned MPJPE
gap = full - root-aligned
root_xz_mean_error
final_xz_error
path_error
speed_bias
local token accuracy/perplexity
root token accuracy per RVQ stage
```

The key failure mode to watch is root drift returning after token prediction.

## First Recommended mmWave Experiments

1. High-quality tokenizer oracle:

   Use GT local tokens and GT high104 root tokens to verify the merged decode
   path matches:

   ```text
   M4Human test196 ~= 52.79 / 52.14 / 0.65 mm
   ```

2. Token predictor proxy:

   Predict high104 root tokens from non-mmWave motion-derived features. This
   isolates classifier difficulty from sensor difficulty.

3. Compact-vs-quality predictor comparison:

   Compare balanced56 and high104 under the same predictor. The codec-only
   result favors high104, but the predictor may learn balanced56 more easily.

4. Local tokenizer upgrade:

   If root prediction is acceptable, focus on local body tokenization. At
   high104, root drift is already below 1 mm gap, so local pose is the dominant
   reconstruction bottleneck.

## Risks

- Root RVQ has multiple stages. Later stages may be harder to predict, but they
  matter for high-quality reconstruction.
- Balanced56 has a much smaller token budget but still has about 4.7 mm full vs
  root-aligned gap in codec-only eval.
- The current results are M4Human-focused. A mixed M4Human/HumanML3D factorized
  cache and eval should be built only if the next application needs that domain
  coverage.
- The mmWave sensor may not observe fine local pose well enough for the current
  local VQ bottleneck. If that happens, improve local body tokenization or use a
  coarser local target instead of changing the root tokenizer first.
