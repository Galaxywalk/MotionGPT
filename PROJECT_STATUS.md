# Project Status: M4Human Tokenizer Handoff

This repository is currently in a good handoff state for the M4Human motion
tokenizer work. The next major step depends on merging with the mmWave
repository, so this document records the stable conclusions, recommended
artifacts, and what should not be re-opened unless new evidence appears.

## Stage Conclusion

The main conclusion is structural:

```text
M4Human motion should not be represented by one single 263-D VQ-VAE stream.
Use separate representations for local body pose and global root trajectory.
```

The single-stream MotionGPT-style VQ-VAE reconstructed local pose reasonably
well, but accumulated root trajectory drift over long windows:

```text
single-stream Exp3, M4Human test196:
MPJPE / root-align / gap = 101.077 / 52.429 / 48.648 mm
```

The root oracle decomposition showed the dominant drift source was local root
velocity, not a remaining coordinate-system bug:

```text
pred yaw + pred vel: 101.077 / 52.429 / 48.648 mm
GT yaw   + pred vel:  91.515 / 50.423 / 41.092 mm
pred yaw + GT vel:    70.459 / 52.429 / 18.030 mm
GT yaw   + GT vel:    52.559 / 50.423 /  2.135 mm
```

The factorized design solved this root drift:

```text
local body:      discrete local VQ tokens
root trajectory: separate root representation
```

R3, a no-skip continuous TCN root branch, is the quality reference:

```text
full scratch R3, M4Human test196:
MPJPE / root-align / gap = 54.696 / 49.605 / 5.091 mm
root_xz_mean_error       = 2.908 mm
final_xz_error           = 4.347 mm
speed_bias               = -0.039 mm/s
```

R3 is not the final token interface because its root latent is too large:

```text
98 x 256 = 25,088 continuous values per 196-frame clip
```

The current recommended token-like interface is therefore:

```text
local body:      local VQ tokens
root trajectory: Root-FAST DCT + RVQ tokens
```

## Recommended Current Tokenizers

Use these as the current stage checkpoints for downstream or mmWave integration.

### High-Quality Setting

This is the best current full tokenizer result.

```text
local checkpoint:
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_v1/checkpoints/best.pt

root quantizer:
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_rvq_v1/chunk16_k2/quantizers/rvq_chunk16_k2_vocab1024_d8.npz

root config:
chunk=16, K=2, vocab=1024, RVQ depth=8

tokens per 196-frame window:
local ~= 47, root = 104, total ~= 151
```

M4Human test196:

```text
full MPJPE / root-align / gap = 52.790 / 52.139 / 0.651 mm
local-only MPJPE              = 51.168 mm
root-only MPJPE               = 7.031 mm
```

Interpretation: root drift is almost gone. Remaining full MPJPE is dominated by
local body reconstruction.

### Compact Balanced Setting

Use this when token budget matters more than the last few millimeters.

```text
local checkpoint:
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_v1/checkpoints/best.pt

root quantizer:
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_rvq_v1/chunk32_k2/quantizers/rvq_chunk32_k2_vocab512_d8.npz

root config:
chunk=32, K=2, vocab=512, RVQ depth=8

tokens per 196-frame window:
local ~= 47, root = 56, total ~= 103
```

M4Human test196:

```text
full MPJPE / root-align / gap = 58.534 / 53.793 / 4.742 mm
local-only MPJPE              = 51.168 mm
root-only MPJPE               = 20.146 mm
```

Interpretation: this is the best compact operating point so far. It preserves
most of the quality while reducing the root token count by about half relative
to the high-quality setting.

## Data and Feature Convention

The validated M4Human cache convention is:

```text
axis_mode = xz-y
fps       = 20 Hz
window    = 196 frames for main eval
```

Current factorized cache:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_cache/v1_m4human_xz-y_20hz
```

Root controls are:

```text
[yaw_rate, local_root_velocity_x, local_root_velocity_z, root_height]
```

Local body features are root-relative body joints, root-frame body joint
velocities, and foot contacts. Root yaw velocity, root x/z velocity, and root
height are intentionally excluded from local VQ input.

## What Is Settled

- `axis_mode=xz-y` should remain the default for M4Human.
- M4Human should be resampled/aligned to the current 20 Hz feature convention.
- Root/local factorization is necessary for long-window reconstruction.
- R3 is useful as a quality reference, but not as the final compact token
  representation.
- Root-FAST continuous DCT is a strong physical root representation.
- Plain one-token-per-chunk vector VQ is too lossy for root DCT coefficients.
- Root-FAST RVQ is the strongest root tokenization path so far.
- At the high104 root setting, root drift is no longer the main bottleneck.

## What Remains Open

- The local body tokenizer is now the reconstruction bottleneck.
- HumanML3D has not yet been rebuilt as a matching factorized cache in the final
  Root-FAST pipeline.
- The mmWave repository still needs a predictor interface for local VQ tokens
  and Root-FAST RVQ tokens.
- Token ordering and serialization should be finalized during the mmWave merge.
- Downstream prediction quality should be tested separately from codec
  reconstruction quality.

## Do Not Reprioritize Without New Evidence

- More training of the original single-stream 263-D VQ-VAE.
- More scalar root speed/path losses on the single-stream tokenizer.
- Global velocity scale calibration heads.
- Head-only root correction after decoded 263-D features.
- Plain full-chunk vector VQ for Root-FAST coefficients.
- Discretizing the oversized R3 `98 x 256` root latent.

## Documentation Map

- `FACTORIZED_TOKENIZER.md`: model structure and training details.
- `ROOT_LATENT_COMPRESSION.md`: why R3 is overcomplete and how Root-FAST was
  introduced.
- `ROOT_FAST_TOKENIZER_TODO.md`: Root-FAST quantization frontier and next
  compute tasks.
- `M4HUMAN_EVAL_20260530.md`: chronological experiment log.
- `EXPERIMENT_INDEX.md`: artifact index and summary JSON locations.
- `MMWAVE_MERGE_PLAN.md`: integration plan for the future mmWave repository.
- `COLD_START.md`: original reproduction and data setup checklist.
