# M4Human MotionGPT Evaluation 2026-05-30

This records the checkpoints and evaluation outputs from the first M4Human
mixed-training pass.

## Checkpoints

- Mixed multi-length finetune:
  `experiments/mgpt/VQVAE_HumanML3D_M4Human20Hz_mix70_lr2e5_multilen_finetune/checkpoints/epoch=499.ckpt`
- M4Human-only cold start:
  `experiments/mgpt/VQVAE_M4Human20Hz_only_coldstart/checkpoints/epoch=499.ckpt`
- Prior fixed-64 baseline:
  `experiments/mgpt/VQVAE_HumanML3D_M4Human20Hz_mix70_lr2e5_finetune/checkpoints/epoch=99.ckpt`

The checkpoint files are local artifacts and are not committed.

## Output Paths

- Mixed multi-length eval:
  `/cpfs01/liangbo/data/MotionGPT/length_drift_analysis/mix70_multilen_epoch499`
- M4Human-only eval:
  `/cpfs01/liangbo/data/MotionGPT/length_drift_analysis/m4human_only_coldstart_epoch499`
- Summary CSV:
  `/cpfs01/liangbo/data/MotionGPT/length_drift_analysis/eval_comparison_20260530.csv`

## Summary

| experiment | M4Human test 196 MPJPE / root-align | M4Human all 196 MPJPE / root-align | M4Human test 196 speed bias | HumanML3D official FID / MPJPE |
| --- | ---: | ---: | ---: | ---: |
| mixed multilen epoch499 | 101.449 / 52.347 mm | 70.527 / 32.295 mm | -11.891 mm/s | 0.194127 / 48.543 mm |
| M4Human-only coldstart epoch499 | 191.659 / 75.531 mm | 178.870 / 67.269 mm | -6.093 mm/s | 1.731401 / 117.442 mm |
| old fixed64 mix70 epoch99 | 139.995 / 62.789 mm | 119.905 / 48.034 mm | -17.039 mm/s | not rerun |

## Interpretation

- Multi-length training reduced long-window M4Human drift substantially:
  M4Human test 196 MPJPE improved from 139.995 mm to 101.449 mm, and the
  full/root-aligned gap improved from 77.205 mm to 49.103 mm.
- M4Human-only cold start was not competitive. It degraded both M4Human
  reconstruction and HumanML3D official metrics, which suggests the original
  HumanML3D tokenizer/checkpoint still provides important codebook and motion
  priors.
- The remaining error is still dominated by recovered root trajectory. On
  M4Human test 196, root-aligned MPJPE is 52.347 mm while full MPJPE is
  101.449 mm.

## Next Experiments

These were run after the mixed multi-length baseline above.

## Follow-up Experiments

### Checkpoints

- Exp3 path/final loss:
  `experiments/mgpt/VQVAE_HumanML3D_M4Human20Hz_mix70_lr2e5_multilen_path_finetune/checkpoints/epoch=649.ckpt`
- Exp4 path + M4Human root velocity calibration:
  `experiments/mgpt/VQVAE_HumanML3D_M4Human20Hz_mix70_lr2e5_multilen_path_calib_finetune/checkpoints/epoch=49.ckpt`
- Exp5 path + M4Human speed mean loss:
  `experiments/mgpt/VQVAE_HumanML3D_M4Human20Hz_mix70_lr2e5_multilen_path_speed_finetune/checkpoints/epoch=99.ckpt`

### Output Paths

- Exp3 path/final eval:
  `/cpfs01/liangbo/data/MotionGPT/length_drift_analysis/mix70_multilen_path_epoch649`
- Exp4 calibration eval:
  `/cpfs01/liangbo/data/MotionGPT/length_drift_analysis/mix70_multilen_path_calib_epoch49`
- Exp5 speed loss eval:
  `/cpfs01/liangbo/data/MotionGPT/length_drift_analysis/mix70_multilen_path_speed_epoch99`

### Results

| experiment | M4Human test 196 MPJPE / root-align / gap | speed bias | path error | final xz error | HumanML3D official FID / MPJPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exp2 mixed multilen | 101.449 / 52.347 / 49.103 mm | -11.891 mm/s | -0.1108 m | 117.543 mm | 0.194127 / 48.543 mm |
| Exp3 path/final | 101.077 / 52.429 / 48.648 mm | -7.257 mm/s | -0.0673 m | 118.717 mm | 0.182322 / 48.643 mm |
| Exp4 calibration | 102.364 / 52.507 / 49.858 mm | -7.018 mm/s | -0.0650 m | 121.306 mm | 0.167669 / 49.010 mm |
| Exp5 speed mean | 102.816 / 52.184 / 50.632 mm | -5.226 mm/s | -0.0484 m | 120.927 mm | 0.214517 / 48.422 mm |

### Interpretation

- Exp3 improved the root speed/path underestimation without hurting HumanML3D,
  but full MPJPE barely moved. This means path length alone was not the main
  source of the remaining 196-frame drift; endpoint direction/phase still
  matters.
- Exp4 was a negative result. The learned M4Human root velocity scale ended at
  `[0.9993, 1.0008]`, effectively no calibration. Under the current losses, a
  global x/z velocity scale does not receive a stable signal to increase speed.
- Exp5 hit the speed-bias target (`-5.226 mm/s`) and further reduced path
  underestimation, but it worsened M4Human full MPJPE and HumanML3D FID. This
  confirms the model can be pushed to match M4Human speed statistics, but speed
  matching alone can increase endpoint/root-position error.

Recommended next direction: keep Exp3 as the best balanced checkpoint for now.
If continuing, try a smaller M4Human speed weight (`0.25` or `0.5`) and a more
direction-aware endpoint/displacement objective, rather than further increasing
path length or global speed alone.

## Exp8/Exp9 Root Step Loss

This pass adds M4Human-domain-only trajectory losses and a direction-sensitive
root step loss. The new config entry `LOSS.TRAJ_DOMAIN` gates
`LAMBDA_TRAJ_FINAL`, `LAMBDA_TRAJ_PATH`, and `LAMBDA_TRAJ_STEP` by domain.
`LOSS.LAMBDA_TRAJ_STEP` computes an L1 loss between recovered global root x/z
step vectors.

### Checkpoints

- Exp8a M4Human-domain step loss 0.005:
  `experiments/mgpt/VQVAE_HumanML3D_M4Human20Hz_mix70_lr2e5_multilen_path_step005_finetune/checkpoints/epoch=99.ckpt`
- Exp8b step loss 0.01 config exists, but was not run because Exp8a was only a
  weak positive result:
  `configs/config_h3d_m4human_mix70_multilen_path_step010_stage1.yaml`

### Output Paths

- Exp8a eval:
  `/cpfs01/liangbo/data/MotionGPT/length_drift_analysis/mix70_multilen_path_step005_epoch99`
- Exp3 root analysis rerun with the new step-vector metric:
  `/cpfs01/liangbo/data/MotionGPT/length_drift_analysis/mix70_multilen_path_epoch649/root_analysis_with_step`

### Results

| experiment | M4Human test 196 MPJPE / root-align / gap | speed bias | path error | final xz error | step L1 | HumanML3D official FID / MPJPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp3 path/final | 101.077 / 52.429 / 48.648 mm | -7.257 mm/s | -0.0673 m | 118.717 mm | 1.865 mm | 0.182322 / 48.643 mm |
| Exp8a M4-only final/path/step | 100.858 / 52.303 / 48.555 mm | -7.982 mm/s | -0.0742 m | 118.300 mm | 1.848 mm | 0.194567 / 48.481 mm |

Exp8a M4Human length breakdown:

| split/window | MPJPE / root-align / gap |
| --- | ---: |
| test 64 | 71.943 / 50.936 / 21.006 mm |
| test 128 | 86.396 / 51.476 / 34.920 mm |
| test 196 | 100.858 / 52.303 / 48.555 mm |
| all 196 | 68.354 / 31.806 / 36.548 mm |

### Interpretation

- Exp8a barely improves full MPJPE and gap over Exp3. The new step-vector metric
  also moves only slightly, from 1.865 mm to 1.848 mm.
- HumanML3D official FID remains acceptable at 0.195, but is worse than Exp3's
  0.182. The M4Human-only gating worked in the sense that it avoided a large
  HumanML3D regression, but the trajectory objective was too weak to materially
  fix root drift.
- Because Exp8a did not show a clear gain, Exp8b (`LAMBDA_TRAJ_STEP=0.01`) was
  not run. Increasing this weight is more likely to trade off feature balance
  than to solve the remaining drift.

### Bug Checks

- `python -m py_compile mGPT/losses/mgpt.py src/motiongpt_m4human/analyze_root_reconstruction.py`
- Synthetic loss checks for mixed, HumanML3D-only, and missing-domain batches all
  produced finite losses.
- Real dataloader smoke checks confirmed M4Human batches log
  `recons_trajstep`, while HumanML3D-only batches skip it without NaNs.

## Root Yaw / Local Velocity Oracle

The diagnostic script `src/motiongpt_m4human/eval_root_oracle.py` decomposes
root drift by replacing only the decoded root yaw velocity feature
(`feature[..., 0]`) and/or decoded local x/z velocity features
(`feature[..., 1:3]`) with ground truth before recovering joints.

### Output Paths

- Exp3:
  `/cpfs01/liangbo/data/MotionGPT/length_drift_analysis/root_oracle/exp3_epoch649_m4human_test196.json`
- Exp8a:
  `/cpfs01/liangbo/data/MotionGPT/length_drift_analysis/root_oracle/exp8a_step005_epoch99_m4human_test196.json`

### Exp3 Results

| case | MPJPE / root-align / gap | root xz mean | final xz | path error | speed bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| Case 0 pred yaw + pred vel | 101.077 / 52.429 / 48.648 mm | 76.054 mm | 118.717 mm | -0.0673 m | -7.202 mm/s |
| Case 1 GT yaw + pred vel | 91.515 / 50.423 / 41.092 mm | 67.053 mm | 102.709 mm | -0.0673 m | -7.202 mm/s |
| Case 2 pred yaw + GT vel | 70.459 / 52.429 / 18.030 mm | 26.478 mm | 43.944 mm | ~0 m | ~0 mm/s |
| Case 3 GT yaw + GT vel | 52.559 / 50.423 / 2.135 mm | 0.000 mm | 0.000 mm | 0 m | 0 mm/s |

### Exp8a Results

| case | MPJPE / root-align / gap | root xz mean | final xz | path error | speed bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| Case 0 pred yaw + pred vel | 100.858 / 52.303 / 48.555 mm | 76.068 mm | 118.300 mm | -0.0742 m | -7.940 mm/s |
| Case 1 GT yaw + pred vel | 90.601 / 50.348 / 40.253 mm | 66.079 mm | 101.869 mm | -0.0742 m | -7.940 mm/s |
| Case 2 pred yaw + GT vel | 70.938 / 52.303 / 18.635 mm | 27.123 mm | 44.164 mm | ~0 m | ~0 mm/s |
| Case 3 GT yaw + GT vel | 52.671 / 50.348 / 2.323 mm | 0.000 mm | 0.000 mm | 0 m | 0 mm/s |

### Interpretation

- Local x/z velocity is the dominant source of the remaining root drift. On
  Exp3, replacing local velocity with GT improves MPJPE by 30.6 mm, while
  replacing yaw velocity alone improves MPJPE by 9.6 mm.
- Yaw still matters through coupling: with GT local velocity but predicted yaw,
  root xz mean error remains 26.5 mm and final xz error remains 43.9 mm.
- Case 3 removes almost the entire full/root-aligned gap. This argues against a
  major coordinate-convention or root-recovery bug in the current eval path.
- The next structural direction should focus on the root local velocity branch,
  with a secondary yaw/heading consistency term. More global path-length losses
  are unlikely to solve the main issue.

## Root Correction Head

This implements a minimal structural patch on top of Exp3. The VQ-VAE encoder,
codebook, and decoder are frozen. A small temporal Conv1d head reads the decoded
263-D feature and predicts residual corrections to root yaw velocity and local
x/z velocity. The correction is applied only for `m4human` domain samples.

Config:
`configs/config_h3d_m4human_mix70_multilen_rootcorr_stage1.yaml`

### Checkpoint and Outputs

- Checkpoint:
  `experiments/mgpt/VQVAE_HumanML3D_M4Human20Hz_mix70_lr1e4_multilen_rootcorr_finetune/checkpoints/epoch=49.ckpt`
- Eval directory:
  `/cpfs01/liangbo/data/MotionGPT/length_drift_analysis/rootcorr_epoch49`

### Training Setup

- Init: Exp3 epoch649
- Frozen: VQ encoder, codebook, decoder body
- Trainable: `RootCorrectionHead`, 301K parameters
- Data: HumanML3D + M4Human mix70, multi-length `[64, 128, 196]`
- Domain: correction and root losses apply only to M4Human
- Loss weights:
  `root_vel=0.05`, `root_yaw=0.01`, `final=0.02`, `path=0.005`
- LR: `1e-4`
- Epochs: 50

### Results

| experiment | M4Human test 196 MPJPE / root-align / gap | root xz mean | final xz | speed bias | path error | HumanML3D official FID / MPJPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp3 path/final | 101.077 / 52.429 / 48.648 mm | 76.054 mm | 118.717 mm | -7.202 mm/s | -0.0673 m | 0.182322 / 48.643 mm |
| Root correction head | 100.340 / 52.410 / 47.929 mm | 75.203 mm | 116.994 mm | -8.261 mm/s | -0.0767 m | 0.182322 / 48.643 mm |

Length breakdown:

| split/window | MPJPE / root-align / gap |
| --- | ---: |
| test 64 | 71.761 / 50.759 / 21.002 mm |
| test 128 | 87.337 / 51.532 / 35.805 mm |
| test 196 | 100.340 / 52.410 / 47.929 mm |
| all 196 | 68.615 / 31.938 / 36.677 mm |

Oracle after correction:

| case | MPJPE / root-align / gap | root xz mean | final xz |
| --- | ---: | ---: | ---: |
| Case 0 pred yaw + pred vel | 100.340 / 52.410 / 47.929 mm | 75.203 mm | 116.994 mm |
| Case 1 GT yaw + pred vel | 90.758 / 50.423 / 40.334 mm | 66.129 mm | 101.043 mm |
| Case 2 pred yaw + GT vel | 70.367 / 52.410 / 17.956 mm | 26.393 mm | 43.713 mm |
| Case 3 GT yaw + GT vel | 52.559 / 50.423 / 2.135 mm | 0.000 mm | 0.000 mm |

### Interpretation

- The implementation is safe: HumanML3D official metrics are unchanged from
  Exp3 because correction is gated to M4Human.
- The head-only correction is only a weak positive result. M4Human test-196
  improves by 0.74 mm full MPJPE and 0.72 mm gap, far below the 80-90 mm target.
- The correction did not fix the velocity bottleneck: speed bias worsened from
  -7.20 to -8.26 mm/s, and the oracle still shows the same dominant local
  velocity gap.
- The most likely reason is that a zero-init head trained with losses measured
  in meters receives a weak correction signal relative to the frozen decoder's
  reconstruction balance. This version validates the wiring and domain gating,
  but not the efficacy of the head.

Next options are either to increase the root-velocity objective in physical
units, for example by training on mm/s-scaled velocity loss, or to unfreeze the
decoder tail/root-specific layers so the correction is not fighting a frozen
decoder output alone.

## Root Correction Tail + Physical Velocity Loss

This pass strengthens the previous correction-head experiment in two ways:

- The root correction head can read `decoded`, `decoder_hidden`, and
  upsampled `quantized` latent features instead of only the decoded 263-D
  feature.
- The VQ-VAE decoder tail is partially unfrozen while the encoder and codebook
  stay frozen. A gradient mask keeps the final decoder Conv1d update limited to
  the first three root channels.

The root velocity and yaw losses can now be computed in physical units with
`LOSS.ROOT_VEL_UNIT=mmps` and `LOSS.ROOT_YAW_UNIT=degps`. The model also logs
gradient norms for the correction head and decoder tail.

Config:
`configs/config_h3d_m4human_mix70_multilen_rootcorr_tail_phys_stage1.yaml`

### Checkpoint and Outputs

- Checkpoint:
  `experiments/mgpt/VQVAE_HumanML3D_M4Human20Hz_mix70_lr1e4_multilen_rootcorr_tail_phys/checkpoints/epoch=49.ckpt`
- Eval directory:
  `/cpfs01/liangbo/data/MotionGPT/length_drift_analysis/rootcorr_tail_phys_epoch49`

### Training Setup

- Init: Exp3 epoch649
- Frozen: encoder, codebook, decoder body except tail
- Trainable: last three decoder modules plus `RootCorrectionHead`, 2.3M
  parameters
- Correction input: decoded 263-D + decoder hidden 512-D + quantized latent
  512-D
- Data: HumanML3D + M4Human mix70, multi-length `[64, 128, 196]` with weights
  `[0.25, 0.25, 0.50]`
- Domain: correction and root trajectory losses apply only to M4Human
- Loss weights:
  `root_vel=0.001` in mm/s, `root_yaw=0.001` in deg/s,
  `final=0.02`, `path=0.005`
- LR: correction head `1e-4`, decoder tail `1e-5`
- Epochs: 50

Smoke check before training:

- Trainable tensors: final decoder root-channel Conv1d weights/bias, two tail
  decoder conv layers, and correction head parameters.
- One real M4Human batch produced finite loss `0.8098`.
- Raw losses on that batch: `root_vel_phys=100.65 mm/s`,
  `root_yaw=5.58 deg/s`.
- Gradient norms were nonzero: correction head `0.0128`, decoder tail `0.0356`.

### Results

| experiment | M4Human test 196 MPJPE / root-align / gap | root xz mean | final xz | speed bias | path error | HumanML3D official FID / MPJPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp3 path/final | 101.077 / 52.429 / 48.648 mm | 76.054 mm | 118.717 mm | -7.202 mm/s | -0.0673 m | 0.182322 / 48.643 mm |
| Root correction head | 100.340 / 52.410 / 47.929 mm | 75.203 mm | 116.994 mm | -8.261 mm/s | -0.0767 m | 0.182322 / 48.643 mm |
| Tail + physical loss | 99.956 / 52.257 / 47.698 mm | 74.802 mm | 116.003 mm | -10.276 mm/s | -0.0961 m | 0.192576 / 48.487 mm |

Length breakdown:

| split/window | MPJPE / root-align / gap |
| --- | ---: |
| test 64 | 71.627 / 50.626 / 21.001 mm |
| test 128 | 87.062 / 51.387 / 35.675 mm |
| test 196 | 99.956 / 52.257 / 47.698 mm |

Oracle after tail + physical loss:

| case | MPJPE / root-align / gap | root xz mean | final xz | path error | speed bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| Case 0 pred yaw + pred vel | 99.956 / 52.257 / 47.698 mm | 74.802 mm | 116.003 mm | -0.0961 m | -10.276 mm/s |
| Case 1 GT yaw + pred vel | 90.311 / 50.294 / 40.017 mm | 65.625 mm | 99.699 mm | -0.0961 m | -10.276 mm/s |
| Case 2 pred yaw + GT vel | 70.393 / 52.257 / 18.136 mm | 26.566 mm | 43.879 mm | ~0 m | ~0 mm/s |
| Case 3 GT yaw + GT vel | 52.472 / 50.294 / 2.178 mm | 0.000 mm | 0.000 mm | 0 m | 0 mm/s |

### Interpretation

- Unfreezing the decoder tail and giving the correction head richer inputs is a
  weak positive result, but still far from the desired 80-90 mm test196 MPJPE.
  It improves Exp3 by about 1.1 mm full MPJPE and 1.0 mm gap.
- HumanML3D is still protected: FID moves from 0.182 to 0.193 and MPJPE is
  essentially unchanged.
- The local-velocity bottleneck remains. Speed bias worsens to `-10.276 mm/s`
  and path underestimation worsens to `-0.0961 m`, even though endpoint/root
  position improves slightly.
- The physical-unit velocity loss is wired correctly and produces gradients,
  but with the conservative `0.001` weight it does not dominate the decoder's
  existing reconstruction balance. Increasing it may reduce speed bias, but the
  previous speed-loss experiment warns that matching speed alone can worsen
  endpoint error.
- This result strengthens the structural conclusion: a late residual head,
  even with decoder tail adaptation, only has limited ability to recover root
  local velocity from the current VQ latent. A cleaner next direction is a
  dedicated root-trajectory branch or a codebook/objective that preserves root
  local velocity before the VQ bottleneck.

## Root/Local Factorized Tokenizer Phase 1-2

This pass implements the first two concrete steps from
`ROOT_LOCAL_TOKENIZER_TODO.md`:

- `src/motiongpt_m4human/factorized/audit_root_quality.py`
  audits physical root statistics, yaw jitter, root height, foot sliding,
  smoothing impact, and VQ upper-bound decompositions.
- `src/motiongpt_m4human/factorized/cache.py` builds a reusable root/local
  factorized cache.
- `src/motiongpt_m4human/factorized/representation.py`,
  `recover.py`, and `dataset.py` provide conversion, round-trip recovery, and
  multi-length window sampling.

### Output Paths

- Exp3 test-split root quality + upper bounds:
  `/cpfs01/liangbo/data/MotionGPT/factorized_audit/root_quality_exp3_test`
- All-split root quality statistics:
  `/cpfs01/liangbo/data/MotionGPT/factorized_audit/root_quality_all_splits`
- M4Human factorized cache:
  `/cpfs01/liangbo/data/MotionGPT/factorized_cache/v1_m4human_xz-y_20hz`

### Root Quality Findings

Test split:

| source | sequences / windows | root speed p50 / mean | window196 path p50 / mean | yaw vel p95 / p99 | contact ratio | contact slide ratio | smoothing MPJPE mean / p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HumanML3D test | 4384 / 4218 | 0.124 / 0.314 m/s | 1.396 / 2.208 m | 15.74 / 40.05 deg/s | 0.857 | 0.349 | 7.19 / 2.75 mm |
| M4Human test | 238 / 1406 | 0.088 / 0.164 m/s | 1.180 / 1.529 m | 12.21 / 28.43 deg/s | 0.923 | 0.248 | 5.74 / 2.66 mm |

All-split highlights:

| source split | root speed p50 / mean | window196 path p50 / mean | contact slide ratio | smoothing MPJPE mean / p50 |
| --- | ---: | ---: | ---: | ---: |
| HumanML3D train | 0.128 / 0.316 m/s | 1.454 / 2.227 m | 0.353 | 7.42 / 2.70 mm |
| HumanML3D val | 0.127 / 0.313 m/s | 1.357 / 2.241 m | 0.353 | 7.76 / 2.48 mm |
| M4Human train | 0.089 / 0.168 m/s | 1.224 / 1.590 m | 0.253 | 5.94 / 2.27 mm |
| M4Human val | 0.081 / 0.147 m/s | 1.231 / 1.411 m | 0.220 | 5.46 / 2.58 mm |
| M4Human test | 0.088 / 0.164 m/s | 1.180 / 1.529 m | 0.248 | 5.74 / 2.66 mm |

### Exp3 Upper Bounds

| source/case | MPJPE / root-align / gap | final xz | path error |
| --- | ---: | ---: | ---: |
| M4Human normal pred root + pred local | 101.077 / 52.429 / 48.648 mm | 118.717 mm | -0.0673 m |
| M4Human GT root + pred local | 52.075 / 52.075 / ~0 mm | 0.000 mm | 0.0000 m |
| M4Human pred root + GT local | 78.626 / 18.181 / 60.445 mm | 118.717 mm | -0.0673 m |
| M4Human smoothed GT root + pred local | 52.317 / 52.109 / 0.208 mm | 1.594 mm | -0.0143 m |
| M4Human smoothed GT root + GT local | 1.531 / 0.783 / 0.747 mm | 1.594 mm | -0.0143 m |

### Factorized Cache

The M4Human cache contains:

- sequences: `1081`
- frames: `1,317,030`
- disk size: `2.5G`
- round-trip MPJPE mean: `1.46e-5 mm`
- round-trip MPJPE max: `0.00042 mm`

Stored fields include `local_joints`, `local_joint_vel`, `local_rot6d`,
`contacts`, `root_xy`, `root_yaw`, `root_height`, `root_vel_local_mps`,
`root_vel_global_mps`, `root_yaw_vel_radps`, `dt`, `source_domain`, and
`valid_mask`. The cache keeps `axis_mode=xz-y`, `world_up_axis=y`, and ground
axes `x/z` in metadata.

### Interpretation

- The upper-bound result is decisive: with GT root and decoded local pose,
  M4Human test196 is about `52.1 mm`, essentially the current root-aligned
  quality. This means the next tokenizer should prioritize root trajectory
  modeling before spending more effort on local pose.
- M4Human is slower than HumanML3D in physical root statistics. On test,
  median speed is `0.088 m/s` vs HumanML3D `0.124 m/s`, and mean speed is
  `0.164 m/s` vs `0.314 m/s`. The root issue is therefore not simply that
  M4Human moves too fast; it is that the model loses direction/timing in the
  root local velocity channel.
- Smoothing 0.35s root controls changes M4Human only mildly:
  `smoothed GT root + GT local` is `1.53 mm`, and
  `smoothed GT root + decoded local` is `52.32 mm`. This suggests smoothed
  root targets are safe as an optional root-branch target, but smoothing alone
  is not a replacement for root prediction.
- Foot contact labels are high-rate for both datasets and M4Human does not look
  worse than HumanML3D by the simple contact-sliding metric. Contact quality is
  not the main blocker for the first factorized tokenizer.
- The factorized cache round-trip is effectively exact, so it is ready for the
  local-only VQ baseline. The next experiment should train a local-only VQ on
  `local_joints + local_joint_vel + contacts` and verify that it reaches about
  `50-52 mm` root-aligned MPJPE on M4Human test196 before adding the continuous
  root branch.
