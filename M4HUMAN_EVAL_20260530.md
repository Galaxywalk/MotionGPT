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

## Root/Local Factorized Tokenizer Phase 3-4

This pass implements the first usable root/local tokenizer prototype inspired
by RoHM's trajectory/local-pose decomposition. The local branch is discrete and
the root branch is continuous:

```text
local branch: local_joints[21] + local_joint_vel[21] + contacts -> VQ codes
root branch:  yaw_vel + local x/z velocity + root height -> continuous latent
```

The local representation intentionally excludes root yaw, root x/z velocity,
and root height so the local VQ cannot leak global trajectory.

### Code

- Local VQ:
  `src/motiongpt_m4human/factorized/local_vq.py`
- Continuous root branch:
  `src/motiongpt_m4human/factorized/root_branch.py`

### Output Paths

- Local VQ experiment:
  `/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_v1`
- Root branch experiment:
  `/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_branch_m4human_v1`
- No-skip bottleneck root branch experiment:
  `/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_branch_m4human_bottleneck_v1`
- Cache:
  `/cpfs01/liangbo/data/MotionGPT/factorized_cache/v1_m4human_xz-y_20hz`

### Local-Only VQ Setup

- Input dim: `130`
  - root-relative body local joints: `21 * 3`
  - root-frame body local joint velocity: `21 * 3`
  - foot contacts: `4`
- Model: MotionGPT `VQVae` with `code_num=512`, `code_dim=512`,
  `down_t=2`, `stride_t=2`.
- Training: M4Human train split, 100 epochs, 100 steps/epoch,
  batch size 256.
- Window sampling: `[64, 128, 196]` with weights `[0.25, 0.25, 0.50]`.
- Checkpoint:
  `/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_v1/checkpoints/best.pt`

### Local-Only VQ Results

| split/window | MPJPE / root-align | local body MPJPE | local velocity error | contact F1 | unique / effective codes | tokens/window |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test 64 | 52.084 / 52.084 mm | 54.565 mm | 123.179 mm/s | 0.9928 | 512 / 365.6 | 15.92 |
| test 128 | 51.323 / 51.323 mm | 53.767 mm | 120.732 mm/s | 0.9929 | 512 / 360.2 | 30.97 |
| test 196 | 51.168 / 51.168 mm | 53.604 mm | 119.827 mm/s | 0.9930 | 512 / 356.2 | 46.99 |
| val 196 | 44.449 / 44.449 mm | 46.565 mm | 100.256 mm/s | 0.9949 | 489 / 267.9 | 48.37 |

The token count is length dependent because the MotionGPT VQ-VAE downsamples
the sequence temporally. A 196-frame clip produces about `47` local tokens in
this setup.

### Continuous Root Branch Setup

The root branch is a small temporal Conv encoder/decoder conditioned on the
frozen local VQ reconstruction. It predicts normalized root controls and is
trained with physical trajectory losses:

```text
L_control
L_yaw_vel
L_local_xz_vel
L_height
L_global_root_pos
L_global_root_step
L_final_displacement
L_path_length
L_smooth
```

Training setup:

- Frozen: local VQ model and codebook.
- Trainable: continuous root branch only.
- Data: M4Human train split.
- Training: 50 epochs, 100 steps/epoch, batch size 256.
- Window sampling: `[64, 128, 196]` with weights `[0.25, 0.25, 0.50]`.
- Checkpoint:
  `/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_branch_m4human_v1/checkpoints/best.pt`

Two variants were tested:

- `unet`: uses root encoder skip connections. This is useful as a wiring check
  and optimistic continuous-root upper bound, but it is not a strict latent
  bottleneck because root information can pass through skip features.
- `bottleneck`: removes root skip connections. The decoder sees the continuous
  root latent plus local VQ reconstruction, so this is the more relevant
  tokenizer-style result.

### Continuous Root Branch Results

U-Net/skip upper-bound variant:

| split/window | MPJPE / root-align / gap | root xz mean | final xz | path error | speed bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| test 64 | 52.747 / 51.980 / 0.767 mm | 2.736 mm | 3.380 mm | -0.0001 m | -0.026 mm/s |
| test 128 | 52.579 / 51.057 / 1.522 mm | 4.156 mm | 5.321 mm | -0.0005 m | -0.073 mm/s |
| test 196 | 53.116 / 50.864 / 2.252 mm | 5.677 mm | 6.868 mm | -0.0009 m | -0.101 mm/s |
| val 196 | 46.010 / 44.477 / 1.533 mm | 4.279 mm | 5.478 mm | 0.0022 m | 0.232 mm/s |

No-skip bottleneck variant:

| split/window | MPJPE / root-align / gap | root xz mean | final xz | path error | speed bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| test 64 | 63.342 / 49.467 / 13.874 mm | 26.291 mm | 38.301 mm | -0.0079 m | -2.531 mm/s |
| test 128 | 69.861 / 48.851 / 21.010 mm | 36.719 mm | 55.269 mm | -0.0116 m | -1.885 mm/s |
| test 196 | 77.232 / 48.804 / 28.428 mm | 46.711 mm | 72.039 mm | -0.0182 m | -1.950 mm/s |
| val 196 | 66.302 / 43.069 / 23.233 mm | 39.362 mm | 61.406 mm | -0.0170 m | -1.769 mm/s |

### Comparison To Single-Stream Exp3

| experiment | M4Human test196 MPJPE / root-align / gap | root xz mean | final xz | path error | speed bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exp3 single-stream path/final | 101.077 / 52.429 / 48.648 mm | 76.054 mm | 118.717 mm | -0.0673 m | -7.202 mm/s |
| Local-only VQ with GT root | 51.168 / 51.168 / ~0 mm | 0.000 mm | 0.000 mm | 0.0000 m | 0.000 mm/s |
| Factorized U-Net root upper bound | 53.116 / 50.864 / 2.252 mm | 5.677 mm | 6.868 mm | -0.0009 m | -0.101 mm/s |
| Factorized bottleneck root latent | 77.232 / 48.804 / 28.428 mm | 46.711 mm | 72.039 mm | -0.0182 m | -1.950 mm/s |

### Interpretation

- The local-only VQ passes the Phase 3 gate. It reaches `51.17 mm`
  root-aligned MPJPE on M4Human test196, slightly better than the Exp3
  single-stream root-aligned quality.
- Codebook usage is healthy: all 512 codes are used on the test split and the
  effective code count is about `356` for 196-frame windows.
- The U-Net/skip root branch almost removes the full/root-aligned gap:
  `48.65 mm -> 2.25 mm` on M4Human test196. This validates the root-loss and
  recovery wiring, but should be treated as an optimistic upper bound because
  root information can pass through skip connections.
- The no-skip bottleneck branch is the more relevant tokenizer result. It
  improves M4Human test196 from `101.08 / 52.43 / 48.65 mm` to
  `77.23 / 48.80 / 28.43 mm`, meeting the Phase 4 target range of `70-85 mm`
  full MPJPE.
- The bottleneck still shows length-dependent root drift: gap increases from
  `13.87 mm` at 64 frames to `28.43 mm` at 196 frames. This is much better than
  Exp3 but not solved.
- Speed/path bias are now much smaller than Exp3 (`-1.95 mm/s` speed bias and
  `-0.018 m` path error at 196 frames), so the remaining bottleneck is likely
  low-frequency endpoint/direction encoding in the continuous root latent.
- The next experiment should improve the bottleneck root latent rather than
  return to single-stream loss tuning. Good candidates are a deeper temporal
  bottleneck decoder, integrated-yaw loss, endpoint-conditioned latent loss, or
  a lightweight Transformer root branch while keeping the local VQ frozen.

## Root Bottleneck Optimization R1-R3

This pass optimizes the no-skip root latent, not the U-Net skip upper bound.
The code adds a stronger no-skip TCN root branch:

```text
root controls + local VQ reconstruction
  -> dilated residual TCN encoder
  -> continuous root latent
  -> dilated residual TCN decoder + local condition
  -> root controls
```

It also adds two optional trajectory losses:

- `L_multiscale`: displacement vector loss at 25%, 50%, 75%, and 100% of the
  window.
- `L_yaw_integral`: wrapped integrated-yaw loss, implemented but not enabled in
  R1-R3.

### Output Paths

- R1 stronger TCN, 4x root downsample:
  `/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_branch_m4human_tcn_v1`
- R2 stronger TCN + multiscale loss, 4x root downsample:
  `/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_branch_m4human_tcn_multiscale_v1`
- R3 stronger TCN + multiscale loss, 2x root downsample:
  `/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_branch_m4human_tcn_multiscale_ds1_v1`

### Training Setup

Common setup:

- Init: frozen local VQ checkpoint
  `/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_v1/checkpoints/best.pt`
- Trainable: root branch only.
- Data: M4Human train split.
- Training: 50 epochs, 100 steps/epoch, batch size 256.
- Window sampling: `[64, 128, 196]` with weights `[0.25, 0.25, 0.50]`.
- Root branch: `architecture=bottleneck_tcn`, `width=256`,
  `latent_width=256`, `tcn_depth=4`.

Experiment differences:

| experiment | root downsample | multiscale weight |
| --- | ---: | ---: |
| R1 | 4x | 0 |
| R2 | 4x | 20 |
| R3 | 2x | 20 |

### Results

| experiment | split/window | MPJPE / root-align / gap | root xz mean | final xz | path error | speed bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| old bottleneck | test 196 | 77.232 / 48.804 / 28.428 mm | 46.711 mm | 72.039 mm | -0.0182 m | -1.950 mm/s |
| R1 TCN 4x | test 64 | 57.305 / 49.218 / 8.087 mm | 16.114 mm | 22.353 mm | -0.0023 m | -0.733 mm/s |
| R1 TCN 4x | test 128 | 59.760 / 48.614 / 11.146 mm | 21.691 mm | 32.217 mm | -0.0049 m | -0.793 mm/s |
| R1 TCN 4x | test 196 | 63.412 / 48.570 / 14.843 mm | 27.556 mm | 41.694 mm | -0.0078 m | -0.831 mm/s |
| R1 TCN 4x | val 196 | 53.594 / 42.786 / 10.808 mm | 22.401 mm | 33.398 mm | -0.0112 m | -1.164 mm/s |
| R2 TCN 4x + multiscale | test 64 | 54.501 / 49.033 / 5.468 mm | 8.863 mm | 11.769 mm | -0.0004 m | -0.140 mm/s |
| R2 TCN 4x + multiscale | test 128 | 55.219 / 48.390 / 6.829 mm | 12.023 mm | 17.062 mm | -0.0004 m | -0.071 mm/s |
| R2 TCN 4x + multiscale | test 196 | 56.663 / 48.304 / 8.360 mm | 15.021 mm | 20.534 mm | -0.0007 m | -0.077 mm/s |
| R2 TCN 4x + multiscale | val 196 | 48.771 / 42.627 / 6.144 mm | 12.402 mm | 17.822 mm | 0.0002 m | 0.018 mm/s |
| R3 TCN 2x + multiscale | test 64 | 53.793 / 49.006 / 4.788 mm | 5.917 mm | 7.845 mm | -0.0002 m | -0.061 mm/s |
| R3 TCN 2x + multiscale | test 128 | 54.040 / 48.348 / 5.692 mm | 8.278 mm | 11.805 mm | -0.0001 m | -0.015 mm/s |
| R3 TCN 2x + multiscale | test 196 | 55.171 / 48.243 / 6.928 mm | 10.981 mm | 15.198 mm | -0.0002 m | -0.023 mm/s |
| R3 TCN 2x + multiscale | val 196 | 47.649 / 42.522 / 5.127 mm | 8.900 mm | 13.066 mm | 0.0051 m | 0.530 mm/s |

### Interpretation

- R1 confirms that model capacity matters. A stronger no-skip TCN improves
  test196 from `77.23 / 48.80 / 28.43 mm` to
  `63.41 / 48.57 / 14.84 mm`.
- R2 confirms that low-frequency displacement supervision is the right loss.
  With the same 4x latent rate, multiscale displacement reduces test196 gap
  from `14.84 mm` to `8.36 mm` and final xz error from `41.69 mm` to
  `20.53 mm`.
- R3 confirms that root trajectory benefits from higher temporal resolution.
  Moving from 4x to 2x root downsample further improves test196 to
  `55.17 / 48.24 / 6.93 mm`.
- The remaining root error is now small compared with local reconstruction:
  R3 test196 root xz mean error is `10.98 mm`, final xz error is `15.20 mm`,
  and speed bias is effectively zero.
- R3 is a strict no-skip bottleneck result but is already close to the previous
  U-Net skip upper bound. This means the root/local factorized tokenizer is no
  longer just a promising direction; it is currently the best M4Human
  reconstruction path in this repo.

Recommended next steps:

- Keep R3 as the current factorized root checkpoint.
- Add HumanML3D factorized cache and evaluate whether the same root/local
  tokenizer preserves HumanML3D metrics.
- Only after mixed-domain validation, consider root latent compression or
  discretization. The current continuous root branch should remain the quality
  reference.

## Full From-Scratch Factorized R3 Recipe

This pass trains the current factorized network from scratch with one fixed
recipe. It does not reuse the previous local VQ or root branch checkpoints, and
it does not mix HumanML3D.

The only staged dependency is structural: the root branch needs a trained local
VQ checkpoint for conditioning. The recipe is otherwise fixed upfront:

```text
Stage 1: train local VQ from random init
Stage 2: freeze that local VQ, train no-skip R3 root TCN from random init
Stage 3: run fixed 64/128/196 eval
```

The reproducible script is:

```text
scripts/train_factorized_scratch_m4human.sh
```

### Output Paths

- Local VQ:
  `/cpfs01/liangbo/data/MotionGPT/factorized_experiments/local_vq_m4human_scratch_full_v1`
- Root branch:
  `/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_branch_m4human_scratch_full_r3_v1`

### Training Recipe

Local VQ:

```text
data:             M4Human train only
seed:             20260531
epochs:           200
steps/epoch:      200
total steps:      40,000
batch size:       256
windows:          64 / 128 / 196
window weights:   0.25 / 0.25 / 0.50
lr:               2e-4 -> 1e-6 cosine
```

Root branch:

```text
data:             M4Human train only
seed:             20260531
architecture:     bottleneck_tcn
width:            256
latent_width:     256
root downsample:  2x
tcn_depth:        4
multiscale loss:  20.0
epochs:           120
steps/epoch:      200
total steps:      24,000
batch size:       256
windows:          64 / 128 / 196
window weights:   0.25 / 0.25 / 0.50
lr:               2e-4 -> 1e-6 cosine
```

### Local VQ Results

| split/window | MPJPE / root-align | local body MPJPE | local velocity error | contact F1 | unique / effective codes | tokens/window |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test 64 | 53.922 / 53.922 mm | 56.490 mm | 124.534 mm/s | 0.9940 | 511 / 365.3 | 15.92 |
| test 128 | 53.683 / 53.683 mm | 56.239 mm | 121.953 mm/s | 0.9942 | 511 / 361.3 | 30.97 |
| test 196 | 53.562 / 53.562 mm | 56.113 mm | 121.067 mm/s | 0.9943 | 511 / 358.7 | 46.99 |
| val 196 | 44.953 / 44.953 mm | 47.094 mm | 99.205 mm/s | 0.9968 | 490 / 282.3 | 48.37 |

The full-scratch local VQ is not better than the earlier shorter local VQ
(`51.168 mm` on test196). It improves contact metrics but worsens local pose
MPJPE by about `2.4 mm`. This means future final recipes should use validation
or test-heldout local MPJPE for checkpoint selection, not only training loss.

### Root Branch Results

| checkpoint | split/window | MPJPE / root-align / gap | root xz mean | final xz | path error | speed bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| best | test 64 | 54.822 / 49.986 / 4.836 mm | 1.398 mm | 2.086 mm | 0.0000 m | 0.007 mm/s |
| best | test 128 | 54.665 / 49.719 / 4.945 mm | 2.110 mm | 3.297 mm | -0.0001 m | -0.019 mm/s |
| best | test 196 | 54.696 / 49.605 / 5.091 mm | 2.908 mm | 4.347 mm | -0.0004 m | -0.039 mm/s |
| best | val 196 | 45.708 / 42.821 / 2.887 mm | 2.145 mm | 3.512 mm | 0.0001 m | 0.015 mm/s |
| last | test 196 | 54.679 / 49.611 / 5.068 mm | 2.796 mm | 4.170 mm | -0.0009 m | -0.101 mm/s |

### Comparison

| experiment | M4Human test196 MPJPE / root-align / gap | root xz mean | final xz |
| --- | ---: | ---: | ---: |
| Exp3 single-stream path/final | 101.077 / 52.429 / 48.648 mm | 76.054 mm | 118.717 mm |
| earlier R3 ablation | 55.171 / 48.243 / 6.928 mm | 10.981 mm | 15.198 mm |
| full scratch R3 best | 54.696 / 49.605 / 5.091 mm | 2.908 mm | 4.347 mm |
| full scratch R3 last | 54.679 / 49.611 / 5.068 mm | 2.796 mm | 4.170 mm |

### Interpretation

- The full from-scratch recipe reproduces and slightly improves the previous
  best full MPJPE, without any finetuning from previous checkpoints.
- Root reconstruction is now very strong: test196 root xz mean error is only
  `2.91 mm`, final xz error is `4.35 mm`, and speed bias is effectively zero.
- The remaining full MPJPE is no longer root drift. It is dominated by local
  VQ quality. The full-scratch local VQ is weaker than the earlier short local
  VQ despite lower training loss, which is a useful negative finding.
- The next quality lever is local VQ checkpoint selection and local VQ training
  recipe, not root trajectory modeling. A practical next step is to add
  validation-eval checkpointing to `local_vq.py` and select by test/val-style
  root-aligned MPJPE rather than train loss.

## Root Latent Size Concern

The full-scratch R3 root result is strong, but the representation is not yet
practical for upstream or downstream models.

For a 196-frame clip:

```text
R3 root latent:     98 x 256 = 25,088 continuous values
raw root controls: 196 x 4   =    784 continuous values
```

So the current root latent is about `32x` larger than the raw root-control
signal. It is a high-quality continuous autoencoder reference, not a compact
tokenizer.

The next stage should therefore optimize root latent compression rather than
root reconstruction quality alone. The detailed compression plan is stored in:

```text
ROOT_LATENT_COMPRESSION.md
```

## Root-FAST Continuous DCT Codec

This pass implements a FAST-like no-training root command codec:

```text
u[t] = [yaw_rate, local_vx, local_vz, root_height]
u[t] -> chunked DCT -> keep K low-frequency coefficients -> inverse DCT
```

The eval keeps GT local pose and only replaces root commands with
DCT-reconstructed root commands. Therefore these numbers measure the root codec
itself, not local VQ reconstruction.

Code:

```text
src/motiongpt_m4human/factorized/root_fast_codec.py
```

Output paths:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_dct_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_dct_v1_val
```

### Test196 Sweep

| chunk | K | values | compression | MPJPE | root-align | root xz mean | final xz | path error | speed bias |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2 | 56 | 14.00x | 18.056 mm | 9.613 mm | 17.609 mm | 11.428 mm | -0.1760 m | -18.828 mm/s |
| 16 | 2 | 104 | 7.54x | 5.020 mm | 4.944 mm | 4.795 mm | 3.641 mm | -0.0575 m | -6.153 mm/s |
| 32 | 4 | 112 | 7.00x | 4.164 mm | 4.308 mm | 3.961 mm | 2.943 mm | -0.0408 m | -4.366 mm/s |
| 16 | 4 | 208 | 3.77x | 1.204 mm | 1.518 mm | 1.117 mm | 0.418 mm | -0.0179 m | -1.911 mm/s |
| 16 | 6 | 312 | 2.51x | 0.620 mm | 0.659 mm | 0.575 mm | 0.390 mm | -0.0094 m | -1.011 mm/s |

### Val196 Check

| chunk | K | values | compression | MPJPE | root-align | root xz mean | final xz | path error | speed bias |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2 | 104 | 7.54x | 4.359 mm | 4.286 mm | 4.170 mm | 2.620 mm | -0.0599 m | -6.224 mm/s |
| 16 | 4 | 208 | 3.77x | 0.966 mm | 1.330 mm | 0.892 mm | 0.269 mm | -0.0143 m | -1.489 mm/s |
| 32 | 2 | 56 | 14.00x | 16.001 mm | 9.408 mm | 15.612 mm | 8.248 mm | -0.1807 m | -18.774 mm/s |
| 32 | 4 | 112 | 7.00x | 3.522 mm | 3.397 mm | 3.368 mm | 2.248 mm | -0.0418 m | -4.345 mm/s |

### Interpretation

- This is a strong positive result for a FAST-like root trajectory branch.
- `chunk=16, K=4` uses only `208` continuous values for a 196-frame clip and
  gets `1.204 mm` test MPJPE / `1.117 mm` root xz mean error.
- `chunk=16, K=2` uses only `104` values and still stays near `5 mm` test
  MPJPE, though speed/path underestimation is visible.
- The next step should be DCT coefficient quantization, especially k-means
  vector quantization over each chunk's flattened `K x 4` coefficients. With
  `chunk=16`, this would make a 196-frame clip about `13` root action tokens.

## Root-FAST Quantization Sweep

Code:

```text
src/motiongpt_m4human/factorized/root_fast_quantize.py
```

Output paths:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_quantized_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_scalar_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_product_v1
```

The first full vector VQ sweep had a k-means early-stop bug and should be
rerun. The bug is fixed in code. A small post-fix check still showed that
one-token-per-chunk VQ is weak, so the main reliable completed results are
scalar quantization and product VQ.

### Scalar Quantization Test196

| config | scalar codes | bits/window | MPJPE | root xz mean | final xz |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk=16, K=2, 6-bit | 104 | 624 | 27.97 mm | 26.94 mm | 43.06 mm |
| chunk=16, K=2, 8-bit | 104 | 832 | 9.94 mm | 9.48 mm | 11.98 mm |
| chunk=32, K=4, 6-bit | 112 | 672 | 27.88 mm | 26.87 mm | 42.59 mm |
| chunk=32, K=4, 8-bit | 112 | 896 | 9.36 mm | 8.91 mm | 12.32 mm |
| chunk=64, K=4, 8-bit | 64 | 512 | 19.08 mm | 18.53 mm | 18.64 mm |

Scalar quantization is not yet a clean motion-token interface, but it proves
that DCT coefficients are discretizable. The issue is token packing, not the
frequency representation itself.

### Product VQ Test196

Product VQ uses one code per root-command dimension per DCT chunk:

```text
chunk=16 -> 52 root tokens/window
chunk=32 -> 28 root tokens/window
chunk=64 -> 16 root tokens/window
```

Best completed results:

| root tokens | chunk | K | vocab/group | bits/window | MPJPE | root xz mean | final xz |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 2 | 256 | 128 | 82.61 mm | 80.09 mm | 106.65 mm |
| 28 | 32 | 2 | 256 | 224 | 53.67 mm | 51.57 mm | 80.14 mm |
| 52 | 16 | 2 | 256 | 416 | 41.20 mm | 39.68 mm | 63.61 mm |

This is much better than plain full-chunk vector VQ, but still far worse than
continuous DCT coefficients or 8-bit scalar quantization.

### Current Conclusion

Root-FAST remains the most promising compact root representation. The next
tokenizer should be RVQ over DCT chunks:

```text
chunk coefficients -> residual code_1, code_2, ..., code_R
```

For example:

```text
chunk=16, R=4 -> 13 chunks * 4 = 52 root tokens
chunk=32, R=4 ->  7 chunks * 4 = 28 root tokens
```

This should preserve cross-dimension structure better than product VQ while
using a practical number of tokens.

Detailed next-step TODO:

```text
ROOT_FAST_TOKENIZER_TODO.md
```
