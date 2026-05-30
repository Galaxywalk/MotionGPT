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
