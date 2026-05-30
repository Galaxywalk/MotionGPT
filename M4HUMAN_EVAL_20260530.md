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
