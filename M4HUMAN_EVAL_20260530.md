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

1. Continue from mixed multilen epoch499 and add recover-space trajectory
   losses only: final displacement and path-length loss with weight 0.01 each.
2. Test a lightweight root velocity calibration head for M4Human-domain root
   x/z velocities.
3. If trajectory loss is insufficient, add a batch-level speed statistic loss.
