# M4Human Feature Cache

This package converts M4Human LMDB pose annotations into MotionGPT/HumanML3D
263-D motion features.

Quick smoke test:

```bash
PYTHONPATH=src:. conda run -p /home/liangbo/conda_envs/widouble \
  python -m motiongpt_m4human.build_feature_cache \
  --subset test \
  --max-sequences 2 \
  --out-root /cpfs01/liangbo/data/MotionGPT/m4human_cache/smoke_xz-y_param_joints \
  --overwrite \
  --save-joints-radar
```

Full 20Hz cache for all p1/s2 splits:

```bash
PYTHONPATH=src:. conda run -p /home/liangbo/conda_envs/widouble \
  python -m motiongpt_m4human.build_feature_cache \
  --subset all \
  --source-fps 10 \
  --target-fps 20 \
  --out-root /cpfs01/liangbo/data/MotionGPT/m4human_cache/v2_xz-y_param_joints_m4ref_20hz \
  --overwrite
```

Output layout:

```text
manifest.json
sequences.jsonl
errors.jsonl
features/*.npy          # [T - 1, 263], float32
canonical_joints/*.npy  # [T - 1, 22, 3], float32
joints_radar/*.npy      # optional [T, 22, 3], float32
meta/args.json
```

Defaults:

- `axis_mode=xz-y`, mapping M4Human radar coordinates to MotionGPT y-up as
  `[x, z, -y]`.
- `pose_source=param_joints`, using cached M4Human SMPL-X joints from
  `params.lmdb`.
- `pose_source=smplx` is available for audit/rebuilds with local SMPL-X models.
- `source_fps=10`, `target_fps=20`, `resample_method=linear_joints`, which
  linearly interpolates the 22-joint sequence before HumanML3D feature
  extraction.

Evaluate a cached feature set with the current VQVAE checkpoint:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src:. conda run -p /home/liangbo/conda_envs/widouble \
  python -m motiongpt_m4human.eval_cached_reconstruction \
  --cache-root /cpfs01/liangbo/data/MotionGPT/m4human_cache/v2_xz-y_param_joints_m4ref_20hz \
  --subset test \
  --device cuda \
  --out-json /cpfs01/liangbo/data/MotionGPT/m4human_cache/v2_xz-y_param_joints_m4ref_20hz/eval/vqvae_test.json
```

Mixed HumanML3D + M4Human VQ fine-tuning:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 \
  /cpfs01/liangbo/data/conda_envs/mgpt/bin/python train.py \
  --cfg configs/config_h3d_m4human_stage1.yaml \
  --batch_size 64 \
  --device 0 1 2 3 \
  --nodebug
```

The mixed training config uses the 20Hz cache above, keeps HumanML3D mean/std,
samples M4Human windows with `MIX_RATIO=0.3`, and validates on HumanML3D test
for comparability with the original tokenizer run.

Evaluate the mixed checkpoint on the full M4Human cache:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src:. /home/liangbo/conda_envs/widouble/bin/python \
  -m motiongpt_m4human.eval_cached_reconstruction \
  --cache-root /cpfs01/liangbo/data/MotionGPT/m4human_cache/v2_xz-y_param_joints_m4ref_20hz \
  --checkpoint experiments/mgpt/VQVAE_HumanML3D_M4Human20Hz_mix30_bs256_finetune/checkpoints/min-MPJPEep=0.ckpt \
  --subset all \
  --batch-size 256 \
  --device cuda \
  --out-json /cpfs01/liangbo/data/MotionGPT/m4human_cache/v2_xz-y_param_joints_m4ref_20hz/eval/vqvae_mixed500_min_mpjpe_all.json
```

Render coordinate-system comparison sheets:

```bash
PYTHONPATH=src:. conda run -p /home/liangbo/conda_envs/widouble \
  python -m motiongpt_m4human.visualize_coordinate_check \
  --out-root /cpfs01/liangbo/data/MotionGPT/m4human_axis_check
```

Find and render clips with the largest root movement:

```bash
PYTHONPATH=src:. conda run -p /home/liangbo/conda_envs/widouble \
  python -m motiongpt_m4human.visualize_large_movement \
  --out-root /cpfs01/liangbo/data/MotionGPT/m4human_large_movement
```
