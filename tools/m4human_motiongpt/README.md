# M4Human MotionGPT Reconstruction Check

This folder contains a small dataloader and evaluation script for testing a
trained MotionGPT VQVAE checkpoint on M4Human pose annotations.

The dataloader reads `/rf3dpose_all/params.lmdb`, `/rf3dpose_all/calib.lmdb`,
and `indeces.pkl.gz`, groups frames by `(subject, action)`, keeps continuous
segments, transforms cached M4Human SMPL-X joints into radar coordinates, then
converts them to MotionGPT's HumanML3D-style 263-D motion features.

Default paths match the current machine:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. conda run -p /home/liangbo/conda_envs/widouble \
  python -m tools.m4human_motiongpt.eval_reconstruction \
  --m4human-root /cpfs01/liangbo/widouble_workspace \
  --checkpoint experiments/mgpt/VQVAE_HumanML3D_H200_2000e_bs256_eval100/checkpoints/min-MPJPEep=0.ckpt \
  --device cuda \
  --max-windows 16
```

Use the SMPL-X forward path instead of cached joints with:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. conda run -p /home/liangbo/conda_envs/widouble \
  python -m tools.m4human_motiongpt.eval_reconstruction \
  --pose-source smplx \
  --smplx-model-root /cpfs01/liangbo/widouble_workspace/models \
  --device cuda \
  --max-windows 16
```

Notes:

- Use the `widouble` conda env for this tool because it already has `lmdb`,
  `msgpack`, and the M4Human-side dependencies.
- `feature_frames=196` produces `196 / 4 = 49` MotionGPT VQVAE tokens per
  motion window with the current tokenizer architecture.
- `--max-windows 16` keeps the command quick. Use `--max-windows 0` to evaluate
  every continuous window in the selected split.
- `--pose-source param_joints` uses cached M4Human SMPL-X joints from
  `params.lmdb`. `--pose-source smplx` rebuilds joints by running local SMPL-X
  forward from `betas`, `pose_body`, `root_orient`, and `trans` after calibration.
- `axis_mode=xz-y` maps M4Human radar coordinates to MotionGPT's y-up
  convention as `[x, z, -y]`. In the current small reconstruction check it is
  much better than `xzy`; compare `-xzy` if the downstream representation looks
  mirrored.
- If raw HumanML3D reference joints are available, pass `--reference-joints` to
  use that skeleton as the feature conversion target. Otherwise the first
  M4Human pose initializes the target offsets for this diagnostic.
