#!/usr/bin/env bash
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

PYTHON_BIN="${PYTHON_BIN:-/cpfs01/liangbo/data/conda_envs/mgpt/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-/cpfs01/liangbo/data/MotionGPT/factorized_cache/v1_m4human_xz-y_20hz}"
EXP_ROOT="${EXP_ROOT:-/cpfs01/liangbo/data/MotionGPT/factorized_experiments}"
SEED="${SEED:-20260531}"
LOCAL_GPU="${LOCAL_GPU:-5}"
ROOT_GPU="${ROOT_GPU:-6}"
EVAL_GPU="${EVAL_GPU:-6}"

LOCAL_EXP="${LOCAL_EXP:-${EXP_ROOT}/local_vq_m4human_scratch_full_v1}"
ROOT_EXP="${ROOT_EXP:-${EXP_ROOT}/root_branch_m4human_scratch_full_r3_v1}"

cd "$(dirname "$0")/.."

echo "[1/4] Train local VQ from scratch"
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES="${LOCAL_GPU}" "${PYTHON_BIN}" \
  -m motiongpt_m4human.factorized.local_vq train \
  --cache-root "${CACHE_ROOT}" \
  --exp-root "${LOCAL_EXP}" \
  --seed "${SEED}" \
  --epochs 200 \
  --steps-per-epoch 200 \
  --batch-size 256 \
  --window-sizes 64 128 196 \
  --window-weights 0.25 0.25 0.5 \
  --lr 2e-4 \
  --lr-min 1e-6 \
  --device cuda:0

echo "[2/4] Evaluate local VQ"
for window in 64 128 196; do
  PYTHONPATH=src:. CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" \
    -m motiongpt_m4human.factorized.local_vq eval \
    --checkpoint "${LOCAL_EXP}/checkpoints/best.pt" \
    --cache-root "${CACHE_ROOT}" \
    --split test \
    --window-frames "${window}" \
    --stride "${window}" \
    --batch-size 512 \
    --device cuda:0 \
    --out-json "${LOCAL_EXP}/eval/best_test${window}.json"
done
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" \
  -m motiongpt_m4human.factorized.local_vq eval \
  --checkpoint "${LOCAL_EXP}/checkpoints/best.pt" \
  --cache-root "${CACHE_ROOT}" \
  --split val \
  --window-frames 196 \
  --stride 196 \
  --batch-size 512 \
  --device cuda:0 \
  --out-json "${LOCAL_EXP}/eval/best_val196.json"

echo "[3/4] Train no-skip R3 root branch from scratch"
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES="${ROOT_GPU}" "${PYTHON_BIN}" \
  -m motiongpt_m4human.factorized.root_branch train \
  --architecture bottleneck_tcn \
  --width 256 \
  --latent-width 256 \
  --root-downsample-layers 1 \
  --tcn-depth 4 \
  --lambda-multiscale 20.0 \
  --cache-root "${CACHE_ROOT}" \
  --local-vq-checkpoint "${LOCAL_EXP}/checkpoints/best.pt" \
  --exp-root "${ROOT_EXP}" \
  --seed "${SEED}" \
  --epochs 120 \
  --steps-per-epoch 200 \
  --batch-size 256 \
  --window-sizes 64 128 196 \
  --window-weights 0.25 0.25 0.5 \
  --lr 2e-4 \
  --lr-min 1e-6 \
  --device cuda:0

echo "[4/4] Evaluate root branch"
for window in 64 128 196; do
  PYTHONPATH=src:. CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" \
    -m motiongpt_m4human.factorized.root_branch eval \
    --checkpoint "${ROOT_EXP}/checkpoints/best.pt" \
    --cache-root "${CACHE_ROOT}" \
    --split test \
    --window-frames "${window}" \
    --stride "${window}" \
    --batch-size 512 \
    --device cuda:0 \
    --out-json "${ROOT_EXP}/eval/best_test${window}.json"
done
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" \
  -m motiongpt_m4human.factorized.root_branch eval \
  --checkpoint "${ROOT_EXP}/checkpoints/best.pt" \
  --cache-root "${CACHE_ROOT}" \
  --split val \
  --window-frames 196 \
  --stride 196 \
  --batch-size 512 \
  --device cuda:0 \
  --out-json "${ROOT_EXP}/eval/best_val196.json"

PYTHONPATH=src:. CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" \
  -m motiongpt_m4human.factorized.root_branch eval \
  --checkpoint "${ROOT_EXP}/checkpoints/last.pt" \
  --cache-root "${CACHE_ROOT}" \
  --split test \
  --window-frames 196 \
  --stride 196 \
  --batch-size 512 \
  --device cuda:0 \
  --out-json "${ROOT_EXP}/eval/last_test196.json"

echo "Done."
echo "Local VQ:    ${LOCAL_EXP}"
echo "Root branch: ${ROOT_EXP}"
