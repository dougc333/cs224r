#!/usr/bin/env bash
set -euo pipefail

# Run from hw1/
#   chmod +x ant-video.sh
#   ./ant-video.sh

ENV_NAME="Ant-v4"

# ----------------------------
# Learning rate sweep
# ----------------------------
LRS=(1e-2 3e-3 1e-3 3e-4 1e-4)

# ----------------------------
# Fixed hyperparameters
# ----------------------------
SEED=0
N_ITER=10
BATCH_SIZE=5000
HIDDEN_SIZE=64
N_LAYERS=2
EP_LEN=1000
VIDEO_LOG_FREQ=10   # only save video at final iter

# Headless MuJoCo (required on Colab)
export MUJOCO_GL="${MUJOCO_GL:-egl}"

TS="$(date +%Y-%m-%d_%H-%M-%S)"

echo "=== Ant BC LR sweep ==="
echo "Timestamp: $TS"

for lr in "${LRS[@]}"; do
  EXP_NAME="bc_ant_lr${lr}_ts${TS}"

  echo
  echo ">>> Running ${EXP_NAME}"

  python cs224r/scripts/run_hw1.py \
    --env_name "${ENV_NAME}" \
    --exp_name "${EXP_NAME}" \
    --seed "${SEED}" \
    --learning_rate "${lr}" \
    --n_iter "${N_ITER}" \
    --batch_size "${BATCH_SIZE}" \
    --ep_len "${EP_LEN}" \
    --video_log_freq "${VIDEO_LOG_FREQ}" \
    --n_layers "${N_LAYERS}" \
    --size "${HIDDEN_SIZE}"

done

echo "=== LR sweep complete ==="