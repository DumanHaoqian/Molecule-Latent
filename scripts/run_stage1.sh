#!/usr/bin/env bash
set -euo pipefail

cd /home/haoqian/Data/Molecule/Latent
mkdir -p logs

# ===== Edit here =====
# Fill your experiment name. It will be used for:
# 1) W&B run name
# 2) checkpoint directory/name tags
# 3) log filename tag
RUN_NAME="naive-lr-4slots"

# Directly edit GPUs here, no extra CLI args needed.
CUDA_VISIBLE_DEVICES="2"
# =====================

if [[ -z "${RUN_NAME// }" ]]; then
  echo "ERROR: RUN_NAME is empty. Please set RUN_NAME in this script."
  exit 1
fi

RUN_TAG="$(echo "${RUN_NAME}" | tr -cs 'A-Za-z0-9._-' '-' | sed 's/^-*//;s/-*$//')"
if [[ -z "${RUN_TAG}" ]]; then
  RUN_TAG="stage1"
fi
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/home/haoqian/Data/Molecule/Latent/logs/stage1_${RUN_TAG}_${TIMESTAMP}.log"

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python /home/haoqian/Data/Molecule/Latent/stage1.py \
  --train_config /home/haoqian/Data/Molecule/Latent/configs/stage1/train_config.yaml \
  --data_config /home/haoqian/Data/Molecule/Latent/configs/stage1/data_config.yaml \
  --run_name "${RUN_NAME}" \
> "${LOG_FILE}" 2>&1 &

echo "Started Stage1 training with nohup."
echo "RUN_NAME=${RUN_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Log: ${LOG_FILE}"
