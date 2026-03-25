#!/usr/bin/env bash
set -euo pipefail

cd /home/haoqian/Data/Molecule/Latent
mkdir -p logs

# ===== Edit here =====
# Fill your experiment name. It will be used for:
# 1) W&B run name
# 2) checkpoint directory/name tags
# 3) log filename tag
RUN_NAME="naive-lr-4slots-continue"
INIT_STAGE1_CKPT="${INIT_STAGE1_CKPT:-/home/haoqian/Data/Molecule/Latent/checkpoints/stage1/20260322_002518-naive-lr-4slots-dorfrz77/last.ckpt}"

# Directly edit GPUs here, no extra CLI args needed.
CUDA_VISIBLE_DEVICES="1"
# =====================

if [[ -z "${RUN_NAME// }" ]]; then
  echo "ERROR: RUN_NAME is empty. Please set RUN_NAME in this script."
  exit 1
fi

RUN_TAG="$(echo "${RUN_NAME}" | tr -cs 'A-Za-z0-9._-' '-' | sed 's/^-*//;s/-*$//')"
if [[ -z "${RUN_TAG}" ]]; then
  RUN_TAG="stage1"
fi
if [[ -n "${INIT_STAGE1_CKPT}" && ! -f "${INIT_STAGE1_CKPT}" ]]; then
  echo "ERROR: INIT_STAGE1_CKPT not found: ${INIT_STAGE1_CKPT}"
  exit 1
fi
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/home/haoqian/Data/Molecule/Latent/logs/stage1_${RUN_TAG}_${TIMESTAMP}.log"

cmd=(python /home/haoqian/Data/Molecule/Latent/stage1.py
  --train_config /home/haoqian/Data/Molecule/Latent/configs/stage1/train_config.yaml \
  --data_config /home/haoqian/Data/Molecule/Latent/configs/stage1/data_config.yaml \
  --run_name "${RUN_NAME}")
if [[ -n "${INIT_STAGE1_CKPT}" ]]; then
  cmd+=(--init_stage1_ckpt "${INIT_STAGE1_CKPT}")
fi

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}" > "${LOG_FILE}" 2>&1 &

echo "Started Stage1 training with nohup."
echo "RUN_NAME=${RUN_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "INIT_STAGE1_CKPT=${INIT_STAGE1_CKPT:-<empty>}"
echo "Log: ${LOG_FILE}"
