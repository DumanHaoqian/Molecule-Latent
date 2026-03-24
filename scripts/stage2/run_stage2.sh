#!/usr/bin/env bash
set -euo pipefail

cd /home/haoqian/Data/Molecule/Latent
mkdir -p logs

RUN_NAME="stage2-grpo-openmolins"
CUDA_VISIBLE_DEVICES="1,5"

if [[ -z "${RUN_NAME// }" ]]; then
  echo "ERROR: RUN_NAME is empty. Please set RUN_NAME in this script."
  exit 1
fi

RUN_TAG="$(echo "${RUN_NAME}" | tr -cs 'A-Za-z0-9._-' '-' | sed 's/^-*//;s/-*$//')"
if [[ -z "${RUN_TAG}" ]]; then
  RUN_TAG="stage2_grpo"
fi
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/home/haoqian/Data/Molecule/Latent/logs/stage2_grpo_${RUN_TAG}_${TIMESTAMP}.log"

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python /home/haoqian/Data/Molecule/Latent/stage2_grpo.py \
  --train_config /home/haoqian/Data/Molecule/Latent/configs/stage2_grpo/train_config.yaml \
  --data_config /home/haoqian/Data/Molecule/Latent/configs/stage2_grpo/data_config.yaml \
  --run_name "${RUN_NAME}" \
> "${LOG_FILE}" 2>&1 &

echo "Started Stage2 GRPO training with nohup."
echo "RUN_NAME=${RUN_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Log: ${LOG_FILE}"
