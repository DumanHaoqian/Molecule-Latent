#!/usr/bin/env bash
set -euo pipefail

cd /home/haoqian/Data/Molecule/Latent
mkdir -p logs
mkdir -p /home/haoqian/Data/Molecule/Latent/checkpoints/stage1

# ===== Baseline-1 (text-only LM path: forward_stage1_base) =====
# RUN_NAME -> W&B run name, checkpoint subdir/file prefix (same rules as stage1.py).
# Checkpoints: checkpoints/stage1/<YYYYMMDD_HHMMSS>-<sanitized_name>[-<wandb_id>]/
# CSV metrics: see train_config.baseline.yaml -> stage1.test_results_csv_path
RUN_NAME="text-baseline-molllama"

# Physical GPU index (4 -> use GPU 4 on this machine).
CUDA_VISIBLE_DEVICES="4"

TRAIN_CONFIG="/home/haoqian/Data/Molecule/Latent/configs/stage1/train_config.baseline.yaml"
DATA_CONFIG="/home/haoqian/Data/Molecule/Latent/configs/stage1/data_config.yaml"
# ==============================================================

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
  --train_config "${TRAIN_CONFIG}" \
  --data_config "${DATA_CONFIG}" \
  --run_name "${RUN_NAME}" \
> "${LOG_FILE}" 2>&1 &

echo "Started Stage1 Baseline-1 training with nohup."
echo "RUN_NAME=${RUN_NAME}  (W&B run name; --run_name overrides yaml wandb_run_name)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Train config: ${TRAIN_CONFIG}  (use_latent_training=false, test_baseline.csv)"
echo "Checkpoints under: /home/haoqian/Data/Molecule/Latent/checkpoints/stage1/<timestamp>-<run>[-id]/"
echo "Log: ${LOG_FILE}"
