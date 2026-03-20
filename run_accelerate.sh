#!/usr/bin/env bash
set -euo pipefail

cd /home/haoqian/Data/Molecule/Latent
mkdir -p logs

# ===== Edit here =====
# Example: GPU_IDS="2,3" means use GPU 2 and GPU 3.
GPU_IDS="2"
MAIN_PROCESS_PORT="29512"
MIXED_PRECISION="bf16"
# =====================

NUM_PROCESSES=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

nohup env CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --mixed_precision "${MIXED_PRECISION}" \
  stage1.py \
  --train_config configs/stage1/train_config.yaml \
  --data_config configs/stage1/data_config.yaml \
> logs/stage1_accelerate.log 2>&1 &

echo "Started Stage1 training with accelerate."
echo "Log: /home/haoqian/Data/Molecule/Latent/logs/stage1_accelerate.log"
echo "CUDA_VISIBLE_DEVICES=${GPU_IDS}"
echo "num_processes=${NUM_PROCESSES}, mixed_precision=${MIXED_PRECISION}, port=${MAIN_PROCESS_PORT}"
