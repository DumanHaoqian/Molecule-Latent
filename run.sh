#!/usr/bin/env bash
set -euo pipefail

cd /home/haoqian/Data/Molecule/Latent

mkdir -p logs

nohup env CUDA_VISIBLE_DEVICES=2,3 TOKENIZERS_PARALLELISM=false \
python stage1.py \
  --train_config configs/stage1/train_config.yaml \
  --data_config configs/stage1/data_config.yaml \
> logs/stage1_nohup_23.log 2>&1 &

echo "Started Stage1 training in background."
echo "Log file: /home/haoqian/Data/Molecule/Latent/logs/stage1_nohup_0_7.log"
