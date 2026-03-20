#!/usr/bin/env bash
set -euo pipefail

cd /home/haoqian/Data/Molecule/Latent
mkdir -p logs

nohup env CUDA_VISIBLE_DEVICES=4 \
python stage1.py \
  --train_config configs/stage1/train_config.yaml \
  --data_config configs/stage1/data_config_moledit_conversation.yaml \
> logs/stage1_nohup_moledit_conversation_gpu4.log 2>&1 &

echo "Started Stage1 training (moledit + conversation) on GPU 4."
echo "Log: /home/haoqian/Data/Molecule/Latent/logs/stage1_nohup_moledit_conversation_gpu4.log"
