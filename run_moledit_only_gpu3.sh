#!/usr/bin/env bash
set -euo pipefail

cd /home/haoqian/Data/Molecule/Latent
mkdir -p logs

nohup env CUDA_VISIBLE_DEVICES=3 \
python stage1.py \
  --train_config configs/stage1/train_config.yaml \
  --data_config configs/stage1/data_config_moledit_only.yaml \
> logs/stage1_nohup_moledit_only_gpu3.log 2>&1 &

echo "Started Stage1 training (moledit only) on GPU 3."
echo "Log: /home/haoqian/Data/Molecule/Latent/logs/stage1_nohup_moledit_only_gpu3.log"
