#!/usr/bin/env bash
set -euo pipefail

cd /home/haoqian/Data/Molecule/Latent
mkdir -p logs

# Two-GPU DDP on physical GPUs 1 and 5.
export CUDA_VISIBLE_DEVICES="1,5"
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN_DEFAULT="/home/haoqian/Data/miniconda3/envs/molllama/bin/python"
if [[ -x "${PYTHON_BIN_DEFAULT}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_DEFAULT}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

TRAIN_CONFIGS=(
  "/home/haoqian/Data/Molecule/Latent/configs/stage2/train_config.300_case1.yaml"
  "/home/haoqian/Data/Molecule/Latent/configs/stage2/train_config.300_case2.yaml"
  "/home/haoqian/Data/Molecule/Latent/configs/stage2/train_config.300_case3.yaml"
)
DATA_CONFIGS=(
  "/home/haoqian/Data/Molecule/Latent/configs/stage2/data_config.300_case1.yaml"
  "/home/haoqian/Data/Molecule/Latent/configs/stage2/data_config.300_case2.yaml"
  "/home/haoqian/Data/Molecule/Latent/configs/stage2/data_config.300_case3.yaml"
)
RUN_NAMES=(
  "stage2-grpo-300-case1-g8-kl3e3"
  "stage2-grpo-300-case2-pure-rl"
  "stage2-grpo-300-case3-reward-lite"
)

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Running 3 x 300-step Stage2-GRPO ablations..."

for i in "${!RUN_NAMES[@]}"; do
  run_name="${RUN_NAMES[$i]}"
  train_cfg="${TRAIN_CONFIGS[$i]}"
  data_cfg="${DATA_CONFIGS[$i]}"
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="/home/haoqian/Data/Molecule/Latent/logs/${run_name}_${ts}.log"

  echo ""
  echo "=================================================="
  echo "[$((i+1))/3] ${run_name}"
  echo "train_config=${train_cfg}"
  echo "data_config=${data_cfg}"
  echo "log=${log_file}"
  echo "=================================================="

  before_latest="$(ls -dt lightning_logs/stage2_grpo/version_* 2>/dev/null | head -n 1 || true)"

  "${PYTHON_BIN}" /home/haoqian/Data/Molecule/Latent/stage2_grpo.py \
    --train_config "${train_cfg}" \
    --data_config "${data_cfg}" \
    --run_name "${run_name}" \
    2>&1 | tee "${log_file}"

  after_latest="$(ls -dt lightning_logs/stage2_grpo/version_* 2>/dev/null | head -n 1 || true)"
  metrics_csv="${after_latest}/metrics.csv"
  if [[ -z "${after_latest}" || ! -f "${metrics_csv}" ]]; then
    echo "[WARN] metrics.csv not found for ${run_name}. before=${before_latest} after=${after_latest}"
    continue
  fi

  echo "[INFO] Summarizing metrics from ${metrics_csv}"
  python - <<'PY' "${metrics_csv}" "${run_name}"
import csv, math, sys
metrics_csv = sys.argv[1]
run_name = sys.argv[2]

def pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

with open(metrics_csv, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    print(f"[{run_name}] metrics.csv is empty.")
    sys.exit(0)

cols = rows[0].keys()
policy_col = pick_col(cols, ["train/policy_loss_step", "train/policy_loss"])
rl_col = pick_col(cols, ["train/rl_loss_step", "train/rl_loss"])
invalid_col = pick_col(cols, ["train/grpo_invalid_batch_step", "train/grpo_invalid_batch"])

def vals(col):
    out = []
    if col is None:
        return out
    for r in rows:
        v = r.get(col, "")
        if v is None or v == "":
            continue
        try:
            fv = float(v)
            if math.isfinite(fv):
                out.append(fv)
        except Exception:
            pass
    return out

policy = vals(policy_col)
rl = vals(rl_col)
invalid = vals(invalid_col)

def summarize(name, arr):
    if len(arr) == 0:
        return f"{name}: no-data"
    nz = sum(1 for x in arr if abs(x) > 1e-9)
    return (
        f"{name}: n={len(arr)} min={min(arr):.6g} max={max(arr):.6g} "
        f"mean={sum(arr)/len(arr):.6g} nonzero_ratio={nz/len(arr):.3f}"
    )

print(f"[{run_name}] {summarize('policy_loss', policy)}")
print(f"[{run_name}] {summarize('rl_loss', rl)}")
if len(invalid) > 0:
    print(f"[{run_name}] {summarize('grpo_invalid_batch', invalid)}")
else:
    print(f"[{run_name}] grpo_invalid_batch: no-data")
PY
done

echo ""
echo "All 3 runs finished."
