#!/usr/bin/env bash
set -euo pipefail

# ===== Fixed defaults (can be overridden by args) =====
LATENTCHEM_ROOT="/home/haoqian/Data/Molecule/Molecule/Baselines/LatentChem"
CKPT_DIR="/home/haoqian/Data/Molecule/Molecule/Baselines/LatentChem/checkpoints/124-taskthinker-bioupdater/stage4"
DATA_PATH="/home/haoqian/Data/Molecule/Datasets/Test/01_mol_edit"
OUTPUT_DIR="/home/haoqian/Data/Molecule/Latent/eval_outputs/latentchem_moledit"
GPU_ID="1"

BATCH_SIZE=4
MAX_NEW_TOKENS=1024
TEMPERATURE=0.7
TOP_P=0.9
NUM_RETURN_SEQUENCES=1
MAX_SEQ_LENGTH=8192
MAX_TEST_SAMPLES=""
PYTHON_BIN="/home/haoqian/Data/miniconda3/envs/latentchem_dev/bin/python"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --data-path)
      DATA_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --ckpt-dir)
      CKPT_DIR="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top-p)
      TOP_P="$2"
      shift 2
      ;;
    --max-test-samples)
      MAX_TEST_SAMPLES="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

LORA_PATH="${CKPT_DIR}/lora_weights"
PROJECTOR_PATH="${CKPT_DIR}/mm_projector.pt"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="latentchem_moledit_stage4_${TIMESTAMP}"
RAW_RESULTS_PATH="${OUTPUT_DIR}/raw/${RUN_NAME}.json"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${OUTPUT_DIR}/raw" "${LOG_DIR}"

if [[ ! -f "${LATENTCHEM_ROOT}/code_train_sft/inference.py" ]]; then
  echo "ERROR: inference.py not found in ${LATENTCHEM_ROOT}/code_train_sft"
  exit 2
fi
if [[ ! -d "${LORA_PATH}" ]]; then
  echo "ERROR: LoRA weights dir not found: ${LORA_PATH}"
  exit 3
fi
if [[ ! -f "${PROJECTOR_PATH}" ]]; then
  echo "ERROR: Projector ckpt not found: ${PROJECTOR_PATH}"
  exit 4
fi
if [[ ! -d "${DATA_PATH}" ]]; then
  echo "ERROR: DATA_PATH not found: ${DATA_PATH}"
  exit 5
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PYTHON_BIN not executable: ${PYTHON_BIN}"
  exit 6
fi

echo "========== LatentChem MolEdit Eval =========="
echo "LATENTCHEM_ROOT:    ${LATENTCHEM_ROOT}"
echo "CKPT_DIR:           ${CKPT_DIR}"
echo "DATA_PATH:          ${DATA_PATH}"
echo "GPU_ID:             ${GPU_ID}"
echo "RAW_RESULTS_PATH:   ${RAW_RESULTS_PATH}"
echo "RUN_NAME:           ${RUN_NAME}"
echo "============================================="

INFER_CMD=(
  "${PYTHON_BIN}" "${LATENTCHEM_ROOT}/code_train_sft/inference.py"
  --data_path "${DATA_PATH}"
  --include_tasks add delete sub
  --lora_path "${LORA_PATH}"
  --projector_path "${PROJECTOR_PATH}"
  --training_stage 3
  --batch_size "${BATCH_SIZE}"
  --num_return_sequences "${NUM_RETURN_SEQUENCES}"
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --is_both_latent false
  --is_biothinker false
  --is_taskthinker true
  --is_bioupdater true
  --is_biothinker_multi false
  --taskthinker_type mlp
  --is_taskthinker_multi false
  --is_bioupdater_multi false
  --is_biothinker_gating false
  --is_taskthinker_gating false
  --is_bioupdater_gating false
  --task_latent_max_steps 10
  --mask_task_latent_steps 0
  --mask_task_latent_noise_std 1.0
  --mask_task_latent_implementation noise
  --shuffle_task_latents false
  --bio_latent_lambda 0.0
  --bio_latent_alpha 0.5
  --max_cot_string_len 2048
  --inference_results_path "${RAW_RESULTS_PATH}"
  --proc_index 0
  --num_procs 1
  --gpu "${GPU_ID}"
)

if [[ -n "${MAX_TEST_SAMPLES}" ]]; then
  INFER_CMD+=( --max_test_samples "${MAX_TEST_SAMPLES}" )
fi

{
  echo "$(date +'%F %T') [INFO] Running inference..."
  echo "CMD: ${INFER_CMD[*]}"
} | tee "${LOG_DIR}/${RUN_NAME}.log"

env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${INFER_CMD[@]}" 2>&1 | tee -a "${LOG_DIR}/${RUN_NAME}.log"

echo "$(date +'%F %T') [INFO] Running MolEdit evaluation..." | tee -a "${LOG_DIR}/${RUN_NAME}.log"

(
  cd "${LATENTCHEM_ROOT}/eval"
  "${PYTHON_BIN}" - <<PY
import json
import os
from group_results import build_grouped_save_data
from ChemCoTBench.eval_moledit import evaluate_moledit_score

raw_results_path = r"${RAW_RESULTS_PATH}"
run_name = r"${RUN_NAME}"
gt_path = r"${DATA_PATH}"
output_dir = r"${OUTPUT_DIR}"
logs_dir = os.path.join(output_dir, "grouped_logs")
results_dir = os.path.join(output_dir, "results")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

build_grouped_save_data(raw_results_path, logs_dir, run_name)
score = evaluate_moledit_score(
    model_name=run_name,
    gt_path=gt_path,
    logs_dir=logs_dir,
    results_dir=results_dir,
    sample_count=1,
)

summary = {
    "run_name": run_name,
    "checkpoint": r"${CKPT_DIR}",
    "data_path": gt_path,
    "raw_results_path": raw_results_path,
    "score": score,
}
summary_path = os.path.join(output_dir, f"summary_{run_name}.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"Summary saved to: {summary_path}")
PY
) | tee -a "${LOG_DIR}/${RUN_NAME}.log"

echo "Done. Full log: ${LOG_DIR}/${RUN_NAME}.log"
