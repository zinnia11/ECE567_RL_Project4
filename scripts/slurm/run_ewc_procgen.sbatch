#!/bin/bash
#SBATCH --job-name=ewc-procgen
#SBATCH --account=yafeng0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=144:00:00
#SBATCH --output=logs/%x-%j-%a.log
#SBATCH --error=logs/%x-%j-%a-E.log
#SBATCH --array=0-2
#
# Online EWC on Procgen — three seeds (#SBATCH --array must match length of SEEDS).
# Environment stack matches scripts/slurm/run_ewc_atari.sbatch:
#   module load python/3.9  ->  libpython3.9.so on LD_LIBRARY_PATH before venv.
#
# Optional env vars:
#   CONTINUAL_RL_LARGE    override large_file_path root (default: ${SCRATCH_PROJECT_ROOT}/continual_rl_data)
#   EWC_CONFIG            config path relative to continual_rl/, default configs/procgen/ewc_procgen.json
#   EWC_OUTPUT_DIR        fixed output dir (overrides per-seed default; avoid with multi-seed unless you know)
#   SCRATCH_PROJECT_ROOT  scratch base (default /scratch/yafeng_root/yafeng0/${USER})

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
CR="${REPO_ROOT}/continual_rl"
if [[ ! -f "${CR}/main.py" ]]; then
  echo "ERROR: continual_rl/main.py not found under REPO_ROOT=${REPO_ROOT}"
  exit 1
fi

module purge
module load python/3.9

source ~/repo/exp/cont_rl/bin/activate

mkdir -p logs

SCRATCH_PROJECT_ROOT="${SCRATCH_PROJECT_ROOT:-/scratch/yafeng_root/yafeng0/${USER}}"

if [[ -n "${CONTINUAL_RL_LARGE:-}" ]]; then
  LARGE_ROOT="${CONTINUAL_RL_LARGE}"
else
  LARGE_ROOT="${SCRATCH_PROJECT_ROOT}/continual_rl_data"
fi
mkdir -p "${LARGE_ROOT}"

SEEDS=(42 123 456)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[$TASK_ID]}"
export CONTINUAL_RL_SEED="${SEED}"

CONFIG_REL="${EWC_CONFIG:-configs/procgen/ewc_procgen.json}"
CONFIG_ABS="${CR}/${CONFIG_REL}"
if [[ ! -f "${CONFIG_ABS}" ]]; then
  echo "ERROR: config not found: ${CONFIG_ABS}"
  exit 1
fi

if [[ -n "${EWC_OUTPUT_DIR:-}" ]]; then
  OUT_DIR="${EWC_OUTPUT_DIR}"
else
  OUT_DIR="${SCRATCH_PROJECT_ROOT}/runs/ewc_procgen_${SLURM_JOB_ID:-local}_seed${SEED}_task${TASK_ID}"
fi
mkdir -p "${OUT_DIR}"
exec > >(tee -a "${OUT_DIR}/run.log") 2>&1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_JSON="$(mktemp /tmp/ewc_procgen_run_XXXXXX.json)"
export LARGE_ROOT CONFIG_ABS RUN_JSON
python3 - <<'PY'
import json, os
large = os.environ["LARGE_ROOT"]
cfg_path = os.environ["CONFIG_ABS"]
out_path = os.environ["RUN_JSON"]
with open(cfg_path, encoding="utf-8") as f:
    cfg = json.load(f)
for b in (cfg if isinstance(cfg, list) else [cfg]):
    if isinstance(b, dict) and "large_file_path" in b:
        b["large_file_path"] = large
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=4)
print("Wrote", out_path, "large_file_path=", large)
PY

cleanup() { rm -f "${RUN_JSON}"; }
trap cleanup EXIT

echo "=== EWC Procgen seed=${SEED} (array task ${TASK_ID}) ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "SCRATCH_PROJECT_ROOT=${SCRATCH_PROJECT_ROOT}"
echo "CONFIG=${CONFIG_ABS}"
echo "LARGE_ROOT=${LARGE_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "CONTINUAL_RL_SEED=${CONTINUAL_RL_SEED}"

cd "${CR}"
python main.py --config-file "${RUN_JSON}" --output-dir "${OUT_DIR}"

echo "=== Done ==="
