#!/bin/bash
#SBATCH --job-name=ewc-procgen
#SBATCH --account=yafeng0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/dev/null
#SBATCH --array=0-2
#
# Online EWC on Procgen — three seeds via job array (edit SEEDS below).
# Aligns with scripts/slurm/run_clear_procgen.sh (conda cora, gcc, LARGE_ROOT inject).
#
# Override:
#   export LARGE_ROOT=/scratch/.../your_data
#   export CONFIG_ABS=$PWD/continual_rl/configs/procgen/ewc_procgen.json
#   SEEDS=(11 22 33) sbatch ...   # must set before sbatch if changing seeds
#
# Single-seed test (no array): sbatch --export=ALL --array=2 run_ewc_procgen.sbatch  OR unset array in script.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f "${REPO_ROOT}/continual_rl/main.py" ]]; then
  echo "ERROR: continual_rl/main.py not found under REPO_ROOT=${REPO_ROOT}"
  exit 1
fi

# Default three seeds (must match #SBATCH --array length)
SEEDS=(42 123 456)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[$TASK_ID]}"
export CONTINUAL_RL_SEED="${SEED}"

LARGE_ROOT="${LARGE_ROOT:-/scratch/yafeng_root/yafeng0/peisen}"
CONFIG_ABS="${CONFIG_ABS:-${REPO_ROOT}/continual_rl/configs/procgen/ewc_procgen.json}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/srun_log/ewc_procgen_${SLURM_JOB_ID}_seed${SEED}_task${TASK_ID}}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUT_DIR}"
exec > >(tee -a "${OUT_DIR}/run.log") 2>&1

module purge
module load gcc/10.3.0

source ~/repo/exp/cont_rl/bin/activate

if [[ ! -f "${CONFIG_ABS}" ]]; then
  echo "ERROR: config not found: ${CONFIG_ABS}"
  exit 1
fi

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
echo "CONFIG=${CONFIG_ABS}"
echo "LARGE_ROOT=${LARGE_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "CONTINUAL_RL_SEED=${CONTINUAL_RL_SEED}"

cd "${REPO_ROOT}"
python continual_rl/main.py --config-file "${RUN_JSON}" --output-dir "${OUT_DIR}"

echo "=== Done ==="
