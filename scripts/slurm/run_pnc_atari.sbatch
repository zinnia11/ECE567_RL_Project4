#!/bin/bash
#SBATCH --job-name=pnc-atari
#SBATCH --account=yafeng0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=144:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j-E.log
#


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

LARGE_ROOT="${CONTINUAL_RL_LARGE:-${SCRATCH:-${HOME}/scratch}/continual_rl_data}"
mkdir -p "${LARGE_ROOT}"

CONFIG_REL="${PNC_CONFIG:-configs/atari/pnc_atari.json}"
CONFIG_ABS="${CR}/${CONFIG_REL}"
if [[ ! -f "${CONFIG_ABS}" ]]; then
  echo "ERROR: config not found: ${CONFIG_ABS}"
  exit 1
fi

OUT_DIR="${PNC_OUTPUT_DIR:-${REPO_ROOT}/runs/pnc_atari_${SLURM_JOB_ID:-local}}"
mkdir -p "${OUT_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_JSON="$(mktemp /tmp/pnc_run_XXXXXX.json)"
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

echo "=== P&C Atari ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "CONFIG=${CONFIG_ABS}"
echo "LARGE_ROOT=${LARGE_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

cd "${CR}"
python main.py --config-file "${RUN_JSON}" --output-dir "${OUT_DIR}"

echo "=== Done ==="
