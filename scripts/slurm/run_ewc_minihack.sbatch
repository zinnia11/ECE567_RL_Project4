#!/bin/bash
#SBATCH --job-name=ewc-minihack
#SBATCH --account=yafeng0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/dev/null
#
# Online EWC on MiniHack — aligned with scripts/slurm/run_clear_minihack.sh (ARC + conda cora).
# Requires: NLE + MiniHack, repo layout with continual_rl/main.py under submit directory.
#
# Override before sbatch:
#   export LARGE_ROOT=/path/to/scratch/replay_root
#   export CONFIG_ABS=$PWD/continual_rl/configs/minihack/ewc_minihack.json
#   export OUT_DIR=/path/to/output   (optional; default ./srun_log/ewc-minihack_<jobid>)

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f "${REPO_ROOT}/continual_rl/main.py" ]]; then
  echo "ERROR: continual_rl/main.py not found under REPO_ROOT=${REPO_ROOT}"
  exit 1
fi

LARGE_ROOT="${LARGE_ROOT:-/scratch/yafeng_root/yafeng0/peisen}"
CONFIG_ABS="${CONFIG_ABS:-${REPO_ROOT}/continual_rl/configs/minihack/ewc_minihack.json}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/srun_log/${SLURM_JOB_NAME:-ewc-minihack}_${SLURM_JOB_ID:-local}}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUT_DIR}"
exec > >(tee -a "${OUT_DIR}/run.log") 2>&1

module purge
module load gcc/10.3.0

if [[ -f "/sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  . "/sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh"
fi
conda activate cora

if [[ ! -f "${CONFIG_ABS}" ]]; then
  echo "ERROR: config not found: ${CONFIG_ABS}"
  exit 1
fi

RUN_JSON="$(mktemp /tmp/ewc_minihack_run_XXXXXX.json)"
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

echo "=== EWC MiniHack ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "CONFIG=${CONFIG_ABS}"
echo "LARGE_ROOT=${LARGE_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

cd "${REPO_ROOT}"
python continual_rl/main.py --config-file "${RUN_JSON}" --output-dir "${OUT_DIR}"

echo "=== Done ==="
