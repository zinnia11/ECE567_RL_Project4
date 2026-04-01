#!/bin/bash

#SBATCH --job-name=impala-minihack
#SBATCH --account=mcity_project_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/dev/null


LARGE_ROOT="/scratch/mcity_project_owned_root/mcity_project_owned1/xjsong/cora"
# Use the paper-default IMPALA hyperparameters for MiniHack reproduction.
CONFIG_ABS="/home/xjsong/ECE567_RL_Project4/continual_rl/configs/minihack/impala_minihack_paperdefaults.json"
OUT_DIR="./srun_log/${SLURM_JOB_NAME:-impala-minihack}_${SLURM_JOB_ID:-local}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUT_DIR}"
exec > >(tee -a "${OUT_DIR}/run.log") 2>&1

module purge
module load gcc/10.3.0

# Conda shell hook
if [ -f "/sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh" ]; then
    . "/sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh"
fi
conda activate cora

RUN_JSON="$(mktemp /tmp/impala_run_XXXXXX.json)"
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

echo "=== IMPALA MiniHack ==="
echo "CONFIG=${CONFIG_ABS}"
echo "LARGE_ROOT=${LARGE_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

python continual_rl/main.py --config-file "${RUN_JSON}" --output-dir "${OUT_DIR}"

echo "=== Done ==="
