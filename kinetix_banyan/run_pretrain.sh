#!/bin/bash -l
#SBATCH --job-name=kb-pretrain-grasp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=kempner_wcarvalho_lab
#SBATCH --partition=kempner_h100
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
OVERRIDES="${OVERRIDES:-}"
echo "pretraining on $(hostname): ${OVERRIDES}"
# shellcheck disable=SC2086
.venv/bin/python experiments/ppo.py ${OVERRIDES}
echo "pretrain complete"
