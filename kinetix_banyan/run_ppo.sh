#!/bin/bash -l
#SBATCH --job-name=kinetix-banyan-ppo
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=kempner_gershman_lab
#SBATCH --partition=kempner_h100

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
PY=".venv/bin/python"

# Hydra-style overrides come straight from OVERRIDES.
OVERRIDES="${OVERRIDES:-}"
echo "launching ppo_banyan on $(hostname) with overrides: ${OVERRIDES}"
# shellcheck disable=SC2086
"${PY}" experiments/ppo_banyan.py ${OVERRIDES}
echo "ppo_banyan complete"
