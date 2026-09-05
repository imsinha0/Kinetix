#!/bin/bash -l
#SBATCH --job-name=banyan-liptune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --account=kempner_gershman_lab
#SBATCH --partition=kempner_h100
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs outputs/gate
.venv/bin/python -m kinetix_banyan.lip_tune --lip "${LIP:-0.40}" --out "outputs/gate/lip_tune_${LIP:-0.40}.json"
echo "liptune complete"
