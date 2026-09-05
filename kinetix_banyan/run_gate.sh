#!/bin/bash -l
#SBATCH --job-name=banyan-gate
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --account=kempner_gershman_lab
#SBATCH --partition=kempner_h100

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs outputs/gate
PY=".venv/bin/python"

DEPTH="${DEPTH:-1}"
TASKS="${TASKS:-8}"
EPISODES="${EPISODES:-2}"
POP="${POP:-512}"
ITERS="${ITERS:-16}"
SEGMENTS="${SEGMENTS:-24}"
HORIZON="${HORIZON:-192}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-}"

EXTRA_ARGS=()
if [[ -n "${MAX_TIMESTEPS}" ]]; then
  EXTRA_ARGS+=(--max-timesteps "${MAX_TIMESTEPS}")
fi

echo "gate depth=${DEPTH} tasks=${TASKS} episodes=${EPISODES} pop=${POP} iters=${ITERS} max_timesteps=${MAX_TIMESTEPS:-default} node=$(hostname)"
"${PY}" -m kinetix_banyan.gate_cem \
  --depth "${DEPTH}" --tasks "${TASKS}" --episodes "${EPISODES}" \
  --pop "${POP}" --iters "${ITERS}" --segments "${SEGMENTS}" --horizon "${HORIZON}" \
  "${EXTRA_ARGS[@]}" \
  --out "outputs/gate/gate_depth${DEPTH}.json"
echo "gate complete"
