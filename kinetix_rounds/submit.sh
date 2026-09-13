#!/bin/bash
# usage: kinetix_rounds/submit.sh <job-name> <hydra overrides...>
set -euo pipefail
NAME="$1"; shift
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p logs
OVERRIDES="$*" sbatch --job-name="$NAME" kinetix_rounds/run_rounds.sh
