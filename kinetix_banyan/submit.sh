#!/bin/bash
# Submit one ppo_banyan run.  usage: kinetix_banyan/submit.sh <job-name> <hydra overrides...>
# e.g. kinetix_banyan/submit.sh d1solo-o1 d1_num_instances=1 misc.group=d1solo-v1 misc.extra_run_name=o1
set -euo pipefail
NAME="$1"; shift
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p logs
OVERRIDES="$*" sbatch --job-name="$NAME" kinetix_banyan/run_ppo.sh
