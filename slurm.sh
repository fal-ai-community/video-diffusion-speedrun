#!/usr/bin/env bash
set -euo pipefail

# TODO: replace me with torchx
# usage: srun --nodes=2 ./slurm.sh ...

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd -P)
cd "$root"

# activate venv
export VIRTUAL_ENV="$root/.venv"
source "$VIRTUAL_ENV/bin/activate"

## build torchrun cmd from slurm env:
RANK_ZERO_NODE=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | sort -h | head -n1)
RDZV_ENDPOINT="${RANK_ZERO_NODE}:29292"
torchrun_args=(
    "--nnodes=${SLURM_JOB_NUM_NODES}"
    "--rdzv-endpoint=${RDZV_ENDPOINT}"
    "--rdzv-id=${SLURM_JOB_ID:-99}"
    "--nproc-per-node=8"
    "--max-restarts=1"
    "--rdzv-backend=c10d"
)

export OMP_NUM_THREADS=32
exec torchrun "${torchrun_args[@]}" train.py "$@"
