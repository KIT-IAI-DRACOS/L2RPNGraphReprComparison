#!/bin/bash

# Argmax rollouts of one policy (experiments/imitation_rollout.py) in parallel processes inside a running job.
#
# Per split, N_PROCS processes each take every N_PROCS-th chronic (process t: chronics t, t+N_PROCS, ...) and write
# <run_dir>/rollout/episodes_<split>env_<t>.csv, as the tasks of imitation_rollout_array.sh do. One IEEE-36
# chronic takes ~5-10 s of one core and ~1 GB per process.
#
# Usage (from the repo root, L2RPN env active, PYTHONPATH=src):
#   slurm/unicluster/parallel_rollouts.sh <run_dir> <agent> <split>...
#
# Environment:
#   N_PROCS       Processes per split (default 4)
#   CHRONICS_DIR  Directory holding the Grid2Op envs, e.g. a copy on the node-local SSD (default: the run config's)
#   LOG_DIR       One log per process: <LOG_DIR>/<agent>_<split>_<t>.log (default <run_dir>/rollout/logs)
#
# Exits non-zero if any process failed.

set -uo pipefail

USAGE="Usage: parallel_rollouts.sh <run_dir> <agent> <split>..."
RUN_DIR="${1:?$USAGE}"
AGENT="${2:?$USAGE}"
shift 2
if [[ $# -eq 0 ]]; then
    echo "$USAGE" >&2
    exit 2
fi
N_PROCS="${N_PROCS:-4}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/rollout/logs}"
mkdir -p "${LOG_DIR}"
if [[ -n "${CHRONICS_DIR:-}" && ! -d "${CHRONICS_DIR}" ]]; then
    # grid2op.make then silently falls back to its data dir (~/data_grid2op)
    echo "WARNING: CHRONICS_DIR=${CHRONICS_DIR} does not exist; Grid2Op reads its default data dir instead." >&2
fi

pids=()
names=()
for split in "$@"; do
    for ((task = 0; task < N_PROCS; task++)); do
        log="${LOG_DIR}/${AGENT}_${split}_${task}.log"
        # One thread per process: N_PROCS processes share the job's cores.
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/imitation_rollout.py --run-dir "${RUN_DIR}" \
            --env-split "${split}" --agent "${AGENT}" --task-id "${task}" --n-tasks "${N_PROCS}" \
            ${CHRONICS_DIR:+--chronics-dir "${CHRONICS_DIR}"} > "${log}" 2>&1 &
        pids+=("$!")
        names+=("${log}")
    done
done

failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "Rollout failed, see ${names[$i]}" >&2
        failed=$((failed + 1))
    fi
done
echo "Rollouts of ${AGENT} ($*): ${#pids[@]} processes, ${failed} failed"
exit $((failed > 0))
