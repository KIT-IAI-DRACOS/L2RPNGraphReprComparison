#!/bin/bash

# Rollouts of a pretrained policy (experiments/imitation_rollout.py) as a CPU SLURM array — BWUniCluster.
#
# Task t of N_TASKS takes chronics t, t+N_TASKS, ... of the chosen env and writes
# <run_dir>/rollout/episodes_<tag>_<t>.csv (or into --out-dir, if passed as extra argument).
# One chronic of IEEE-36 takes ~30 s.
#
# Usage:
#   slurm/unicluster/imitation_rollout_array.sh <run_dir> <env_split> <agent> [extra imitation_rollout.py args...]
#
# Environment:
#   N_TASKS       Array tasks (default 8).
#   DEPENDENCY    Start after this job id finished successfully (sbatch afterok).
#   SBATCH_TIME   Walltime per task (default 04:00:00).
#   LOG_DIR       SLURM log directory (default results/junior/rollout_logs).
#   DRY_RUN=1     Print the sbatch script instead of submitting.
#
# Examples:
#   # Greedy policy of the best trial on the Grid2Op val env (144 chronics, unseen by the pipeline)
#   slurm/unicluster/imitation_rollout_array.sh results/junior/junior_hard/best val junior_hard
#   # Do-nothing reference; the run dir only provides the env config
#   slurm/unicluster/imitation_rollout_array.sh results/junior/junior_hard val do_nothing \
#       --do-nothing --out-dir results/junior/junior_do_nothing/rollout

USAGE="Usage: imitation_rollout_array.sh <run_dir> <env_split> <agent> [extra args...]"
RUN_DIR="${1:?$USAGE}"
ENV_SPLIT="${2:?$USAGE}"
AGENT="${3:?$USAGE}"
shift 3
REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"
N_TASKS="${N_TASKS:-8}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
LOG_DIR="${LOG_DIR:-results/junior/rollout_logs}"
mkdir -p "$LOG_DIR"

SCRIPT=$(cat <<EOF
#!/bin/bash -l
# --export=NIL: see imitation_tutor_array.sh.
#SBATCH --export=NIL
#SBATCH --job-name=rollout_${AGENT}
#SBATCH --output=${LOG_DIR}/${AGENT}_${ENV_SPLIT}_%a.%A.log
#SBATCH --error=${LOG_DIR}/${AGENT}_${ENV_SPLIT}_%a.%A.err
#SBATCH --array=0-$((N_TASKS - 1))
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=${SBATCH_TIME}
#SBATCH --partition=cpu_il,cpu

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN
echo "Node: \$(hostname)"

PYTHONPATH="\$(pwd)/src" python experiments/imitation_rollout.py --run-dir ${RUN_DIR} --env-split ${ENV_SPLIT} \
    --agent ${AGENT} --task-id \$SLURM_ARRAY_TASK_ID --n-tasks ${N_TASKS} $*
EOF
)

if [[ -n "${DRY_RUN:-}" ]]; then
    echo "----- sbatch ${DEPENDENCY:+--dependency=afterok:${DEPENDENCY}}" >&2
    echo "${SCRIPT}" >&2
    echo "DRY_RUN"
    exit 0
fi
sbatch --parsable ${DEPENDENCY:+--dependency=afterok:${DEPENDENCY}} <<< "${SCRIPT}"
