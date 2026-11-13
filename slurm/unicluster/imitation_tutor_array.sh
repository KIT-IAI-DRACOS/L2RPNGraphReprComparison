#!/bin/bash

# Imitation-data Tutor (src/imitation/tutor.py) as a SLURM array — BWUniCluster.
#
# Each array task runs a strided set of chronics and writes one .npz per chronic to
# results/tutor/<experiment>/experience/ with, per state above rho_threshold, obs.to_vect(),
# the simulated max rho + validity of all policy actions (do-nothing + action-space JSON), and the
# executed action. Chronics whose file already exists are skipped, so re-submitting the same
# experiment only redoes unfinished chronics. Survival per chronic is in each task's .err log.
#
# Chronics are assigned with a stride (task t gets lines t, t+N_TASKS, ...) so every task gets a
# mix of winter and summer chronics (winter chronics take much longer).
#
# Cost: ~0.4 s per recorded state on IEEE-36 with 209 candidates; ~15 s for a summer and ~1 min
# for a winter chronic (measured on a login node, 2026-10-07).
#
# Runs in the main `L2RPN` conda environment.
#
# Usage:
#   slurm/unicluster/imitation_tutor_array.sh <grid> <experiment_name> <action_space_json> <max_concurrent> <max_array_size>
#
# Environment:
#   SBATCH_TIME       Walltime per array task (default 04:00:00).
#   ARRAY_INDICES     Run only these array tasks (sbatch --array syntax), e.g. to redo tasks that hit
#                     the walltime. Pass the same max_array_size as the original submission.
#   TUTOR_ARGS        Extra arguments for run-tutor-chronics, e.g. "--do-nothing-margin 0.01".
#
# Example:
#   slurm/unicluster/imitation_tutor_array.sh case36 2026_10_07_imitation_case36_k208 \
#       data/action_spaces/l2rpn_wcci_2020/teacher_n1_k208.json 100 100

USAGE="Usage: imitation_tutor_array.sh <grid> <experiment_name> <action_space_json> <max_concurrent> <max_array_size>"
GRID="${1:?$USAGE}"
EXPERIMENT="${2:?$USAGE}"
ACTION_SPACE="${3:?$USAGE}"
MAX_CONCURRENT="${4:?$USAGE}"
MAX_ARRAY_SIZE="${5:?$USAGE}"

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"
if [[ ! -f "$ACTION_SPACE" ]]; then
    echo "Action space ${ACTION_SPACE} not found." >&2
    exit 1
fi

OUT_DIR="results/tutor/${EXPERIMENT}"
mkdir -p "$OUT_DIR"

module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate L2RPN
PYTHONPATH="$(pwd)/src" python experiments/imitation_data.py list-chronics --grid "${GRID}" \
    > "${OUT_DIR}/chronic_ids.txt"
N_CHRONICS=$(wc -l < "${OUT_DIR}/chronic_ids.txt")
if [[ "$N_CHRONICS" -eq 0 ]]; then
    echo "No chronics listed for grid ${GRID} — aborting." >&2
    exit 1
fi

CHUNK_SIZE=$(( (N_CHRONICS + MAX_ARRAY_SIZE - 1) / MAX_ARRAY_SIZE ))
N_TASKS=$(( (N_CHRONICS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
ARRAY_INDICES="${ARRAY_INDICES:-0-$((N_TASKS - 1))}"
echo "Sharding ${N_CHRONICS} chronics into ${N_TASKS} tasks of up to ${CHUNK_SIZE} chronics each (max ${MAX_CONCURRENT} concurrent), --time=${SBATCH_TIME}."

MAX_SLURM_ARRAY_SIZE=$(scontrol show config 2>/dev/null | awk -F'= *' '/^MaxArraySize/ {print $2}')
if [[ -n "$MAX_SLURM_ARRAY_SIZE" && "$N_TASKS" -gt "$MAX_SLURM_ARRAY_SIZE" ]]; then
    echo "N_TASKS=${N_TASKS} exceeds this cluster's MaxArraySize=${MAX_SLURM_ARRAY_SIZE}." >&2
    exit 1
fi

sbatch <<EOF
#!/bin/bash -l
# --export=NIL: a conda env active in the submitting shell would otherwise put its python
# first on PATH and break the 'conda' launcher (ModuleNotFoundError: No module named 'conda').
#SBATCH --export=NIL
#SBATCH --job-name=imitation_tutor_${GRID}
#SBATCH --output=${OUT_DIR}/task_%a.%j.log
#SBATCH --error=${OUT_DIR}/task_%a.%j.err
#SBATCH --array=${ARRAY_INDICES}%${MAX_CONCURRENT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=${SBATCH_TIME}
#SBATCH --mem=8G
#SBATCH --partition=cpu_il,cpu

module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN

mapfile -t CHRONIC_IDS < <(awk -v n=${N_TASKS} -v t=\$SLURM_ARRAY_TASK_ID '(NR - 1) % n == t' "${OUT_DIR}/chronic_ids.txt")

echo "========================================"
echo "Job ID:       \$SLURM_JOB_ID (array task \$SLURM_ARRAY_TASK_ID)"
echo "Node:         \$(hostname)"
echo "Grid:         ${GRID}"
echo "Action space: ${ACTION_SPACE}"
echo "Chronics:     \${CHRONIC_IDS[*]}"
echo "========================================"

PYTHONPATH="\$(pwd)/src" python experiments/imitation_data.py run-tutor-chronics \
    --grid "${GRID}" \
    --action-space "${ACTION_SPACE}" \
    --chronic-ids "\${CHRONIC_IDS[@]}" \
    --save-dir "${OUT_DIR}/experience" ${TUTOR_ARGS}
EOF
