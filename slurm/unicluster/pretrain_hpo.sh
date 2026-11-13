#!/bin/bash

# Optuna search over imitation pretraining settings for one dataset (experiments/pretrain_hpo.py,
# configs/rllib/pretrain_hpo_graph36.yaml) and survival of its best trial — BWUniCluster.
#
# Job chain, all writing to results/junior/<study_name>/:
#   1. convert    (CPU)  converts and caches the dataset's observations, writes config.yaml
#   2. workers    (GPU array, after 1) N_WORKERS workers share the study until it has hpo.n_trials trials
#   3. summarize  (CPU, after 2 ended, also on timeout) trials.csv + link best -> best trial
#   4. rollout    (CPU array, after 3) greedy policy of the best trial on the Grid2Op val env
#                 -> best/rollout/episodes_valenv_*.csv
# Re-submitting the same study resumes it: finished trials count towards the budget, trials cut off
# by the walltime are redone.
#
# Usage:
#   slurm/unicluster/pretrain_hpo.sh <study_name> <dataset.npz> [hydra overrides...]
#
# Environment:
#   N_WORKERS          Parallel GPU workers (default 4).
#   SBATCH_TIME        Walltime per worker (default 24:00:00).
#   N_ROLLOUT_TASKS    Rollout array tasks (default 8).
#   CONFIG_NAME        Hydra config of experiments/pretrain_hpo.py (default pretrain_hpo_graph36;
#                      pretrain_hpo_graph36_mlp for the MLP baseline).
#   DRY_RUN=1          Print the sbatch scripts instead of submitting.
#
# Example:
#   slurm/unicluster/pretrain_hpo.sh graph36_substation data/imitation/case36_k208_greedy_hard.npz obs_space=graph_comparison/substation
#   CONFIG_NAME=pretrain_hpo_graph36_mlp slurm/unicluster/pretrain_hpo.sh graph36_flat data/imitation/case36_k208_greedy_hard.npz

USAGE="Usage: pretrain_hpo.sh <study_name> <dataset.npz> [hydra overrides...]"
STUDY="${1:?$USAGE}"
DATASET="${2:?$USAGE}"
shift 2
REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"
if [[ ! -f "$DATASET" ]]; then
    echo "Dataset ${DATASET} not found." >&2
    exit 1
fi
N_WORKERS="${N_WORKERS:-4}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
N_ROLLOUT_TASKS="${N_ROLLOUT_TASKS:-8}"
STUDY_DIR="results/junior/${STUDY}"
LOG_DIR="${STUDY_DIR}/out"
mkdir -p "$LOG_DIR"

CONFIG_NAME="${CONFIG_NAME:-pretrain_hpo_graph36}"
OVERRIDES="--config-name ${CONFIG_NAME} experiment.name=${STUDY} imitation.dataset=${DATASET} $*"
ENV_SETUP='module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate L2RPN
export PYTHONPATH="$(pwd)/src"
echo "Node: $(hostname)"'

# submit <dependency type:id or ""> <script on stdin> -> prints the job id
submit() {
    local dependency="$1" script
    script="$(cat)"
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "----- sbatch ${dependency:+--dependency=${dependency}}" >&2
        echo "${script}" >&2
        echo "DRY_RUN"
        return
    fi
    sbatch --parsable ${dependency:+--dependency=${dependency}} <<< "${script}"
}

CONVERT_JOB=$(submit "" <<EOF
#!/bin/bash -l
# --export=NIL: see imitation_tutor_array.sh.
#SBATCH --export=NIL
#SBATCH --job-name=convert_${STUDY}
#SBATCH --output=${LOG_DIR}/convert.%j.log
#SBATCH --error=${LOG_DIR}/convert.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --partition=cpu_il,cpu
${ENV_SETUP}
python experiments/pretrain_hpo.py ${OVERRIDES} hpo.convert_only=true
EOF
)
echo "convert: ${CONVERT_JOB}"

WORKER_JOB=$(submit "afterok:${CONVERT_JOB}" <<EOF
#!/bin/bash -l
#SBATCH --export=NIL
#SBATCH --job-name=hpo_${STUDY}
#SBATCH --output=${LOG_DIR}/worker_%a.%A.log
#SBATCH --error=${LOG_DIR}/worker_%a.%A.err
#SBATCH --array=0-$((N_WORKERS - 1))
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=${SBATCH_TIME}
#SBATCH --partition=gpu_h100,gpu_h100_il,gpu_a100_il,gpu_h100_vip1
${ENV_SETUP}
python experiments/pretrain_hpo.py ${OVERRIDES} imitation.device=cuda hpo.worker_id=\$SLURM_ARRAY_TASK_ID
EOF
)
echo "workers: ${WORKER_JOB}"

SUMMARY_JOB=$(submit "afterany:${WORKER_JOB}" <<EOF
#!/bin/bash -l
#SBATCH --export=NIL
#SBATCH --job-name=summarize_${STUDY}
#SBATCH --output=${LOG_DIR}/summarize.%j.log
#SBATCH --error=${LOG_DIR}/summarize.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu_il,cpu
${ENV_SETUP}
python experiments/summarize_hpo.py ${STUDY_DIR}
EOF
)
echo "summarize: ${SUMMARY_JOB}"

ROLLOUT_JOB=$(DEPENDENCY="${SUMMARY_JOB}" N_TASKS="${N_ROLLOUT_TASKS}" \
    slurm/unicluster/imitation_rollout_array.sh "${STUDY_DIR}/best" val "${STUDY}")
echo "rollout (val env): ${ROLLOUT_JOB}"
echo "Results: ${STUDY_DIR}/trials.csv, ${STUDY_DIR}/best/, ${STUDY_DIR}/best/rollout/episodes_valenv_*.csv"
