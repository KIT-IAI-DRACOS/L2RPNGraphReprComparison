#!/bin/bash

# GNN case14 — 5 seeds, 200k timesteps

experiment_name="$(date +%Y_%m_%d)_compare_graph_obs_spaces_IEEE14"
export experiment_name

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"

mkdir -p "results/${experiment_name}"

G2OP_ENV=l2rpn_case14_sandbox

BASE_ARGS="training=ppo model=gnn relation_awareness=disabled experiment.nb_timesteps=100000 rollouts.num_rollout_workers=48 experiment.post_training_evaluation.enabled=True"
BASE_ARGS_GPU="${BASE_ARGS} rollouts.num_gpus=1 rollouts.num_gpus_per_learner_worker=1 rollouts.num_learner_workers=1"

OBS_SPACES=(
    default
    elements
    elements_lodf
    heterogenous
    lodf
    ptdf
    substation
    substation_ptdf
    substation_zbus
    zbus
)

JOB_IDS=()

for seed in 0 1 2 3 4; do
    for obs_space in "${OBS_SPACES[@]}"; do

        # Slurm needs these directories to exist before it starts the job.
        OUT_DIR="results/${experiment_name}/${obs_space}/out"
        mkdir -p "$OUT_DIR"

        echo "Submitting: seed=${seed}, obs_space=${obs_space}"

        JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${obs_space}_${seed}
#SBATCH --output=${OUT_DIR}/${obs_space}_${seed}.%j.log
#SBATCH --error=${OUT_DIR}/${obs_space}_${seed}.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --partition=accelerated,accelerated-h100
#SBATCH --account=hk-project-pai00074

export RAY_gcs_rpc_server_reconnect_timeout_s=300
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
source /hkfs/home/project/hk-project-tacos/hw6998/miniforge3/etc/profile.d/conda.sh
conda activate L2RPN

echo "========================================"
echo "Job ID:       \$SLURM_JOB_ID"
echo "Node:         \$(hostname)"
echo "Seed:         ${seed}"
echo "Obs space:    ${obs_space}"
echo "Experiment:   ${experiment_name}"
echo "G2OP_ENV:     \${G2OP_ENV:-<not set>}"
echo "TMPDIR:       \$TMPDIR"
echo "========================================"

mkdir -p \$TMPDIR/data_grid2op
cp -r ~/data_grid2op/${G2OP_ENV}_train \$TMPDIR/data_grid2op/
cp -r ~/data_grid2op/${G2OP_ENV}_val   \$TMPDIR/data_grid2op/
echo "Chronics copied to \$TMPDIR/data_grid2op"

PYTHONPATH="\$(pwd)/src" python experiments/train.py \
    ${BASE_ARGS_GPU} \
    env.chronics_dir="\$TMPDIR/data_grid2op" \
    experiment.name="${experiment_name}/${obs_space}" \
    obs_space="${obs_space}" \
    experiment.seed=${seed}
EOF
)
        JOB_IDS+=("$JOB_ID")
        echo "  → Job ID: ${JOB_ID}"

    done
done

# ── Submit ablation with dependency on ALL training jobs ───────────────────────
DEPS=$(IFS=':'; echo "${JOB_IDS[*]}")
echo ""
echo "All training jobs submitted: ${JOB_IDS[*]}"
echo "Submitting ablation with dependency afterok:${DEPS}"

ABLATION_JOB=$(bash slurm/horeka/graph_ablation.sh "${experiment_name}" "afterok:${DEPS}" | grep -oP '\d+')
echo "Ablation job ID: ${ABLATION_JOB}"