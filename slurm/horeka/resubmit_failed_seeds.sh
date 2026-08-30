#!/bin/bash

# Resubmit failed seeds from the graph obs space comparison and add MLP baseline.
#
# Group 1 — zbus NaN failures (fixed: 1/(1+|Zbus|) formula, bounded in (0,1]):
#   substation_zbus seeds 2,3
#   zbus          seeds 2,3
#
# Group 2 — MLP baseline (flat obs space, MLP-PPO), seeds 0-4.

experiment_name="2026_08_17_compare_graph_obs_spaces_IEEE14"
export experiment_name

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"

G2OP_ENV=l2rpn_case14_sandbox

BASE_ARGS="training=ppo relation_awareness=disabled experiment.nb_timesteps=100000 rollouts.num_rollout_workers=48 experiment.post_training_evaluation.enabled=True"
BASE_ARGS_GPU="${BASE_ARGS} rollouts.num_gpus=1 rollouts.num_gpus_per_learner_worker=1 rollouts.num_learner_workers=1"

submit_job() {
    local obs_space="$1"
    local seed="$2"
    local extra_args="$3"   # additional hydra overrides (e.g. model=mlp)
    local dir_name="$4"     # subdirectory under experiment_name (defaults to obs_space)
    dir_name="${dir_name:-$obs_space}"

    OUT_DIR="results/${experiment_name}/${dir_name}/out"
    mkdir -p "$OUT_DIR"

    echo "Submitting: dir=${dir_name}, obs_space=${obs_space}, seed=${seed}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${dir_name}_${seed}
#SBATCH --output=${OUT_DIR}/${dir_name}_${seed}.%j.log
#SBATCH --error=${OUT_DIR}/${dir_name}_${seed}.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
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
echo "Dir:          ${dir_name}"
echo "Experiment:   ${experiment_name}"
echo "TMPDIR:       \$TMPDIR"
echo "========================================"

mkdir -p \$TMPDIR/data_grid2op
cp -r ~/data_grid2op/${G2OP_ENV}_train \$TMPDIR/data_grid2op/
cp -r ~/data_grid2op/${G2OP_ENV}_val   \$TMPDIR/data_grid2op/
echo "Chronics copied to \$TMPDIR/data_grid2op"

PYTHONPATH="\$(pwd)/src" python experiments/train.py \
    ${BASE_ARGS_GPU} \
    ${extra_args} \
    env.chronics_dir="\$TMPDIR/data_grid2op" \
    experiment.name="${experiment_name}/${dir_name}" \
    obs_space="${obs_space}" \
    experiment.seed=${seed}
EOF
}

# --- Group 1: zbus NaN failures ---
for seed in 2 3; do
    submit_job "substation_zbus" "$seed" "model=gnn"
    submit_job "zbus"            "$seed" "model=gnn"
done

# --- Group 2: MLP baseline (flat obs space) ---
for seed in 0 1 2 3 4; do
    submit_job "flat" "$seed" "model=mlp" "mlp"
done
