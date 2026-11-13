#!/bin/bash

# Graph comparison on case14 (Table I of the paper) — 12 representations × 5 seeds, 100k timesteps each, bwUniCluster.
#
# Every representation observes the node features of CompleteGraphObsSpace (configs/rllib/obs_space/
# graph_comparison/), so the comparison varies the graph structure, not the information:
#   - GNNs: GCN with a per-substation readout (model=gcn_substation_readout)
#   - no_edges: the nodes and features of complete without edges (control for the structure)
#   - flat_rllib_mlp: RLlib's default MLP (model=mlp, 256-256-256) on the same features plus each node's bus
#     (obs_space graph_comparison/flat)
# Rewards are divided by the episode length (otherwise vf_clip_param zeroes the critic's gradient)
# and the PPO KL loss is off (kl_coeff=0).
#
# Usage (from the repo root): slurm/unicluster/compare_graph_obs_spaces_multiseed.sh
# Results: results/<date>_compare_graph_obs_spaces_IEEE14/<obs_space>/
#
# Environment:
#   EXPERIMENT_NAME  Results folder (default <today>_compare_graph_obs_spaces_IEEE14)
#   SEEDS          Seeds to train (default "0 1 2 3 4")
#   OBS_SPACES     Representations to train (default: all 12; flat_rllib_mlp = graph_comparison/flat with RLlib's default MLP)
#   N_WORKERS      Rollout workers per run (default 48)
#   SBATCH_TIME    Walltime per run (default 06:00:00; the August runs took 1.7–2.9 h on HoreKa)
#   DEVICE         gpu (default): one GPU for the PPO updates, on gpu_h100, gpu_a100_il or gpu_h100_vip1
#                  (not gpu_mi300: AMD GPUs and only 40 cores per node); cpu: CPU nodes (cpu_il, cpu)
#   DRY_RUN=1      Print the sbatch scripts instead of submitting

# Set EXPERIMENT_NAME to resubmit failed runs into an earlier submission's folder.
experiment_name="${EXPERIMENT_NAME:-$(date +%Y_%m_%d)_compare_graph_obs_spaces_IEEE14}"

read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4}"
read -r -a OBS_SPACES <<< "${OBS_SPACES:-complete heterogeneous no_edges substation ptdf zbus substation_ptdf substation_zbus elements lodf elements_lodf flat_rllib_mlp}"
N_WORKERS="${N_WORKERS:-48}"
SBATCH_TIME="${SBATCH_TIME:-06:00:00}"
DEVICE="${DEVICE:-gpu}"

# The rollout workers always run on CPU; with DEVICE=gpu the driver does the PPO updates on the GPU.
case "$DEVICE" in
    gpu) PARTITION="gpu_h100,gpu_a100_il,gpu_h100_vip1"; GRES_LINE="#SBATCH --gres=gpu:1"; DEVICE_OVERRIDES="rollouts.num_gpus=1" ;;
    cpu) PARTITION="cpu_il,cpu"; GRES_LINE=""; DEVICE_OVERRIDES="rollouts.num_gpus=0" ;;
    *) echo "DEVICE must be gpu or cpu, got '${DEVICE}'" >&2; exit 1 ;;
esac

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"

COMMON_OVERRIDES="training=ppo experiment.nb_timesteps=100000 \
rollouts.num_rollout_workers=${N_WORKERS} rollouts.num_gpus_per_learner_worker=0 ${DEVICE_OVERRIDES} \
experiment.post_training_evaluation.enabled=True reward.normalize_by_episode_length=true training.kl_coeff=0.0"
GNN_MODEL="model=gcn_substation_readout"
# flat_rllib_mlp: the flat observation with RLlib's default model (model=mlp, 256-256-256, shared policy/value
# trunk, output layers initialised with std 0.01).
RLLIB_MLP_MODEL="model=mlp"

for seed in "${SEEDS[@]}"; do
    for obs_space in "${OBS_SPACES[@]}"; do
        obs_config="$obs_space"
        if [[ "$obs_space" == "flat_rllib_mlp" ]]; then
            MODEL_ARGS="$RLLIB_MLP_MODEL"
            obs_config="flat"
        else
            MODEL_ARGS="$GNN_MODEL"
        fi

        OUT_DIR="results/${experiment_name}/${obs_space}/out"

        SCRIPT=$(cat <<EOF
#!/bin/bash -l
#SBATCH --export=NIL
#SBATCH --job-name=${obs_space}_${seed}
#SBATCH --output=${OUT_DIR}/${obs_space}_${seed}.%j.log
#SBATCH --error=${OUT_DIR}/${obs_space}_${seed}.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=${SBATCH_TIME}
#SBATCH --partition=${PARTITION}
${GRES_LINE}

export RAY_gcs_rpc_server_reconnect_timeout_s=300
module load devel/miniforge
eval "\$(conda shell.bash hook)"
conda activate L2RPN
export PYTHONPATH="\$(pwd)/src"
echo "Node: \$(hostname)  seed: ${seed}  obs_space: ${obs_space}  model: ${MODEL_ARGS}"

python experiments/train.py ${COMMON_OVERRIDES} ${MODEL_ARGS} \
    obs_space="graph_comparison/${obs_config}" \
    experiment.name="${experiment_name}/${obs_space}" \
    experiment.seed=${seed}
EOF
)
        if [[ -n "${DRY_RUN:-}" ]]; then
            echo "----- ${obs_space} seed ${seed}"
            echo "${SCRIPT}"
        else
            mkdir -p "$OUT_DIR"  # Slurm needs the log directory before the job starts.
            JOB_ID=$(sbatch --parsable <<< "${SCRIPT}")
            echo "seed=${seed} obs_space=${obs_space} -> job ${JOB_ID}"
        fi
    done
done
