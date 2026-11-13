#!/bin/bash

# 36-bus graph comparison with RL fine-tuning — BWUniCluster.
#
# For each graph representation, the best Junior HPO trial of graph_comparison_36bus_il.sh is retrained
# with new seeds, each Junior is fine-tuned with PPO (Senior with critic warm-up), and Junior and Senior are
# rolled out greedily (argmax) on the val and test envs.
#
# Jobs per obs_space and seed (the rollouts run inside these jobs, slurm/unicluster/parallel_rollouts.sh):
#   junior  GPU: pretrain_policy.py with results/junior/<PREFIX>_<obs>/best/config.yaml, only the seed changed,
#           then the Junior's rollouts. With JUNIOR_EXPERIMENT, an existing Junior of that experiment is copied
#           instead (with its rollouts), and there is no Junior job.
#     └─> senior  one train.py run per RL_DEVICES entry, all from the Junior's weights (default: CPU and GPU
#                 in parallel). The first to finish claims the seed, exports its policy, cancels the others and
#                 rolls the policy out, so the GPU run gives speed and the CPU run is the fallback if the GPU run
#                 fails or never starts. Only the claim holder exports and rolls out.
#
# RL setup: every rollout worker plays the train chronics in its own random order (env.shuffle_scenarios);
# every evaluation plays all val chronics once with a fixed seed per chronic; the exported Senior is the kept
# checkpoint with the highest val survival (export_policy_weights.py --select best_val), so its val rollouts
# are slightly optimistic and test is the unbiased split.
#
# The RL model is built from the representation's base configs (model=gcn_substation_readout, model=mlp for
# flat; obs_space=graph_comparison/<obs>) plus every model/obs_space value of the Junior's config
# (experiments/architecture_overrides.py, dropout 0), so ImitationInitCallback can load the weights strictly.
#
# Node-local SSD ($TMPDIR, deleted at job end): training and rollouts read the chronics from a copy there and
# Juniors write their conversion cache there. Results stay on the shared file system (the CPU/GPU race needs
# them visible). The exported Senior config has env.chronics_dir=null (the copy is gone after the job).
#
# Layout (results/senior/<experiment>/):
#   <obs>/junior_s<seed>/                 pretrain_policy.py run dir: policy_weights.pt, config.yaml, rollout/
#                                         (episodes_<split>env_<t>.csv, one per rollout process)
#   <obs>/senior_s<seed>_<cpu|gpu>/       train.py output: Hydra config, Tune trial (checkpoints, analysis/)
#   <obs>/eval/senior_s<seed>/            exported Senior policy: policy_weights.pt, config.yaml, rollout/,
#                                         trained_on.txt (cpu or gpu), source_checkpoint.txt,
#                                         checkpoint_selection.json (iteration and val survival of the export)
#   <obs>/eval/senior_s<seed>.claim/      created by the Senior that exported first
#   <obs>/eval/senior_s<seed>.jobs        job ids of this seed's Seniors, for the winner to cancel the others
#   out/                                  SLURM logs; out/rollouts/: one log per rollout process
#
# Usage (from the repo root with the L2RPN env active; the architecture is read on the login node):
#   slurm/unicluster/graph_comparison_36bus_senior.sh
#
# Environment:
#   EXPERIMENT_NAME      Results folder (default <today>_IEEE36_graph_comparison_senior)
#   OBS_SPACES           Representations (default: all 12 of graph_comparison_36bus_il.sh)
#   JUNIOR_EXPERIMENT    Reuse the Juniors of results/senior/<JUNIOR_EXPERIMENT>/ where they exist (copied,
#                        no Junior job); missing ones are trained
#   SEEDS                Seeds of Junior and Senior (default "0 1 2 3 4")
#   PREFIX               HPO study prefix (default graph36)
#   RL_DEVICES           Seniors per seed, raced (default "cpu gpu"; "cpu" or "gpu" alone runs one)
#   NB_TIMESTEPS         RL agent steps per Senior (default 200000)
#   N_WORKERS            RL rollout workers (default 24; the PPO updates, not sampling, dominate the run time)
#   WARMUP_ITERS         Critic warm-up iterations (default 10)
#   RL_CPUS, RL_MEM      Cores and memory per Senior (default 40, 150G: Ray uses 33 cores, the CPU Seniors of
#                        2026_10_26 used 68 GB; the larger graphs are unmeasured)
#   G2OP_ENV             Grid2Op env copied to $TMPDIR (default l2rpn_wcci_2020; its _train/_val/_test splits)
#   EVAL_SPLITS          Grid2Op envs of the rollouts (default "val test")
#   ROLLOUT_PROCS        Rollout processes per split, in Junior and Senior jobs (default 2: 72 chronics at ~5-10 s
#                        each, ~10 min). Keep it equal for all agents and to the reused Juniors' rollouts (2 tasks):
#                        the converter's running normalizer makes results depend slightly on the chronic split.
#   SBATCH_TIME_IL       Walltime per Junior incl. rollouts (default 01:30:00; the best HPO trials trained 2-10 min)
#   SBATCH_TIME_RL_CPU   Walltime per CPU Senior (default 48:00:00; the CPU Seniors of 2026_10_26 took 22-37 h)
#   SBATCH_TIME_RL_GPU   Walltime per GPU Senior (default 24:00:00; must stay below gpu_a100_il's 48 h)
#   RAY_HEALTH_CHECK_FAILURES  Missed GCS health checks (one per ~13 s when unresponsive) before Ray declares the
#                        node dead (default 60, ~13 min; Ray's default 5 killed 7 CPU Seniors of
#                        2026_11_11_IEEE36_graph_comparison_senior_v2 after node stalls of 30-110 s)
#   DRY_RUN=1            Print the sbatch scripts instead of submitting
#
# Resubmitting a failed Senior (its seed has no claim yet): move its run dir <obs>/senior_s<seed>_<device>/ aside
# (export_policy_weights.py expects exactly one trial per run dir), then rerun with
# EXPERIMENT_NAME=<experiment> JUNIOR_EXPERIMENT=<experiment> OBS_SPACES=<obs> SEEDS=<seed> RL_DEVICES=<device>.
# The existing Junior is reused without a job, and the new job id is appended to the seed's .jobs file, so it
# still races the seed's other Seniors.

set -euo pipefail

EXPERIMENT="${EXPERIMENT_NAME:-$(date +%Y_%m_%d)_IEEE36_graph_comparison_senior}"
read -r -a OBS_SPACES <<< "${OBS_SPACES:-no_edges substation complete elements heterogeneous lodf ptdf zbus elements_lodf substation_ptdf substation_zbus flat}"
JUNIOR_EXPERIMENT="${JUNIOR_EXPERIMENT:-}"
G2OP_ENV="${G2OP_ENV:-l2rpn_wcci_2020}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4}"
read -r -a RL_DEVICES <<< "${RL_DEVICES:-cpu gpu}"
read -r -a EVAL_SPLITS <<< "${EVAL_SPLITS:-val test}"
PREFIX="${PREFIX:-graph36}"
NB_TIMESTEPS="${NB_TIMESTEPS:-200000}"
N_WORKERS="${N_WORKERS:-24}"
WARMUP_ITERS="${WARMUP_ITERS:-10}"
RL_CPUS="${RL_CPUS:-40}"
RL_MEM="${RL_MEM:-150G}"
ROLLOUT_PROCS="${ROLLOUT_PROCS:-2}"
SBATCH_TIME_IL="${SBATCH_TIME_IL:-01:30:00}"
RAY_HEALTH_CHECK_FAILURES="${RAY_HEALTH_CHECK_FAILURES:-60}"

declare -A RL_PARTITION=([cpu]="cpu_il,cpu" [gpu]="gpu_h100,gpu_h100_il,gpu_a100_il,gpu_h100_vip1")
declare -A RL_GRES=([cpu]="" [gpu]="#SBATCH --gres=gpu:1")
declare -A RL_NUM_GPUS=([cpu]=0 [gpu]=1)
declare -A RL_TIME=([cpu]="${SBATCH_TIME_RL_CPU:-48:00:00}" [gpu]="${SBATCH_TIME_RL_GPU:-24:00:00}")
for device in "${RL_DEVICES[@]}"; do
    if [[ -z "${RL_PARTITION[$device]:-}" ]]; then
        echo "RL_DEVICES entries must be cpu or gpu, got '${device}'" >&2
        exit 1
    fi
done

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"
OUT_DIR="results/senior/${EXPERIMENT}"
LOG_DIR="${OUT_DIR}/out"

# Fail before submitting anything if a study is missing, its architecture cannot be read, or a Senior of
# this experiment was already exported (its claim would make every new Senior cancel itself).
declare -A BEST_DIRS ARCH_OVERRIDES BASE_MODEL
for obs in "${OBS_SPACES[@]}"; do
    # flat: RLlib's default model on the flattened features (its Junior's config has custom_model: null)
    if [[ "${obs}" == "flat" ]]; then BASE_MODEL[$obs]=mlp; else BASE_MODEL[$obs]=gcn_substation_readout; fi
    best_dir="results/junior/${PREFIX}_${obs}/best"
    if [[ ! -f "${best_dir}/config.yaml" ]]; then
        echo "${best_dir}/config.yaml not found: run graph_comparison_36bus_il.sh first." >&2
        exit 1
    fi
    BEST_DIRS[$obs]="$(realpath "$best_dir")"
    if ! ARCH_OVERRIDES[$obs]=$(PYTHONPATH="$(pwd)/src" python experiments/architecture_overrides.py "${best_dir}/config.yaml"); then
        echo "Reading the architecture of ${best_dir} failed: is the L2RPN env active?" >&2
        exit 1
    fi
    for seed in "${SEEDS[@]}"; do
        if [[ -d "${OUT_DIR}/${obs}/eval/senior_s${seed}.claim" ]]; then
            echo "${OUT_DIR}/${obs}/eval/senior_s${seed}.claim exists: that Senior was already exported." >&2
            exit 1
        fi
    done
done

# Reward 1/max_step per step: value targets stay in [0, 1], so vf_clip_param does not zero the critic's gradient.
# kl_coeff=0: RLlib's old API stack ignores use_kl_loss=false. freeze_batch_norm: RLlib samples in eval mode
# but computes the loss in train mode, so with batch statistics sampling and loss would use different policies.
# Checkpoints at every evaluation (evaluation.evaluation_interval=10), Tune keeps the 3 best by val survival.
COMMON_RL="training=ppo env=case36 imitation=default \
reward=constant reward.normalize_by_episode_length=true training.kl_coeff=0.0 imitation.freeze_batch_norm=true \
imitation.critic_warmup_iters=${WARMUP_ITERS} experiment.nb_timesteps=${NB_TIMESTEPS} \
rollouts.num_rollout_workers=${N_WORKERS} rollouts.num_gpus_per_learner_worker=0 \
env.shuffle_scenarios=true evaluation.evaluation_interval=10 experiment.checkpoint_freq=10 \
optimization.score_metric=evaluation/custom_metrics/grid2op_end_mean env.chronics_dir=\$TMPDIR/data_grid2op"

# copy_chronics <split>... -> job script lines copying these Grid2Op envs to the node-local SSD
# (env.chronics_dir above and the rollouts' CHRONICS_DIR point there).
copy_chronics() {
    echo "mkdir -p \$TMPDIR/data_grid2op
for split in $*; do cp -r ~/data_grid2op/${G2OP_ENV}_\$split \$TMPDIR/data_grid2op/; done
echo \"Chronics ($*) copied to \$TMPDIR/data_grid2op\""
}

# rollout_lines <run dir> <agent> <processes per split> -> job script line rolling the policy out on every split
rollout_lines() {
    echo "CHRONICS_DIR=\$TMPDIR/data_grid2op N_PROCS=$3 LOG_DIR=${LOG_DIR}/rollouts \\
    slurm/unicluster/parallel_rollouts.sh $1 $2 ${EVAL_SPLITS[*]}"
}

ENV_SETUP='module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate L2RPN
export PYTHONPATH="$(pwd)/src"
echo "Node: $(hostname)"'

# submit <job id to wait for (afterok) or ""> <script on stdin> -> prints the job id
submit() {
    local dependency="$1" script
    script="$(cat)"
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "----- sbatch ${dependency:+--dependency=afterok:${dependency}}" >&2
        echo "${script}" >&2
        echo "DRY_RUN"
        return
    fi
    sbatch --parsable ${dependency:+--dependency=afterok:${dependency}} <<< "${script}"
}

echo "Experiment: ${OUT_DIR}"
echo "Obs spaces: ${OBS_SPACES[*]}   seeds: ${SEEDS[*]}   Seniors: ${RL_DEVICES[*]}"
if [[ -z "${DRY_RUN:-}" ]]; then
    mkdir -p "${LOG_DIR}/rollouts"
fi

for obs in "${OBS_SPACES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        junior_dir="${OUT_DIR}/${obs}/junior_s${seed}"
        senior_eval="${OUT_DIR}/${obs}/eval/senior_s${seed}"
        claim="${senior_eval}.claim"
        jobs_file="${senior_eval}.jobs"

        reused="results/senior/${JUNIOR_EXPERIMENT}/${obs}/junior_s${seed}"
        if [[ -n "${JUNIOR_EXPERIMENT}" && -f "${reused}/policy_weights.pt" ]]; then
            if [[ -z "${DRY_RUN:-}" && ! -e "${junior_dir}" ]]; then
                mkdir -p "$(dirname "${junior_dir}")"
                cp -r "${reused}" "${junior_dir}"
                echo "${reused}" > "${junior_dir}/source_run_dir.txt"
            fi
            echo "${obs} s${seed}: junior reused from ${reused}"
            IL_JOB=""
        else
        IL_JOB=$(submit "" <<EOF
#!/bin/bash -l
#SBATCH --export=NIL
#SBATCH --job-name=junior_${obs}_s${seed}
#SBATCH --output=${LOG_DIR}/junior_${obs}_s${seed}.%j.log
#SBATCH --error=${LOG_DIR}/junior_${obs}_s${seed}.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=${SBATCH_TIME_IL}
#SBATCH --partition=gpu_h100,gpu_h100_il,gpu_a100_il,gpu_h100_vip1
${ENV_SETUP}
set -e
python experiments/pretrain_policy.py --config-path "${BEST_DIRS[$obs]}" --config-name config \
    experiment.seed=${seed} imitation.device=cuda imitation.out_dir=${junior_dir} hydra.run.dir=${junior_dir}/hydra \
    imitation.cache_dir=\$TMPDIR/imitation_cache
# A failed rollout must not fail the job: the Seniors start only after it succeeded (afterok).
if ! (
$(copy_chronics "${EVAL_SPLITS[@]}")
$(rollout_lines "${junior_dir}" "junior_${obs}_s${seed}" "${ROLLOUT_PROCS}")
); then
    echo "WARNING: Junior rollouts failed (logs: ${LOG_DIR}/rollouts/junior_${obs}_s${seed}_*); the Seniors start anyway." >&2
fi
EOF
)
        echo "${obs} s${seed}: junior ${IL_JOB} (with rollouts)"
        fi

        RL_JOBS=()
        for device in "${RL_DEVICES[@]}"; do
            senior_dir="${obs}/senior_s${seed}_${device}"
            # The claim (mkdir is atomic on the shared filesystem) decides which Senior exports and rolls out.
            # The loser cancels itself before exporting, so two Seniors never write to the same eval dir.
            RL_JOBS+=("$(submit "${IL_JOB}" <<EOF
#!/bin/bash -l
#SBATCH --export=NIL
#SBATCH --job-name=senior_${device}_${obs}_s${seed}
#SBATCH --output=${LOG_DIR}/senior_${device}_${obs}_s${seed}.%j.log
#SBATCH --error=${LOG_DIR}/senior_${device}_${obs}_s${seed}.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${RL_CPUS}
#SBATCH --mem=${RL_MEM}
#SBATCH --time=${RL_TIME[$device]}
#SBATCH --partition=${RL_PARTITION[$device]}
${RL_GRES[$device]}
export RAY_gcs_rpc_server_reconnect_timeout_s=300
export RAY_health_check_failure_threshold=${RAY_HEALTH_CHECK_FAILURES}
${ENV_SETUP}
set -e
$(copy_chronics train val test)
python experiments/train.py ${COMMON_RL} rollouts.num_gpus=${RL_NUM_GPUS[$device]} \
    model=${BASE_MODEL[$obs]} obs_space=graph_comparison/${obs} ${ARCH_OVERRIDES[$obs]} \
    imitation.weights_path=${junior_dir}/policy_weights.pt \
    experiment.seed=${seed} experiment.name=senior/${EXPERIMENT}/${senior_dir}
mkdir -p ${OUT_DIR}/${obs}/eval
if ! mkdir ${claim} 2>/dev/null; then
    echo "Another Senior of this seed was exported first: cancelling this job."
    scancel \$SLURM_JOB_ID || true
    sleep 120
    exit 1
fi
if ! python experiments/export_policy_weights.py --experiment-dir ${OUT_DIR}/${senior_dir} --out ${senior_eval} --select best_val; then
    rmdir ${claim}
    exit 1
fi
echo ${device} > ${senior_eval}/trained_on.txt
for job in \$(cat ${jobs_file}); do
    if [[ "\$job" != "\$SLURM_JOB_ID" ]]; then
        scancel "\$job" || true
    fi
done
$(rollout_lines "${senior_eval}" "senior_${obs}_s${seed}" "${ROLLOUT_PROCS}")
EOF
)")
        done
        if [[ -z "${DRY_RUN:-}" ]]; then
            mkdir -p "${OUT_DIR}/${obs}/eval"
            # Append: a resubmitted Senior must still cancel (and be cancelled by) the seed's earlier Seniors.
            echo $(cat "${jobs_file}" 2>/dev/null) "${RL_JOBS[*]}" > "${jobs_file}"
        fi
        echo "${obs} s${seed}: seniors ${RL_JOBS[*]} (${RL_DEVICES[*]}, with rollouts)"
    done
done

echo "Episode tables: ${OUT_DIR}/<obs>/{junior_s<seed>,eval/senior_s<seed>}/rollout/episodes_{valenv,testenv}_*.csv"
