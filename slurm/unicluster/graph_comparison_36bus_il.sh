#!/bin/bash

# 36-bus graph comparison via imitation learning (Junior pipeline).
#
# For each graph_comparison obs_space, runs the full Junior HPO chain:
#   1. convert → 2. HPO (4 GPU workers) → 3. summarize → 4. rollouts (argmax + top-k)
#
# Replicates the 14-bus PPO graph comparison on the 36-bus grid using IL only.
# Model: gcn_substation_readout (GCN + per-substation pooling) for 11 graph obs_spaces,
# RLlib's default MLP for the flat baseline. Dataset: hard labels (best survival in Junior2).
#
# Usage (from repo root):
#   # Step 1: submit all HPO chains (each ends with an argmax rollout)
#   slurm/unicluster/graph_comparison_36bus_il.sh
#
#   # Step 2: after all HPO jobs complete, submit top-k rollouts
#   slurm/unicluster/graph_comparison_36bus_il.sh --rollouts-only
#
# Environment:
#   DATASET        Tutor dataset (default: data/imitation/case36_k208_greedy_hard.npz)
#   OBS_SPACES     Space-separated list (default: all 11 graph_comparison obs_spaces)
#   TOP_KS         Space-separated top-k values for rollouts (default: 20 50)
#   PREFIX         Study name prefix (default: graph36)
#   N_WORKERS      GPU workers per study (default: 4)
#   SBATCH_TIME    Walltime per HPO worker (default: 06:00:00)
#   DRY_RUN=1      Print scripts instead of submitting

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"

# Export DRY_RUN so child scripts (pretrain_hpo.sh, imitation_rollout_array.sh) inherit it.
[[ -n "${DRY_RUN:-}" ]] && export DRY_RUN

ROLLOUTS_ONLY=0
if [[ "${1:-}" == "--rollouts-only" ]]; then
    ROLLOUTS_ONLY=1
    shift
fi

DATASET="${DATASET:-data/imitation/case36_k208_greedy_hard.npz}"
PREFIX="${PREFIX:-graph36}"
read -r -a TOP_KS <<< "${TOP_KS:-20 50}"
SBATCH_TIME="${SBATCH_TIME:-06:00:00}"

DEFAULT_OBS_SPACES="no_edges substation complete elements heterogeneous lodf ptdf zbus elements_lodf substation_ptdf substation_zbus flat"
read -r -a OBS_SPACES <<< "${OBS_SPACES:-$DEFAULT_OBS_SPACES}"

if [[ "$ROLLOUTS_ONLY" -eq 1 ]]; then
    echo "=== Submitting top-k rollouts for completed studies ==="
    for obs_space in "${OBS_SPACES[@]}"; do
        study="${PREFIX}_${obs_space}"
        study_dir="results/junior/${study}"
        best_dir="${study_dir}/best"

        if [[ ! -d "$best_dir" ]]; then
            echo "SKIP ${obs_space}: ${best_dir} not found (HPO not finished?)"
            continue
        fi

        for k in "${TOP_KS[@]}"; do
            LOG_DIR="results/junior/rollout_logs" \
                slurm/unicluster/imitation_rollout_array.sh \
                    "${best_dir}" val "${study}" \
                    --top-k "${k}"
            echo "  submitted top-${k} rollout for ${obs_space}"
        done
    done
    echo "=== Done ==="
    exit 0
fi

echo "=== 36-bus Graph Comparison (IL) ==="
echo "Dataset:    ${DATASET}"
echo "Obs spaces: ${OBS_SPACES[*]}"
echo "Prefix:     ${PREFIX}"
echo ""

for obs_space in "${OBS_SPACES[@]}"; do
    study="${PREFIX}_${obs_space}"
    echo "--- ${obs_space} (study: ${study}) ---"

    if [[ "$obs_space" == "flat" ]]; then
        # MLP baseline: different model and obs_space (FlatGraphObsSpace, not a graph Dict)
        CONFIG_NAME=pretrain_hpo_graph36_mlp \
        N_WORKERS="${N_WORKERS:-4}" \
        SBATCH_TIME="${SBATCH_TIME}" \
            slurm/unicluster/pretrain_hpo.sh \
                "${study}" \
                "${DATASET}"
    else
        CONFIG_NAME=pretrain_hpo_graph36 \
        N_WORKERS="${N_WORKERS:-4}" \
        SBATCH_TIME="${SBATCH_TIME}" \
            slurm/unicluster/pretrain_hpo.sh \
                "${study}" \
                "${DATASET}" \
                "obs_space=graph_comparison/${obs_space}"
    fi
    echo ""
done

echo "=== All HPO chains submitted ==="
echo "After all jobs complete, submit top-k rollouts with:"
echo "  slurm/unicluster/graph_comparison_36bus_il.sh --rollouts-only"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: results/junior/${PREFIX}_*/trials.csv"
