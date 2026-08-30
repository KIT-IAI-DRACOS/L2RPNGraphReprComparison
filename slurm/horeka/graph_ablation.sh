#!/bin/bash

# Graph ablation evaluation — forced RL mode.
#
# Each method is evaluated twice (baseline + ablated) with activation
# threshold=0.0 so the RL agent acts at every step.  This makes the GNN's
# structural dependence visible in survival statistics.
#
# 10 methods × 2 conditions × 50 episodes, 20 parallel workers (one per condition).
# Results → experiments/survival/observation_spaces/ablation_gat_conv/
#
# Usage:
#   slurm/horeka/graph_ablation.sh <experiment_name> [dependency]
#
# Arguments:
#   experiment_name  Results sub-directory, e.g. 2026_08_30_compare_graph_obs_spaces_IEEE14
#   dependency       Optional sbatch --dependency string, e.g. afterok:123:456
#
# Example (standalone):
#   slurm/horeka/graph_ablation.sh 2026_08_30_compare_graph_obs_spaces_IEEE14
#
# Example (chained, called from another script):
#   slurm/horeka/graph_ablation.sh 2026_08_30_compare_graph_obs_spaces_IEEE14 afterok:101:102

EXPERIMENT="${1:?Usage: graph_ablation.sh <experiment_name> [dependency]}"
DEPENDENCY="${2:-}"

REPO_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
cd "$REPO_ROOT"

OUT_DIR="experiments/survival/observation_spaces/ablation_gat_conv/out"
mkdir -p "$OUT_DIR"

DEPENDENCY_FLAG=""
if [[ -n "$DEPENDENCY" ]]; then
    DEPENDENCY_FLAG="--dependency=${DEPENDENCY}"
fi

sbatch ${DEPENDENCY_FLAG} <<EOF
#!/bin/bash
#SBATCH --job-name=graph_ablation
#SBATCH --output=${OUT_DIR}/graph_ablation.%j.log
#SBATCH --error=${OUT_DIR}/graph_ablation.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --mem=300G
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-pai00074

source /hkfs/home/project/hk-project-tacos/hw6998/miniforge3/etc/profile.d/conda.sh
conda activate L2RPN

echo "========================================"
echo "Job ID:       \$SLURM_JOB_ID"
echo "Node:         \$(hostname)"
echo "Experiment:   ${EXPERIMENT}"
echo "========================================"

PYTHONPATH="\$(pwd)/src" python experiments/graph_ablation.py \
    --experiment "${EXPERIMENT}" \
    --workers 20 \
    --episodes 50 \
    --threshold 0.0
EOF
