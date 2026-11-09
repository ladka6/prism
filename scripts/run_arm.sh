#!/bin/bash
#SBATCH --job-name=prism-arm
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# One rebuttal arm of one benchmark, one seed per array task (1993..1997).
#
#   sbatch --job-name=prism-ina-entropy scripts/run_arm.sh ina entropy
#
# Arguments are positional: neither `VAR=x sbatch` nor --export reaches the job
# reliably on this cluster (same constraint as tosca-eval).
#
# dataset : cifar | cub | ina | inr | omni | vtab     (exps/prism_<dataset>.json)
# arm     : base      default router (class_max), instrumented
#           task      ridge over task labels (paper Eq. 4)
#           classsum  ridge over class labels, per-task SUM of class scores
#           entropy   per-sample lowest-entropy expert (same backbone/experts)
#           oracle    ground-truth task
#           perm1|perm2|perm3   same partition, tasks in a different order
set -euo pipefail

DATASET="${1:?dataset}"
ARM="${2:?arm}"

cd "${PRISM_DIR:-$HOME/prism}"
mkdir -p logs

# Use prism's own venv if it has the dependencies, else tosca-eval's (same
# stack: torch, timm, scipy, easydict).
DEPS="import torch, timm, scipy, easydict, tqdm"
for VENV in "$PWD/.venv" "$HOME/tosca-eval/.venv"; do
  if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
    if python -c "$DEPS" 2>/dev/null; then echo "using venv $VENV"; break; fi
    deactivate
  fi
done
python -c "$DEPS" || { echo "no venv with: $DEPS" >&2; exit 3; }

case "$ARM" in
  base)     SET=(--set prefix=base) ;;
  task)     SET=(--set prefix=task --set router_mode=task) ;;
  classsum) SET=(--set prefix=classsum --set router_mode=class_sum) ;;
  entropy)  SET=(--set prefix=entropy --set router_mode=entropy) ;;
  oracle)   SET=(--set prefix=oracle --set router_mode=oracle) ;;
  perm[1-3]) SET=(--set "prefix=$ARM" --set "task_perm_seed=${ARM#perm}") ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac

SEED=$((1993 + SLURM_ARRAY_TASK_ID))
python main.py --config "exps/prism_${DATASET}.json" --seed "$SEED" "${SET[@]}"
