#!/bin/bash
# Submits the rebuttal grid. Run from the repo root on the login node:
#   bash scripts/submit_rebuttal.sh            # everything
#   bash scripts/submit_rebuttal.sh ina        # one dataset
set -euo pipefail
mkdir -p logs

DATASETS=("${@:-cifar cub ina inr omni vtab}")
read -r -a DATASETS <<< "${DATASETS[*]}"

for ds in "${DATASETS[@]}"; do
  # Instrumented default + the router ablations + same-backbone entropy.
  for arm in base task classsum entropy; do
    sbatch --job-name="prism-$ds-$arm" scripts/run_arm.sh "$ds" "$arm"
  done
done

# Oracle and task-order runs: the datasets the paper discusses most.
for ds in ina inr cifar; do
  sbatch --job-name="prism-$ds-oracle" scripts/run_arm.sh "$ds" oracle
  for arm in perm1 perm2 perm3; do
    sbatch --job-name="prism-$ds-$arm" scripts/run_arm.sh "$ds" "$arm"
  done
done
