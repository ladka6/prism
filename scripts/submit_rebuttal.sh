#!/bin/bash
# Submit rebuttal runs: one 5-seed array job per (dataset, arm).
# Run from the repo root on the login node.
#
#   bash scripts/submit_rebuttal.sh "<datasets>" "<arms>"
#
#   bash scripts/submit_rebuttal.sh "cifar cub inr omni vtab" "base entropy"
#   bash scripts/submit_rebuttal.sh "inr cifar" "task classsum"
#   bash scripts/submit_rebuttal.sh "inr" "oracle perm1 perm2 perm3"
#
# Datasets: cifar cub ina inr omni vtab
# Arms:     base task classsum entropy oracle perm1 perm2 perm3
#
# Prints what it will submit and asks before sending anything.
set -euo pipefail

read -r -a DATASETS <<< "${1:?usage: submit_rebuttal.sh \"<datasets>\" \"<arms>\"}"
read -r -a ARMS <<< "${2:?usage: submit_rebuttal.sh \"<datasets>\" \"<arms>\"}"

for ds in "${DATASETS[@]}"; do
  [ -f "exps/prism_${ds}.json" ] || { echo "no config exps/prism_${ds}.json" >&2; exit 2; }
done
for arm in "${ARMS[@]}"; do
  case "$arm" in base|task|classsum|entropy|oracle|perm1|perm2|perm3) ;;
    *) echo "unknown arm $arm" >&2; exit 2 ;; esac
done

# Per-run wall-time limit. Snellius reserves limit x 128 SBU per run at submit
# time, so the 6h default in run_arm.sh would block submission long before the
# budget is actually spent. ina measured at ~9 min/seed; the others are
# estimated from tosca-eval's training times x1.3, with ~2.5x margin.
limit() {
  case "$1" in
    ina|vtab) echo 00:40:00 ;;
    cub)      echo 01:00:00 ;;
    inr)      echo 01:30:00 ;;
    cifar)    echo 02:30:00 ;;
    omni)     echo 04:00:00 ;;
  esac
}

minutes() { IFS=: read -r h m _ <<< "$1"; echo $(( 10#$h * 60 + 10#$m )); }

n=$(( ${#DATASETS[@]} * ${#ARMS[@]} ))
reserve=0
echo "Will submit $n array jobs ($(( n * 5 )) runs):"
for ds in "${DATASETS[@]}"; do
  echo "  $ds (limit $(limit "$ds")): ${ARMS[*]}"
  reserve=$(( reserve + ${#ARMS[@]} * 5 * $(minutes "$(limit "$ds")") * 128 / 60 ))
done
echo "Reserved at submit (worst case): ~$reserve SBU. Check 'accinfo' -> 'Budget left for submit'."
read -r -p "Submit? [y/N] " ok
[ "$ok" = "y" ] || { echo "nothing submitted"; exit 0; }

mkdir -p logs
for ds in "${DATASETS[@]}"; do
  for arm in "${ARMS[@]}"; do
    # Drop Snellius's four-line shared-node notice; keep errors and job ids.
    echo -n "$ds/$arm: "
    sbatch --time="$(limit "$ds")" --job-name="prism-$ds-$arm" scripts/run_arm.sh "$ds" "$arm" 2>&1 \
      | grep -v -e "shared node" -e "full node" -e "By default shared" -e "charged for" || true
  done
done
