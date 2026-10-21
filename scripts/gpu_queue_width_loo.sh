#!/usr/bin/env bash
# On a GPU box: the leave-circuits-out width test (docs/findings/width-test-loo-plan.md).
# v3 configuration with real width profiles from one fold of R_DEV (alternating order),
# seeds 0-3 per fold: 8 runs, in LANES parallel lanes on one GPU.
#
#   ssh <box> 'cd ~/apexzero && nohup bash scripts/gpu_queue_width_loo.sh [lanes] > runs/width-loo.log 2>&1 &'
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
LANES="${1:-2}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=$(python3 -c "print(round(0.9 / ${LANES}, 2))")
F1=$(~/.local/bin/uv run --no-sync python -c "import sys; sys.path.insert(0,'src'); from apexzero.tracks.corners import R_DEV; print(','.join(R_DEV[0::2]))")
F2=$(~/.local/bin/uv run --no-sync python -c "import sys; sys.path.insert(0,'src'); from apexzero.tracks.corners import R_DEV; print(','.join(R_DEV[1::2]))")

run() {
  local name="$1"; shift
  echo "[$(date '+%F %T')] start $name"
  if ~/.local/bin/uv run --no-sync python scripts/train_dr.py --steps 200e6 --name "$name" \
      --tilt-fraction 0.5 --tilt-alpha 2 --real-widths "$@" > "runs/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] done  $name"
  else
    echo "[$(date '+%F %T')] FAILED $name (see runs/$name.log)"
  fi
}

jobs=()
for s in 0 1 2 3; do
  jobs+=("dr-realw-loo-f1-s$s --seed $s --real-widths-from $F1")
  jobs+=("dr-realw-loo-f2-s$s --seed $s --real-widths-from $F2")
done
lane() {
  local k="$1"
  for ((i = k; i < ${#jobs[@]}; i += LANES)); do
    # shellcheck disable=SC2086
    run ${jobs[$i]}
  done
}
for ((k = 0; k < LANES; k++)); do lane "$k" & done
wait
echo "[$(date '+%F %T')] width leave-out runs finished"
