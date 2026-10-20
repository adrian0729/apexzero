#!/usr/bin/env bash
# On the GPU box: more seeds of each DR generalist variant (v1, v2, v3), with exactly the
# seed-0 runs' arguments (runs/<name>/config.json). Seeds 1 and 2 (the default) are the
# psychophysics study's replicates (protocol ruling R18); further seeds measure the seed
# spread of R-dev completion. One lane per seed; each lane is sequential.
#
#   ssh <box> 'cd ~/apexzero && nohup bash scripts/gpu_queue_replicates.sh [seeds...] > runs/replicates.log 2>&1 &'
#   scripts/gpu_sync.sh <box> pull dr-pilot-200M-s1 ...     # back on the Mac
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1 2)
# Lanes share the 46 GB card (about 430k steps/s each with two lanes on an L40S).
export XLA_PYTHON_CLIENT_MEM_FRACTION=$(python3 -c "print(round(0.9 / ${#SEEDS[@]}, 2))")

run() {
  local name="$1"; shift
  echo "[$(date '+%F %T')] start $name"
  if ~/.local/bin/uv run --no-sync python scripts/train_dr.py --steps 200e6 --name "$name" "$@" \
      > "runs/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] done  $name"
  else
    echo "[$(date '+%F %T')] FAILED $name (see runs/$name.log)"
  fi
}

lane() {
  local s="$1"
  run "dr-pilot-200M-s$s" --seed "$s"
  run "dr-pilot-v2-tilt30-s$s" --seed "$s" --tilt-fraction 0.3
  run "dr-pilot-v3-tilt50a2-s$s" --seed "$s" --tilt-fraction 0.5 --tilt-alpha 2
}

for s in "${SEEDS[@]}"; do lane "$s" & done
wait
echo "[$(date '+%F %T')] replicates finished"
