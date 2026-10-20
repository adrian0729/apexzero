#!/usr/bin/env bash
# On the GPU box: seeds 1 and 2 of each DR generalist variant (v1, v2, v3), with exactly
# the seed-0 runs' arguments (runs/<name>/config.json), for the psychophysics study's
# replication (protocol ruling R18). Two lanes share the GPU; each lane is sequential.
#
#   ssh <box> 'cd ~/apexzero && nohup bash scripts/gpu_queue_replicates.sh > runs/replicates.log 2>&1 &'
#   scripts/gpu_sync.sh <box> pull dr-pilot-200M-s1 ...     # back on the Mac
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45      # two runs share the 46 GB card

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

lane 1 &
lane 2 &
wait
echo "[$(date '+%F %T')] replicates finished"
