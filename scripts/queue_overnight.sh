#!/usr/bin/env bash
# Run training jobs one after another (the Mac's CPU only fits one heavy job at a time;
# running them concurrently is slower overall). Each job logs to runs/<name>.log.
# Usage: bash scripts/queue_overnight.sh
set -u
cd "$(dirname "$0")/.."

run() {
  local name="$1"; shift
  echo "[$(date '+%F %T')] start $name: $*"
  if uv run python "$@" > "runs/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] done  $name"
  else
    echo "[$(date '+%F %T')] FAILED $name (see runs/$name.log)"
  fi
}

mkdir -p runs
# K4 gate: two more specialist seeds (seed 1 is runs/hockenheim-specialist-mlp-200M).
run hockenheim-specialist-mlp-200M-s2 scripts/train.py --tracks Hockenheim --steps 200e6 \
    --name hockenheim-specialist-mlp-200M-s2 --network mlp --seed 2 --log-every 80
run hockenheim-specialist-mlp-200M-s3 scripts/train.py --tracks Hockenheim --steps 200e6 \
    --name hockenheim-specialist-mlp-200M-s3 --network mlp --seed 3 --log-every 80
# DR v3: half the strips from a stronger hairpin tilt (weights (1/r_min)^2).
run dr-pilot-v3-tilt50a2 scripts/train_dr.py --steps 200e6 --name dr-pilot-v3-tilt50a2 \
    --tilt-fraction 0.5 --tilt-alpha 2
# First racing curriculum pilot: PLR-perp scored by MaxMC on a 1,000-level pool.
run ued-plr-maxmc-pilot scripts/train_ued.py --arm plr-maxmc --steps 200e6 \
    --name ued-plr-maxmc-pilot --pool-size 1000 --num-steps 64
echo "[$(date '+%F %T')] queue finished"
