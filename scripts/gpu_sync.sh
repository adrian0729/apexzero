#!/usr/bin/env bash
# Copy what training needs but git does not carry to a GPU box, and fetch runs back.
#
#   scripts/gpu_sync.sh <box>                 # push: R-dev track CSVs and R-dev OCP laps
#   scripts/gpu_sync.sh <box> pull <run>...   # pull: runs/<run> (checkpoints, logs, evals)
#
# Only R_DEV circuits are pushed: the lockbox (R_TEST) circuits never leave this machine,
# so nothing on the box can evaluate on them. OpenF1 and TUMFTM data are licensed for this
# use but are never committed.
set -euo pipefail
cd "$(dirname "$0")/.."
BOX="$1"; shift || true
if [ "${1:-push}" = "pull" ]; then
  shift
  for run in "$@"; do
    mkdir -p runs
    rsync -az "$BOX:apexzero/runs/$run" runs/
    rsync -az "$BOX:apexzero/runs/$run.log" runs/ 2>/dev/null || true
  done
  exit 0
fi
mapfile_rdev=$(uv run python -c "
import sys; sys.path.insert(0, 'src')
from apexzero.tracks.corners import R_DEV
print(' '.join(R_DEV))")
ssh "$BOX" 'mkdir -p apexzero/data/tracks/tumftm apexzero/outputs/ocp apexzero/runs'
for n in $mapfile_rdev; do
  scp -q "data/tracks/tumftm/$n.csv" "$BOX:apexzero/data/tracks/tumftm/"
  [ -f "outputs/ocp/$n.pkl" ] && scp -q "outputs/ocp/$n.pkl" "$BOX:apexzero/outputs/ocp/"
done
echo "pushed $(echo "$mapfile_rdev" | wc -w | tr -d ' ') R-dev circuits to $BOX"
