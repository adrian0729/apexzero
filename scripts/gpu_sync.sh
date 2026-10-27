#!/usr/bin/env bash
# Copy what training needs but git does not carry to a GPU box, and fetch runs back.
#
#   scripts/gpu_sync.sh <box>                 # push: R-dev track CSVs and R-dev OCP laps
#   scripts/gpu_sync.sh <box> pull <run>...   # pull: runs/<run> (checkpoints, logs, evals)
#   scripts/gpu_sync.sh <box> pull-round <round> <runs-script> [options...]
#
# pull-round pulls every run of a round's run list (scripts/round_manifest.py; the options go to
# the runs script): each run's directory, log and first-attempt log (.log.1, if it was retried),
# and the round's manifests, queue logs and status files. With SHARD=i/K (scripts/gpu_round.sh's)
# it expects only that shard's runs on the box, so with K boxes pull each with its own SHARD. It
# pulls what is there, then exits 1 if any expected run directory or log is missing or the box
# has no manifest. round_manifest.py check is the full check.
#
# Only R_DEV circuits are pushed: the lockbox (R_TEST) circuits never leave this machine,
# so nothing on the box can evaluate on them. OpenF1 and TUMFTM data are licensed for this
# use but are never committed.
set -euo pipefail
cd "$(dirname "$0")/.."
BOX="$1"; shift || true
if [ "${1:-push}" = "pull-round" ]; then
  shift
  if [ $# -lt 2 ]; then
    echo "usage: scripts/gpu_sync.sh <box> pull-round <round> <runs-script> [options...]" >&2
    exit 1
  fi
  round="$1"; runs_script="$2"; shift 2
  shard="${SHARD:-0/1}"
  if [ -n "${PYTHON:-}" ]; then py=("$PYTHON"); else py=(uv run python); fi
  # round_manifest.py checks the round and run names (letters, digits and . _ + -), so they
  # are safe unquoted in the listing, which bash runs on the box (the patterns expand there).
  names=$("${py[@]}" scripts/round_manifest.py names --round "$round" \
    --runs-script "$runs_script" --shard "$shard" --runs-args "$@")
  wanted=()
  for n in $names; do wanted+=("$n" "$n.log" "$n.log.1"); done
  wanted+=("$round-manifest-*of*.json" "$round-queue-*of*.log" "$round-status-*of*.tsv")
  listing="cd apexzero/runs || { echo 'no apexzero/runs on the box' >&2; exit 1; }
for f in ${wanted[*]}; do [ -e \"\$f\" ] && echo \"\$f\"; done
true"
  present=$(printf '%s\n' "$listing" | ssh "$BOX" 'bash -s')
  mkdir -p runs
  pulled=0
  for f in $present; do
    rsync -az "$BOX:apexzero/runs/$f" runs/
    pulled=$((pulled + 1))
  done
  missing=()
  for n in $names; do
    [ -d "runs/$n" ] || missing+=("runs/$n")
    [ -f "runs/$n.log" ] || missing+=("runs/$n.log")
  done
  manifests=0
  for f in $present; do
    case "$f" in "$round"-manifest-*) manifests=$((manifests + 1)) ;; esac
  done
  echo "pulled $pulled files and run directories of round $round (shard $shard) from $BOX," \
    "$manifests manifest(s)"
  if [ "$manifests" -eq 0 ]; then
    echo "missing: no runs/$round-manifest-*of*.json on $BOX (has its queue finished?)" >&2
  fi
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "missing (${#missing[@]}): ${missing[*]}" >&2
  fi
  [ "$manifests" -gt 0 ] && [ "${#missing[@]}" -eq 0 ] || exit 1
  exit 0
fi
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
