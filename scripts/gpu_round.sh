#!/usr/bin/env bash
# On a GPU box: one Phase 3 round from its run list, on every GPU of the box, or one shard of the
# round's seeds per box. Run from the repo root (it changes there itself) at the round's tag:
#
#   bash scripts/gpu_round.sh <round> <runs-script> [runs-script options...]
#
# The run list is scripts/<round>_runs.py: its runs(**opts) gives the runs in the
# pre-registered order, each {"name", "seed", "cmd", optional "config"}, and run as a script it
# prints them as JSON lines. scripts/round_manifest.py holds the contract and checks every
# entry before anything starts.
#
# Environment:
#   GPUS           GPUs to use, 0 to GPUS-1 (default: the number nvidia-smi lists; without
#                  nvidia-smi it must be set, as the tests do)
#   LANES_PER_GPU  runs at a time on each GPU (default 4, as the recorded rates assume)
#   SHARD          i/K (default 0/1). With K > 1 this box keeps the runs whose seed's index
#                  among the round's distinct seeds, in first-appearance order, is i mod K, so
#                  each box takes whole seeds.
#   DRY_RUN=1      print the schedule as TSV (lane, gpu, position, name) and exit, training nothing
#   BREV_TYPE      the box type, recorded in the manifest (default "unknown")
#   PYTHON         the interpreter (default: uv run --no-sync python, as gpu_setup.sh needs)
#
# Job j of the shard's list goes to lane j mod L, L = GPUS x LANES_PER_GPU. Lane l runs on GPU
# l mod GPUS, with CUDA_VISIBLE_DEVICES set to that GPU and XLA_PYTHON_CLIENT_MEM_FRACTION to
# 0.9 / LANES_PER_GPU, and runs its jobs in order. Each run logs to runs/<name>.log; a failed run
# is re-run once from scratch (its log kept as runs/<name>.log.1, its directory removed), as in
# gpu_queue_h4r.sh. The queue logs to runs/<round>-queue-<i>of<K>.log and each run's start and
# end to runs/<round>-status-<i>of<K>.tsv; at the end round_manifest.py write records
# runs/<round>-manifest-<i>of<K>.json. It refuses to start if any of its runs' directories or
# logs, or its shard's status file or manifest, exists already, or if the tracked tree is not
# clean (round_manifest.py check refuses such a round; DRY_RUN only notes it), and exits 1 if a
# run failed twice or the manifest was not written.
#
# One massedcompute_A6000_plusx4 box (4 GPUs x 4 lanes = 16 runs at a time), from the Mac, with
# the round's tag pushed (GW-3's commands, docs/prereg/gw3-width-round.md Section 8; a round's
# run-list options, such as VG-1's --phase, go after the runs script, the same on every box and
# in pull-round and check):
#   ssh <box> 'bash -s' < scripts/gpu_setup.sh && scripts/gpu_sync.sh <box>
#   ssh <box> 'cd ~/apexzero && git fetch --tags && git checkout --detach <tag>'
#   ssh <box> 'cd ~/apexzero && DRY_RUN=1 bash scripts/gpu_round.sh gw3 scripts/gw3_runs.py'
#   ssh <box> 'cd ~/apexzero && BREV_TYPE=massedcompute_A6000_plusx4 nohup bash \
#     scripts/gpu_round.sh gw3 scripts/gw3_runs.py > /dev/null 2>&1 &'
#   ssh <box> 'tail -n 20 ~/apexzero/runs/gw3-queue-0of1.log'
#   scripts/gpu_sync.sh <box> pull-round gw3 scripts/gw3_runs.py
#   uv run python scripts/round_manifest.py check --round gw3 --runs-script scripts/gw3_runs.py \
#     --tag <tag>
#
# K single-GPU boxes (massedcompute_A6000_plus), box i of 0..K-1: the same setup on each, then
#   ssh <box i> 'cd ~/apexzero && SHARD=i/K BREV_TYPE=massedcompute_A6000_plus nohup bash \
#     scripts/gpu_round.sh gw3 scripts/gw3_runs.py > /dev/null 2>&1 &'
#   SHARD=i/K scripts/gpu_sync.sh <box i> pull-round gw3 scripts/gw3_runs.py
# and one check on the Mac once every box's shard is back (it reads all K manifests).
#
# Stopping early (the stop rule): pkill -f scripts/gpu_round.sh first (its lanes, so nothing new
# starts), then the runs (pkill -f with the trainer's path), then record what finished with
#   uv run --no-sync python scripts/round_manifest.py write --round <round> --runs-script \
#     <runs-script> --shard i/K --gpus G --lanes-per-gpu N [--runs-args ...]
# (a run that was stopped is "unfinished", one never started "not run").
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

QLOG=""
stamp() { date '+%F %T'; }
die() {
  echo "gpu_round.sh: $*" >&2
  if [ -n "$QLOG" ]; then echo "[$(stamp)] ERROR $*" >> "$QLOG"; fi
  exit 1
}
say() { echo "[$(stamp)] $*"; }
posint_re='^[1-9][0-9]*$'
posint() { [[ "$1" =~ $posint_re ]]; }

[ $# -ge 2 ] || die "usage: bash scripts/gpu_round.sh <round> <runs-script> [options...]"
ROUND="$1"; RUNS_SCRIPT="$2"; shift 2
RUNS_ARGS=("$@")
round_re='^[A-Za-z0-9][A-Za-z0-9._-]*$'
[[ "$ROUND" =~ $round_re ]] || die "round name: letters, digits and . _ - only: '$ROUND'"
SHARD="${SHARD:-0/1}"
shard_re='^(0|[1-9][0-9]*)/([1-9][0-9]*)$'
[[ "$SHARD" =~ $shard_re ]] || die "SHARD must be i/K with 0 <= i < K: '$SHARD'"
SHARD_I="${SHARD%/*}"; SHARD_K="${SHARD#*/}"
[ "$SHARD_I" -lt "$SHARD_K" ] || die "SHARD must be i/K with 0 <= i < K: '$SHARD'"
LANES_PER_GPU="${LANES_PER_GPU:-4}"
posint "$LANES_PER_GPU" || die "LANES_PER_GPU must be a positive integer: '$LANES_PER_GPU'"
DRY_RUN="${DRY_RUN:-0}"

if [ -n "${PYTHON:-}" ]; then PY=("$PYTHON")
elif [ -x "$HOME/.local/bin/uv" ]; then PY=("$HOME/.local/bin/uv" run --no-sync python)
elif command -v uv > /dev/null 2>&1; then PY=("$(command -v uv)" run --no-sync python)
else die "uv not found (gpu_setup.sh installs it in ~/.local/bin); or set PYTHON"
fi

LISTED=""
if command -v nvidia-smi > /dev/null 2>&1; then
  listing=$(nvidia-smi --query-gpu=name --format=csv,noheader) || die "nvidia-smi failed"
  LISTED=$(printf '%s\n' "$listing" | awk 'NF { n++ } END { print n + 0 }')
fi
GPUS="${GPUS:-$LISTED}"
[ -n "$GPUS" ] || die "nvidia-smi not found: set GPUS"
posint "$GPUS" || die "GPUS must be a positive integer: '$GPUS'"
if [ -n "$LISTED" ] && [ "$GPUS" -gt "$LISTED" ]; then
  die "GPUS=$GPUS, but nvidia-smi lists $LISTED GPU(s)"
fi

HELPER=(scripts/round_manifest.py)
SEL=(--round "$ROUND" --runs-script "$RUNS_SCRIPT" --shard "$SHARD")
SIZE=(--gpus "$GPUS" --lanes-per-gpu "$LANES_PER_GPU")
if [ "$DRY_RUN" = 1 ]; then
  if [ -n "$(git status --porcelain --untracked-files=no 2> /dev/null)" ]; then
    echo "gpu_round.sh: note: the tracked tree is not clean, so the queue would refuse to start" >&2
  fi
  exec "${PY[@]}" "${HELPER[@]}" schedule "${SEL[@]}" "${SIZE[@]}" --format tsv \
    --runs-args ${RUNS_ARGS[@]+"${RUNS_ARGS[@]}"}
fi

TAG="${SHARD_I}of${SHARD_K}"
mkdir -p runs || die "cannot create runs/"
QLOG="runs/$ROUND-queue-$TAG.log"
# round_manifest.py check refuses a round whose box's tracked tree was not clean, so the queue
# does not start on one (after QLOG is set, so the refusal is in the queue log: nohup hides it)
DIRTY=$(git status --porcelain --untracked-files=no) || die "git status failed"
[ -z "$DIRTY" ] || die "the tracked tree is not clean, so round_manifest.py check would refuse" \
  "the round: $(printf '%s\n' "$DIRTY" | head -n 3 | tr '\n' ' ')"
STATUS="runs/$ROUND-status-$TAG.tsv"
PLAN=$(mktemp "${TMPDIR:-/tmp}/gpu_round.XXXXXX") || die "mktemp failed"
trap 'rm -f "$PLAN" "$PLAN.err"' EXIT
if ! "${PY[@]}" "${HELPER[@]}" schedule "${SEL[@]}" "${SIZE[@]}" --format nul --fresh \
    --runs-args ${RUNS_ARGS[@]+"${RUNS_ARGS[@]}"} > "$PLAN" 2> "$PLAN.err"; then
  die "no schedule: $(cat "$PLAN.err")"
fi

# The schedule, NUL-separated: lane, gpu, position, name, argument count, then the cmd.
J_LANE=(); J_GPU=(); J_NAME=(); J_FIRST=(); J_ARGC=(); ARGV=()
while IFS= read -r -d '' lane; do
  IFS= read -r -d '' gpu && IFS= read -r -d '' _position && IFS= read -r -d '' name \
    && IFS= read -r -d '' argc || die "truncated schedule"
  J_LANE+=("$lane"); J_GPU+=("$gpu"); J_NAME+=("$name"); J_FIRST+=("${#ARGV[@]}"); J_ARGC+=("$argc")
  for ((a = 0; a < argc; a++)); do
    IFS= read -r -d '' arg || die "truncated schedule"
    ARGV+=("$arg")
  done
done < "$PLAN"
rm -f "$PLAN" "$PLAN.err"
[ "${#J_NAME[@]}" -gt 0 ] || die "empty schedule"
XLA_PYTHON_CLIENT_MEM_FRACTION=$("${PY[@]}" -c "print(round(0.9 / $LANES_PER_GPU, 2))") \
  || die "cannot compute XLA_PYTHON_CLIENT_MEM_FRACTION"
export XLA_PYTHON_CLIENT_MEM_FRACTION

echo "queue log: $QLOG"
exec >> "$QLOG" 2>&1

event() {  # name event lane gpu
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$3" "$4" >> "$STATUS"
}

train() {  # gpu name cmd...
  local gpu="$1" name="$2"; shift 2
  CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" "$@" > "runs/$name.log" 2>&1 < /dev/null
}

run_job() {  # lane gpu name cmd...
  local lane="$1" gpu="$2" name="$3"; shift 3
  say "start $name (lane $lane, GPU $gpu)"
  event "$name" start "$lane" "$gpu"
  if train "$gpu" "$name" "$@"; then
    event "$name" done "$lane" "$gpu"; say "done  $name"; return 0
  fi
  say "retry $name (first attempt failed; see runs/$name.log.1)"
  event "$name" retry "$lane" "$gpu"
  mv -f "runs/$name.log" "runs/$name.log.1" || say "could not keep runs/$name.log as .log.1"
  if ! rm -rf "runs/${name:?}"; then
    event "$name" failed "$lane" "$gpu"; say "FAILED $name (could not remove runs/$name)"; return 1
  fi
  if train "$gpu" "$name" "$@"; then
    event "$name" retried "$lane" "$gpu"; say "done  $name (second attempt)"
  else
    event "$name" failed "$lane" "$gpu"; say "FAILED $name (twice; see runs/$name.log)"
  fi
}

lane() {  # lane: its jobs, in order
  local l="$1" j
  for ((j = 0; j < ${#J_NAME[@]}; j++)); do
    [ "${J_LANE[$j]}" = "$l" ] || continue
    run_job "$l" "${J_GPU[$j]}" "${J_NAME[$j]}" "${ARGV[@]:${J_FIRST[$j]}:${J_ARGC[$j]}}"
  done
}

LANES=$((GPUS * LANES_PER_GPU))
say "round $ROUND, shard $SHARD: ${#J_NAME[@]} runs on $GPUS GPU(s) x $LANES_PER_GPU lanes" \
  "(XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION; box ${BREV_TYPE:-unknown})"
for ((l = 0; l < LANES && l < ${#J_NAME[@]}; l++)); do lane "$l" & done
wait

rc=0
if ! "${PY[@]}" "${HELPER[@]}" write "${SEL[@]}" "${SIZE[@]}" \
    --runs-args ${RUNS_ARGS[@]+"${RUNS_ARGS[@]}"}; then
  say "manifest NOT written"; rc=1
fi
failed=$(awk -F '\t' '$2 == "failed"' "$STATUS" 2> /dev/null | wc -l | tr -d ' ')
if [ "$failed" -gt 0 ]; then rc=1; fi
say "round $ROUND, shard $SHARD finished ($failed failed)"
exit "$rc"
