#!/usr/bin/env bash
# On a GPU box: H4R's 20 runs (docs/interp/h4r-design.md, Section 2), LANES at a time on one GPU.
# Jobs interleave the arms and variants so each round mixes them; a failed run is re-run once.
#
#   ssh <box> 'cd ~/apexzero && nohup bash scripts/gpu_queue_h4r.sh [lanes] > runs/h4r.log 2>&1 &'
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
LANES="${1:-4}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=$(python3 -c "print(round(0.9 / ${LANES}, 2))")

train() {
  local name="$1"; shift
  ~/.local/bin/uv run --no-sync python scripts/train_dr.py --steps 200e6 --name "$name" "$@" \
    > "runs/$name.log" 2>&1
}

run() {
  local name="$1"; shift
  echo "[$(date '+%F %T')] start $name"
  if train "$name" "$@"; then echo "[$(date '+%F %T')] done  $name"; return; fi
  echo "[$(date '+%F %T')] retry $name (first attempt failed; see runs/$name.log.1)"
  mv "runs/$name.log" "runs/$name.log.1"; rm -rf "runs/$name"
  if train "$name" "$@"; then echo "[$(date '+%F %T')] done  $name"
  else echo "[$(date '+%F %T')] FAILED $name (twice; see runs/$name.log)"; fi
}

V3="--tilt-fraction 0.5 --tilt-alpha 2"
jobs=()
for s in 10 11 12 13 14; do
  jobs+=("h4r-v1-base-s$s --seed $s")
  jobs+=("h4r-v1-realw-s$s --seed $s --real-widths")
  jobs+=("h4r-v3-base-s$s --seed $s $V3")
  jobs+=("h4r-v3-realw-s$s --seed $s $V3 --real-widths")
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
echo "[$(date '+%F %T')] H4R runs finished"
