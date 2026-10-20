#!/usr/bin/env bash
# Set up a fresh Linux CUDA 12 box (e.g. a Brev instance) to train ApexZero agents.
#
#   ssh <box> 'bash -s' < scripts/gpu_setup.sh     # from the Mac
#   scripts/gpu_sync.sh <box>                       # then copy the R-dev data (not in git)
#
# Installs uv, clones the public repo (or fast-forwards it), syncs the locked environment
# and adds JAX's CUDA plugin at the locked JAX version. The plugin is not in the lockfile,
# so on the box run scripts with `uv run --no-sync` (a plain sync would remove it).
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ~/apexzero ] || git clone https://github.com/adrian0729/apexzero.git ~/apexzero
cd ~/apexzero
git pull --ff-only
uv sync
JAXV=$(uv run --no-sync python -c "import jax; print(jax.__version__)")
uv pip install "jax[cuda12]==${JAXV}"
uv run --no-sync python -c "import jax; print('jax', jax.__version__, jax.devices())"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
