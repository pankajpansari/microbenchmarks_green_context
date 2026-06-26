#!/bin/bash
# Shared environment for the profiling runners in this directory.
# SOURCE this file (`source env.sh`); do not execute it.
#
# Why this exists: `sudo` resets PATH/PYTHONPATH/LD_LIBRARY_PATH, so a plain
# `sudo ncu python3 ...` loses the venv and CUDA libs. Every runner needs the
# same env forwarded, so it lives here once instead of being copy-pasted.
#
# Per-machine overrides: export PROFILE_PATH / PROFILE_PYTHONPATH /
# PROFILE_LD_LIBRARY_PATH before calling a runner if your venv or CUDA install
# is not already on the interactive shell's PATH. By default we forward the
# current shell's values.

: "${PROFILE_PATH:=${PATH:-}}"
: "${PROFILE_PYTHONPATH:=${PYTHONPATH:-}}"
: "${PROFILE_LD_LIBRARY_PATH:=${LD_LIBRARY_PATH:-}}"

# Run a command under sudo with the profiling env preserved.
# Usage: sudo_env ncu --metrics ... python3 profile_x.py ...
sudo_env() {
  sudo env "PATH=$PROFILE_PATH" \
           "PYTHONPATH=$PROFILE_PYTHONPATH" \
           "LD_LIBRARY_PATH=$PROFILE_LD_LIBRARY_PATH" \
           "$@"
}

# Read a value from the repo's config.py as a bash string.
# Lists/tuples become space-separated ("512 1024 2048"); scalars print as-is.
# Usage: D_SWEEP=$(cfg D_sweep)   (requires REPO to point at the repo root)
cfg() {
  python3 -c "import sys; sys.path.insert(0, '$REPO'); import config; \
v = getattr(config, sys.argv[1]); \
print(' '.join(map(str, v)) if isinstance(v, (list, tuple)) else v)" "$1"
}
