#!/usr/bin/env bash
# Helper: run a python script inside WSL with the ROCm torch venv and the WSL GPU env.
# Usage: wsl -d Ubuntu-24.04 -- bash /mnt/c/Projects/Qcompute/gpu/run_py.sh <script.py> [args...]
set -e
export HSA_ENABLE_DXG_DETECTION=1
export PYTORCH_ROCM_ARCH=gfx1100
export PATH=/opt/rocm/bin:/usr/lib/wsl/lib:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd /mnt/c/Projects/Qcompute
exec "$HOME/src/rocm-smoke/.venv/bin/python" "$@"