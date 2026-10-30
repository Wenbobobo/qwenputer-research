"""Run the upstream fp32 chip check with torch fallbacks and bounded CPU threads.

Usage: bash gpu/run_py.sh gpu/trim/gpucheck_fp32chip.py --decode --dtype fp32 --progs signext,call
This tests standalone emitted chip arithmetic; the mixed trim path is covered by
fp32_chip_bench.py. Upstream fp32 mode accepts non-bit-exact dims if tokens agree.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
torch.set_num_threads(8)
from gpu.trim import common  # establishes chipformer paths and blocks FLA by default
from chipformer.rv import gpucheck

if __name__ == "__main__":
    gpucheck.main()
