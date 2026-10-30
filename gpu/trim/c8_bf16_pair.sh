#!/usr/bin/env bash
set -u
cd /mnt/c/Projects/Qcompute
export PYTHONPATH=/mnt/c/Projects/Qcompute:/mnt/c/Projects/Qcompute/repro/qwenputer/experimental/riscv:/mnt/c/Projects/Qcompute/repro/qwenputer
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
dl=$(( $(date +%s) + 1500 ))
TRIM_FP32_CHIP=0 timeout 1400 bash gpu/run_py.sh gpu/trim/fp32_chip_bench.py --out gpu/out/c8_bf16_pair --cases 1,2,5 --deadline $dl > gpu/out/c8_bf16_pair.log 2>&1
echo "bf16_pair rc=$? wall_done=$(date +%T)" >> gpu/out/c8_bf16_pair.log
tail -3 gpu/out/c8_bf16_pair.log