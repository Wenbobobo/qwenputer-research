#!/usr/bin/env bash
# Stage 7: gpucheck --decode with selective FLA (chip layers on the reference GDN kernels).
set -u
cd /mnt/c/Projects/Qcompute
export PYTHONPATH=/mnt/c/Projects/Qcompute/repro/qwenputer/experimental/riscv
LOG=gpu/out/${1:-gpucheck_selfla}.log
t0=$(date +%s)
timeout ${GC_TIMEOUT:-1500} bash gpu/run_py.sh gpu/trim/gpucheck_selective.py --decode > $LOG 2>&1
rc=$?
echo "gpucheck rc=$rc wall=$(( $(date +%s) - t0 ))s" >> $LOG
tail -24 $LOG
