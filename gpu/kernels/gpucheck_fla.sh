#!/usr/bin/env bash
set -u
cd /mnt/c/Projects/Qcompute
export PYTHONPATH=/mnt/c/Projects/Qcompute/repro/qwenputer/experimental/riscv
t0=$(date +%s)
timeout 900 bash gpu/run_py.sh -m chipformer.rv.gpucheck --decode > gpu/out/gpucheck_fla.log 2>&1
rc=$?
echo "gpucheck rc=$rc wall=$(( $(date +%s) - t0 ))s" >> gpu/out/gpucheck_fla.log
tail -8 gpu/out/gpucheck_fla.log