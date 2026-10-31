#!/usr/bin/env bash
# Stage-8 serial speed matrix: one process at a time, GPU exclusive.
set -u
cd /mnt/c/Projects/Qcompute
export PYTHONPATH=/mnt/c/Projects/Qcompute:/mnt/c/Projects/Qcompute/repro/qwenputer/experimental/riscv:/mnt/c/Projects/Qcompute/repro/qwenputer
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTORCH_CUDA_ALLOC_CONF HSA_OVERRIDE_GFX_VERSION TRIM_INT8_HEAD
run() { name=$1; shift; t0=$(date +%s); timeout 600 bash gpu/run_py.sh gpu/trim/trim_chat.py "$@" --output gpu/out/$name.json > gpu/out/$name.log 2>&1; rc=$?; echo "$name rc=$rc wall=$(( $(date +%s) - t0 ))s" | tee -a gpu/out/s8_matrix.log; }
rm -f gpu/out/s8_*store*.npz
# 1) bf16 no-FLA
unset TRIM_FP32_CHIP TRIM_SELECTIVE_FLA TRIM_ALLOW_FLA
run s8_bf16_hi --ask Hi --max-new 12
# 2) bf16 selective
export TRIM_SELECTIVE_FLA=1
run s8_bf16_sel_hi --ask Hi --max-new 12
unset TRIM_SELECTIVE_FLA
# 3) bf16 full FLA
export TRIM_ALLOW_FLA=1
run s8_bf16_fla_hi --ask Hi --max-new 12
unset TRIM_ALLOW_FLA
# 4) fp32 no-FLA
export TRIM_FP32_CHIP=1
run s8_fp32_hi --ask Hi --max-new 12
# 5) fp32 selective
export TRIM_FP32_CHIP=1 TRIM_SELECTIVE_FLA=1
run s8_fp32_sel_hi --ask Hi --max-new 12
unset TRIM_SELECTIVE_FLA
# 6) fp32 full FLA
export TRIM_FP32_CHIP=1 TRIM_ALLOW_FLA=1
run s8_fp32_fla_hi --ask Hi --max-new 12
unset TRIM_ALLOW_FLA
# 7) put/get for two best guesses: fp32 no-FLA and fp32 full-FLA (fill later if needed)
export TRIM_FP32_CHIP=1
run s8_fp32_put --ask 'Remember that my locker code is 4471-AZ.' --max-new 120 --memory gpu/out/s8_fp32_store.npz
run s8_fp32_get --ask "What's my locker code?" --max-new 120 --memory gpu/out/s8_fp32_store.npz
unset TRIM_FP32_CHIP
echo S8_MATRIX_DONE