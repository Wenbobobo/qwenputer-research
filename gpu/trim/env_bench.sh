#!/usr/bin/env bash
# Follow-up env-only suite. Caller must hold GPU ownership after D7_DONE.
set -u
cd /mnt/c/Projects/Qcompute
export PYTHONPATH=/mnt/c/Projects/Qcompute
unset PYTORCH_CUDA_ALLOC_CONF HSA_OVERRIDE_GFX_VERSION
unset TRIM_INT8_HEAD TRIM_ALLOW_FLA TRIM_SELECTIVE_FLA
stop=$(TZ=Asia/Shanghai date -d '2026-10-30 00:25' +%s)
run() {
  name=$1; shift
  left=$(( stop - $(date +%s) ))
  [ "$left" -lt 120 ] && { echo 'hard stop'; return 1; }
  timeout "$left" bash gpu/run_py.sh gpu/trim/trim_chat.py "$@" --output "gpu/out/$name.json" > "gpu/out/$name.log" 2>&1
  rc=$?
  echo "$name rc=$rc"
  return "$rc"
}
export TORCH_BLAS_PREFER_HIPBLASLT=1
unset PYTORCH_TUNABLEOP_ENABLED PYTORCH_TUNABLEOP_TUNING
run hip_put --ask 'Remember that my locker code is 4471-AZ.' --max-new 120 --memory /mnt/c/Projects/Qcompute/gpu/out/hip_store.npz || exit 1
run hip_get --ask "What's my locker code?" --max-new 120 --memory /mnt/c/Projects/Qcompute/gpu/out/hip_store.npz || exit 1
unset TORCH_BLAS_PREFER_HIPBLASLT
export PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_FILENAME=/mnt/c/Projects/Qcompute/gpu/out/tunableop_gfx1100.csv
run tuned_hi --ask Hi --max-new 12 || true
run tuned_put --ask 'Remember that my locker code is 4471-AZ.' --max-new 120 --memory /mnt/c/Projects/Qcompute/gpu/out/tuned_store.npz || exit 1
run tuned_get --ask "What's my locker code?" --max-new 120 --memory /mnt/c/Projects/Qcompute/gpu/out/tuned_store.npz || exit 1
echo ENV_BENCH_DONE
