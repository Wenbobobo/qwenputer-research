#!/usr/bin/env bash
# Serial GPU measurements for baseline, INT8 head, and ROCm env tuning.
# Never enables expandable_segments or HSA_OVERRIDE_GFX_VERSION.
set -u
cd /mnt/c/Projects/Qcompute
export PYTHONPATH=/mnt/c/Projects/Qcompute
unset PYTORCH_CUDA_ALLOC_CONF HSA_OVERRIDE_GFX_VERSION
unset TORCH_BLAS_PREFER_HIPBLASLT PYTORCH_TUNABLEOP_ENABLED
unset PYTORCH_TUNABLEOP_TUNING PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS
unset TRIM_INT8_HEAD TRIM_ALLOW_FLA TRIM_SELECTIVE_FLA
stop=$(TZ=Asia/Shanghai date -d '2026-10-30 00:25' +%s)
fallback=$(TZ=Asia/Shanghai date -d '2026-10-29 23:30' +%s)
reduced=0
until [ -e gpu/D7_DONE ]; do
  left=$(( stop - $(date +%s) ))
  [ "$left" -le 600 ] && { echo 'hard stop before GPU handoff'; exit 1; }
  if [ "$(date +%s)" -ge "$fallback" ]; then
    bash gpu/run_py.sh gpu/mem.py | tee gpu/out/int8_idle_mem.txt
    free_gib=$(awk '/gpu free/{print $3}' gpu/out/int8_idle_mem.txt)
    busy=$(pgrep -af '[/]python|^python' || true)
    if awk "BEGIN{exit !($free_gib > 17)}" && [ -z "$busy" ]; then reduced=1; break; fi
  fi
  echo "$(date +%T) waiting for gpu/D7_DONE" | tee -a gpu/out/int8_wait.log
  sleep 90
done
bash gpu/run_py.sh gpu/mem.py | tee gpu/out/int8_start_mem.txt
run() {
  name=$1; shift
  left=$(( stop - $(date +%s) ))
  [ "$left" -le 120 ] && { echo "$name skipped at hard stop"; return 1; }
  t0=$(date +%s)
  timeout --kill-after=5 "$left" bash gpu/run_py.sh gpu/trim/trim_chat.py "$@"     --output "gpu/out/$name.json" > "gpu/out/$name.log" 2>&1
  rc=$?
  echo "$name rc=$rc wall=$(( $(date +%s) - t0 ))s" | tee -a "gpu/out/$name.log"
  return "$rc"
}
if [ "$reduced" = 0 ]; then
run trim_base_hi --ask Hi --max-new 12 || exit 1
run trim_base_put --ask 'Remember that my locker code is 4471-AZ.' --max-new 120   --memory /mnt/c/Projects/Qcompute/gpu/out/int8_base_store.npz || exit 1
run trim_base_get --ask "What's my locker code?" --max-new 120   --memory /mnt/c/Projects/Qcompute/gpu/out/int8_base_store.npz || exit 1
fi
export TRIM_INT8_HEAD=1
run int8_hi --ask Hi --max-new 12 || exit 1
run int8_put --ask 'Remember that my locker code is 4471-AZ.' --max-new 120   --memory /mnt/c/Projects/Qcompute/gpu/out/int8_store.npz || exit 1
run int8_get --ask "What's my locker code?" --max-new 120   --memory /mnt/c/Projects/Qcompute/gpu/out/int8_store.npz || exit 1
[ "$reduced" = 1 ] && { echo INT8_REDUCED_DONE; exit 0; }
unset TRIM_INT8_HEAD
export TORCH_BLAS_PREFER_HIPBLASLT=1
run hip_hi --ask Hi --max-new 12 || true
export TRIM_INT8_HEAD=1
run hip_int8_hi --ask Hi --max-new 12 || true
unset TORCH_BLAS_PREFER_HIPBLASLT TRIM_INT8_HEAD
export PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1
export PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS=2
export PYTORCH_TUNABLEOP_FILENAME=/mnt/c/Projects/Qcompute/gpu/out/tunableop_gfx1100.csv
run tune_hi --ask Hi --max-new 12 || true
echo INT8_BENCH_DONE
