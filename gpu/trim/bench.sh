#!/usr/bin/env bash
# Serial trimmed-RvChat GPU benchmark (fresh process each). Waits for a free GPU first.
set -u
cd /mnt/c/Projects/Qcompute
export PYTHONPATH=/mnt/c/Projects/Qcompute
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stop=$(TZ=Asia/Shanghai date -d 'tomorrow 00:25' +%s); [ $(( stop - $(date +%s) )) -gt 86400 ] && stop=$(TZ=Asia/Shanghai date -d 'today 00:25' +%s)
while pgrep -f 'quant_chat.py|run_e2e.py|trim_cpu_check.py' >/dev/null; do echo "$(date +%T) waiting for quant"; sleep 30; done
bash gpu/run_py.sh gpu/mem.py
rm -f gpu/out/trim_store.npz
run() { name=$1; shift; left=$(( stop - $(date +%s) )); [ $left -le 60 ] && { echo 'hard stop'; exit 1; }
  t0=$(date +%s); timeout $left bash gpu/run_py.sh gpu/trim/trim_chat.py "$@" --output gpu/out/$name.json > gpu/out/$name.log 2>&1; rc=$?
  echo "$name rc=$rc wall=$(( $(date +%s) - t0 ))s" | tee -a gpu/out/$name.log; }
run trim_hi --ask Hi --max-new 12
run trim_put --ask 'Remember that my locker code is 4471-AZ.' --max-new 120 --memory /mnt/c/Projects/Qcompute/gpu/out/trim_store.npz
run trim_get --ask "What's my locker code?" --max-new 120 --memory /mnt/c/Projects/Qcompute/gpu/out/trim_store.npz
[ "${TRIM_HEAD:-1}" = 1 ] && run trim_hi_head --ask Hi --max-new 12 --head-trim
echo BENCH_DONE
