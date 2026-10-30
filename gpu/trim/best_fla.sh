#!/usr/bin/env bash
set -u
cd /mnt/c/Projects/Qcompute
export PYTHONPATH=/mnt/c/Projects/Qcompute
stop=$(( $(date +%s) + 3600 ))
run() { name=$1; shift; t0=$(date +%s); TRIM_ALLOW_FLA=1 timeout $(( stop - $(date +%s) )) bash gpu/run_py.sh gpu/trim/trim_chat.py "$@" --head-trim --output gpu/out/$name.json > gpu/out/$name.log 2>&1; echo "$name rc=$? wall=$(( $(date +%s) - t0 ))s" | tee -a gpu/out/$name.log; }
rm -f gpu/out/trim_best_store.npz
run trim_best_hi --ask Hi --max-new 12
run trim_best_put --ask 'Remember that my locker code is 4471-AZ.' --max-new 120 --memory /mnt/c/Projects/Qcompute/gpu/out/trim_best_store.npz
run trim_best_get --ask "What's my locker code?" --max-new 120 --memory /mnt/c/Projects/Qcompute/gpu/out/trim_best_store.npz
echo BEST_DONE