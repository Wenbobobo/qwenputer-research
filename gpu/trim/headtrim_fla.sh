#!/usr/bin/env bash
# head-trim + trim+FLA follow-ups (run only when GPU is idle and no other agent runs)
set -u
cd /mnt/c/Projects/Qcompute
export PYTHONPATH=/mnt/c/Projects/Qcompute
stop=$(( $(date +%s) + 5400 ))
while pgrep -f 'quant_chat.py|run_fla|trim_chat.py' >/dev/null; do
  now=$(date +%s); [ $(( stop - now )) -le 120 ] && { echo hard_stop; exit 1; }
  echo "$(date +%T) waiting"; sleep 20
done
run() { name=$1; shift; t0=$(date +%s); timeout $(( stop - $(date +%s) )) bash gpu/run_py.sh gpu/trim/trim_chat.py "$@" --output gpu/out/$name.json > gpu/out/$name.log 2>&1; echo "$name rc=$? wall=$(( $(date +%s) - t0 ))s" | tee -a gpu/out/$name.log; }
run trim_hi_head --ask Hi --max-new 12 --head-trim
TRIM_ALLOW_FLA=1 run trim_hi_fla --ask Hi --max-new 12
TRIM_ALLOW_FLA=1 run trim_put_fla --ask 'Remember that my locker code is 4471-AZ.' --max-new 120 --memory /mnt/c/Projects/Qcompute/gpu/out/trim_fla_store.npz
TRIM_ALLOW_FLA=1 run trim_get_fla --ask "What's my locker code?" --max-new 120 --memory /mnt/c/Projects/Qcompute/gpu/out/trim_fla_store.npz
echo FOLLOWUP_DONE