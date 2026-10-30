#!/usr/bin/env bash
# Head-trim GPU variant (waits for any quant run), then the CPU store_hi resume check.
set -u
cd /mnt/c/Projects/Qcompute
export PYTHONPATH=/mnt/c/Projects/Qcompute PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stop=$(TZ=Asia/Shanghai date -d 'tomorrow 00:25' +%s); [ $(( stop - $(date +%s) )) -gt 86400 ] && stop=$(TZ=Asia/Shanghai date -d 'today 00:25' +%s)
while pgrep -f 'quant_chat.py|run_e2e.py' >/dev/null; do echo "$(date +%T) waiting for quant"; sleep 20; done
bash gpu/run_py.sh gpu/mem.py
t0=$(date +%s)
timeout $(( stop - t0 )) bash gpu/run_py.sh gpu/trim/trim_chat.py --ask Hi --max-new 12 --head-trim --output gpu/out/trim_hi_head.json > gpu/out/trim_hi_head.log 2>&1
echo "trim_hi_head rc=$? wall=$(( $(date +%s) - t0 ))s"
t0=$(date +%s)
CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 bash gpu/run_py.sh gpu/trim/trim_cpu_resume.py --key store_hi --threads 10 > gpu/out/trim_cpu_resume.log 2>&1
echo "cpu_resume rc=$? wall=$(( $(date +%s) - t0 ))s"
echo HEAD_DONE
