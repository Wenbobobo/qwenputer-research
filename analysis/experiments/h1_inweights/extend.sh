#!/bin/bash
set -eu
cd /mnt/c/Projects/Qcompute
if ps -eo comm,args | awk '$1 ~ /^python/ && $0 ~ /gpu|run_inw|chipformer/ {found=1} END {exit !found}'; then echo GPU_BUSY; exit 2; fi
export PYTHONPATH=/mnt/c/Projects/Qcompute
unset TRIM_ALLOW_FLA
stop=$(TZ=Asia/Shanghai date -d '2026-10-30 00:25:00' +%s)
timeout $((stop-$(date +%s))) bash gpu/run_py.sh analysis/experiments/h1_inweights/run_inw.py --n 10 --start 5 --deadline $stop > analysis/experiments/h1_inweights/run10.log 2>&1
