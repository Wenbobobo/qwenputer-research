#!/bin/bash
set -u
cd /mnt/c/Projects/Qcompute
stop=$(TZ=Asia/Shanghai date -d '2026-10-30 00:25:00' +%s)
while [ ! -f gpu/A7_DONE ]; do
  [ $(date +%s) -ge $((stop-600)) ] && exit 3
  echo "$(date) waiting A7_DONE"
  sleep 60
done
while ps -eo comm,args | awk '$1 ~ /^python/ && $0 ~ /gpu|run_inw|chipformer/ {found=1} END {exit !found}'; do
  [ $(date +%s) -ge $((stop-600)) ] && exit 4
  echo "$(date) waiting Python model process"
  sleep 60
done
export PYTHONPATH=/mnt/c/Projects/Qcompute
unset TRIM_ALLOW_FLA
left=$((stop-$(date +%s)))
timeout $left bash gpu/run_py.sh analysis/experiments/h1_inweights/run_inw.py --n 5 --deadline $stop > analysis/experiments/h1_inweights/run5.log 2>&1
rc=$?
echo "five-case rc=$rc $(date)"
