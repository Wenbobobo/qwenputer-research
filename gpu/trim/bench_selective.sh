#!/usr/bin/env bash
# Stage 7 serialized selective FLA benchmark. Refuse to run past the GPU hard stop.
set -u
cd /mnt/c/Projects/Qcompute
export PYTHONPATH=/mnt/c/Projects/Qcompute
export TRIM_SELECTIVE_FLA=1
cutoff=$(TZ=Asia/Shanghai date -d 'tomorrow 00:25' +%s)
run() {
  local name=$1; shift
  local left=$(( cutoff - $(date +%s) - 60 ))
  [ $left -gt 0 ] || return 124
  timeout $left bash gpu/run_py.sh gpu/trim/trim_chat.py "$@" --output gpu/out/$name.json > gpu/out/$name.log 2>&1
  local rc=$?
  echo "$name rc=$rc" >> gpu/out/$name.log
  return $rc
}
run trim_selfla_hi --ask Hi --max-new 12 || exit $?
run trim_selfla_put --ask 'Remember that my locker code is 4471-AZ.' --max-new 120 --memory /mnt/c/Projects/Qcompute/gpu/out/trim_selfla_store.npz || exit $?
run trim_selfla_get --ask "What's my locker code?" --max-new 120 --memory /mnt/c/Projects/Qcompute/gpu/out/trim_selfla_store.npz
