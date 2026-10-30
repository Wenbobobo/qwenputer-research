#!/usr/bin/env bash
# Stage-6 kernels: wait for GPU idle, then run bnb8 Hi / put / fresh get with FLA installed.
# Usage (WSL): bash gpu/kernels/run_fla.sh
cd /mnt/c/Projects/Qcompute
STATUS=gpu/out/fla_status.log
cutoff=$(TZ=Asia/Shanghai date -d 'tomorrow 00:25' +%s); now=$(date +%s)
[ $((cutoff-now)) -gt 86400 ] && cutoff=$((cutoff-86400))
busy() { ps aux | grep -E 'quant_chat|run_e2e|trim_|vanilla_gpu|bench\.sh' | grep -v grep | grep -v run_fla >/dev/null; }
deadline=$(( $(date +%s) + ${WAIT_S:-2700} ))
while :; do
  if ! busy; then
    free=$(bash gpu/run_py.sh gpu/mem.py 2>/dev/null | awk '/gpu free/{print $3}')
    echo "$(TZ=Asia/Shanghai date +%T) idle? free=$free" >> $STATUS
    awk -v f="$free" 'BEGIN{exit !(f>17)}' && break
  else
    echo "$(TZ=Asia/Shanghai date +%T) busy" >> $STATUS
  fi
  [ $(date +%s) -ge $deadline ] && { echo "wait timeout" >> $STATUS; exit 2; }
  sleep 60
done
run() { # stem ask maxnew [memory]
  local left=$(( cutoff - $(date +%s) - 60 ))
  [ $left -le 0 ] && { echo "hard stop" >> $STATUS; exit 3; }
  echo "$(TZ=Asia/Shanghai date +%T) start $1 (timeout $left s)" >> $STATUS
  local extra=(); [ -n "$4" ] && extra=(--memory "$4")
  timeout $left bash gpu/run_py.sh gpu/quant/quant_chat.py --scheme bnb8 --ask "$2" --max-new $3 --output gpu/out/$1.json "${extra[@]}" > gpu/out/$1.log 2>&1
  local rc=$?; echo "$(TZ=Asia/Shanghai date +%T) end $1 rc=$rc" >> $STATUS; return $rc
}
run fla_bnb8_hi Hi 12 || exit 1
rm -f gpu/out/fla_bnb8_store.npz
run fla_bnb8_put 'Remember that my locker code is 4471-AZ.' 120 gpu/out/fla_bnb8_store.npz || exit 1
run fla_bnb8_get "What's my locker code?" 120 gpu/out/fla_bnb8_store.npz
echo done >> $STATUS
