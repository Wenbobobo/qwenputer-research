#!/usr/bin/env bash
# Run H1 conditions sequentially. Usage: bash run_chain.sh <outdir> <n> COND:SEED ...
cd /mnt/c/Projects/Qcompute
OUT=$1; N=$2; shift 2
for i in $(seq 1 300); do [ -f gpu/A_DONE ] && break; sleep 15; done
for CS in "$@"; do
  C=${CS%%:*}; S=${CS##*:}
  echo "=== $C seed $S $(date)"
  bash gpu/run_py.sh analysis/experiments/h1/h1_run.py --cond $C --n $N --seed $S --out $OUT
done
echo "CHAIN DONE $(date)"
