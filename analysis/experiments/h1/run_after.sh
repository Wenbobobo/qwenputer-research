#!/usr/bin/env bash
# Wait for the main chain to finish, then run extra conditions. Usage: bash run_after.sh COND:SEED ...
cd /mnt/c/Projects/Qcompute
for i in $(seq 1 600); do grep -q "CHAIN DONE" analysis/experiments/h1/full.log && break; sleep 15; done
for CS in "$@"; do
  C=${CS%%:*}; S=${CS##*:}
  echo "=== $C seed $S $(date)"
  bash gpu/run_py.sh analysis/experiments/h1/h1_run.py --cond $C --n 30 --seed $S --out analysis/experiments/h1/out
done
echo "AFTER DONE $(date)"
