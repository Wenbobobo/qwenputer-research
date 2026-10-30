#!/usr/bin/env bash
cd /mnt/c/Projects/Qcompute
for i in $(seq 1 600); do grep -q "AFTER DONE" analysis/experiments/h1/after.log && break; sleep 10; done
for CS in "$@"; do
  C=${CS%%:*}; S=${CS##*:}
  echo "=== $C seed $S $(date)"
  bash gpu/run_py.sh analysis/experiments/h1/h1_run.py --cond $C --n 30 --seed $S --out analysis/experiments/h1/out
done
echo "AFTER2 DONE $(date)"
