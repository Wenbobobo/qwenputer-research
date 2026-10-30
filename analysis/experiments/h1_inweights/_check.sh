#!/bin/bash
set -e
cd /mnt/c/Projects/Qcompute
/home/wenbo/src/rocm-smoke/.venv/bin/python -m py_compile analysis/experiments/h1_inweights/run_inw.py
echo COMPILE_OK
/home/wenbo/src/rocm-smoke/.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, 'analysis/experiments/h1')
import h1_gen
cs = h1_gen.recall_cases(2, 0)
c = cs[0]
print(c['id'], len(c['sessions']), [q['kind'] for q in c['questions']])
print('gold/old:', [(q['gold'], q['old']) for q in c['questions']])
PY
