# CPU checkpoint demo — status: BLOCKED (not run)

Date: 2026-10-29. Machine: Windows 11, 47.9 GB RAM, CPU only.

## Outcome
**BLOCKED — the model was never loaded.** The precondition "free RAM >= 25 GB" (the bf16
checkpoint alone is ~20.7 GiB resident) was never met during ~11 min of monitoring.
Concurrent pytest runs of sibling agents in the same venv (`tests/test_rv_hf.py`,
`test_rv_core.py`, `test_rv_ref/fuzz/bridge.py`) held 6-21 GB, leaving 5.9-23.0 GB free.
No smoke/A/B/calc logs exist because no run was started (no load time, s/token, or
transcript to report).

## RAM observations
| t (+min) | free RAM | python WS (sum) | python procs |
|---|---|---|---|
| 0 | 23.0 GB | — | — |
| ~1 | 18.5 GB | — | — |
| ~2 | 13.2 GB | — | 8 |
| ~4 | 5.9 GB | 21.0 GB | 14 |
| ~6 | 18.6 GB | 9.7 GB | 12 |
| ~7.5 | 21.6 GB | 6.6 GB | 10 |
| ~9 | 13.4 GB | 14.5 GB | 12 |
| ~11 | 19.7 GB | 8.4 GB | 8 |

## How to run once the machine is quiet
```
pwsh -File C:\Projects\Qcompute\demo\run-demo.ps1 -Step smoke   # --ask Hi --max-new 32 -> run-smoke.log
pwsh -File C:\Projects\Qcompute\demo\run-demo.ps1 -Step a       # put: locker code 4471-AZ -> run-a.log, store.npz
pwsh -File C:\Projects\Qcompute\demo\run-demo.ps1 -Step b       # get, empty context -> run-b.log
pwsh -File C:\Projects\Qcompute\demo\run-demo.ps1 -Step calc    # expect 121932631112635269 -> run-calc.log
```
Equivalent raw command (workdir `repro\qwenputer\experimental\riscv`):
`.venv\Scripts\python.exe -u -m chipformer.rv.chat C:\Projects\Qcompute\.data\Qwen3.5-4B-Qwenputer-Filesystem --memory C:\Projects\Qcompute\demo\store.npz --ask "..." --max-new 200`

The script refuses to start below 25 GB free. Run steps one at a time, in the background, and
watch free RAM; kill if it drops below 25 GB or a step exceeds ~40 min.
Check: A should contain a `tool_call`/put and chip/trace tokens; B (fresh context) should answer 4471-AZ.
