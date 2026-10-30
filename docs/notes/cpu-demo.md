# 发布 checkpoint CPU 演示（已合并）

本文件原记录"BLOCKED（内存竞争）"的历史。实际演示已于 2026-10-29 成功：

- 记录与结论：docs/notes/cpu-checkpoint-demo.md
- smoke 日志：demo/run-smoke.log
- put 日志：demo/run-a.log
- get 日志：demo/run-b.log
- 快照：demo/store.npz（11.5 KB）
- 脚本：demo/driver_single_process.py（一次性加载模型做 A/B/C）
- 旧脚本：demo/run-demo.ps1（已修复 PowerShell 5.1 把 python stderr 当终止错误的问题）

结论：22.3GB checkpoint 在 48GB CPU 机器上可加载（~20s，peak ~22GB），单 token ~4.7s；
模型会自发调用内建 list/put，电路在权重内回答，head mismatch 0、lock error None。
