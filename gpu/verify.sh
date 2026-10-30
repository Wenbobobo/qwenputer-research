#!/usr/bin/env bash
set -e
cd /mnt/c/Projects/Qcompute
bash gpu/run_py.sh gpu/rvchat_offload.py /mnt/c/Projects/Qcompute/.data/Qwen3.5-4B-Qwenputer-Filesystem --firmware /mnt/c/Projects/Qcompute/repro/qwenputer/experimental/riscv/tests/data/rv/calc-bridge.elf --ask Hi --max-new 12 --slow --memory /mnt/c/Projects/Qcompute/gpu/out/store_gpu.npz > gpu/out/verify_gpu.log 2>&1