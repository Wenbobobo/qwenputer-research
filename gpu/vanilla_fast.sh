#!/usr/bin/env bash
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/c/Projects/Qcompute
rm -f gpu/out/store_vanilla.npz
t0=$(date +%s)
bash gpu/run_py.sh gpu/vanilla_gpu.py /mnt/c/Projects/Qcompute/.data/Qwen3.5-4B-Qwenputer-Filesystem --firmware /mnt/c/Projects/Qcompute/repro/qwenputer/experimental/riscv/tests/data/rv/calc-bridge.elf --ask "Remember that my locker code is 4471-AZ." --max-new 120 --memory /mnt/c/Projects/Qcompute/gpu/out/store_vanilla.npz > gpu/out/vanilla_put2.log 2>&1
echo "put_wall_s=$(( $(date +%s) - t0 ))" >> gpu/out/vanilla_put2.log
echo DONE