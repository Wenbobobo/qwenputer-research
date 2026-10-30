#!/usr/bin/env bash
set -e
cd /mnt/c/Projects/Qcompute
rm -f gpu/out/store_gpu2.npz
t0=$(date +%s)
bash gpu/run_py.sh gpu/rvchat_offload.py /mnt/c/Projects/Qcompute/.data/Qwen3.5-4B-Qwenputer-Filesystem --firmware /mnt/c/Projects/Qcompute/repro/qwenputer/experimental/riscv/tests/data/rv/calc-bridge.elf --ask "Remember that my locker code is 4471-AZ." --max-new 120 --slow --memory /mnt/c/Projects/Qcompute/gpu/out/store_gpu2.npz > gpu/out/put_gpu2.log 2>&1
echo "put_wall_s=$(( $(date +%s) - t0 ))" >> gpu/out/put_gpu2.log
t1=$(date +%s)
bash gpu/run_py.sh gpu/rvchat_offload.py /mnt/c/Projects/Qcompute/.data/Qwen3.5-4B-Qwenputer-Filesystem --firmware /mnt/c/Projects/Qcompute/repro/qwenputer/experimental/riscv/tests/data/rv/calc-bridge.elf --ask "What's my locker code?" --max-new 120 --slow --memory /mnt/c/Projects/Qcompute/gpu/out/store_gpu2.npz > gpu/out/get_gpu2.log 2>&1
echo "get_wall_s=$(( $(date +%s) - t1 ))" >> gpu/out/get_gpu2.log
echo DONE