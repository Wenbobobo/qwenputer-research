"""Run upstream RvChat natively on the remote GPU (no accelerate, no patching).

RvChat.__init__ compacts the 12 chip layers to their nonzero blocks on CPU, then
moves the whole model to cuda: Qwen 11.97 + tied embed/lm_head 4.47 + chip ~0.2 GiB
= ~16.6 GiB, which fits the 19.96 GiB card. This is the intended fast path.
"""
import sys
sys.path.insert(0, "/mnt/c/Projects/Qcompute/repro/qwenputer")
sys.path.insert(0, "/mnt/c/Projects/Qcompute/repro/qwenputer/experimental/riscv")

from chipformer.rv import chat

if __name__ == "__main__":
    chat.main()