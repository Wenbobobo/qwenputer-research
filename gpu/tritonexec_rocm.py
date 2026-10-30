"""The circuit's Triton executor (chipformer.rv.tritonexec.Executor2) on ROCm: IR-exactness
(tritonexec.check: every dim at every position) and speed under a HIP graph (fastexec.timed2).
Upstream code unmodified.

    wsl -d Ubuntu-24.04 -- bash /mnt/c/Projects/Qcompute/gpu/run_py.sh gpu/tritonexec_rocm.py
"""
import sys
import time
import traceback

RV = "/mnt/c/Projects/Qcompute/repro/qwenputer/experimental/riscv"
sys.path.insert(0, RV)

import torch  # noqa: E402

from chipformer.rv import core, elf, fastexec, ref, tritonexec  # noqa: E402
from chipformer.rv.measure import DATA  # noqa: E402
from chipformer.rv.programs import ALL, program  # noqa: E402


def prog_of(name):
    return program(name) if name in ALL else elf.program(DATA / f"{name}.elf")


def main():
    print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
    c = core.build()
    for name, cap in [("signext", None), ("memcpy", None), ("call", None),
                      ("rv32ui-add", None), ("c-crc32", 1500)]:
        prog = prog_of(name)
        core.snapshot(c, ref.Machine.from_program(prog))
        toks = ref.Machine.from_program(prog).run(10**6)
        toks = toks[:cap] if cap else toks
        t = time.time()
        try:
            n = tritonexec.check(c, toks)
            print(f"tritonexec.check {name:12s} exact on {n} positions ({time.time() - t:.1f} s)",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"tritonexec.check {name:12s} FAILED: {type(e).__name__}: {str(e)[:400]}",
                  flush=True)
            traceback.print_exc(limit=3)
    prog = prog_of("c-crc32")
    core.snapshot(c, ref.Machine.from_program(prog))
    toks = ref.Machine.from_program(prog).run(10**6)
    for b in (1, 8, 32):
        try:
            s = fastexec.timed2(c, toks, steps=500, batch=b)
            print(f"executor (Triton, HIP graph) batch {b:2d}: {s * 1e3:.3f} ms/step, "
                  f"{b / s:,.0f} tok/s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"timed2 batch {b} FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
            traceback.print_exc(limit=3)


if __name__ == "__main__":
    main()
