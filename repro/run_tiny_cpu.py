"""Free-run rv-tiny (saved by tinycheck --save) on CPU for selected programs and compare the
generated token stream with the reference trace. Upstream code is used unmodified.

    cd repro/qwenputer/experimental/riscv
    uv run python -X faulthandler ../../../run_tiny_cpu.py dist/rv-tiny signext call rv32ui-simple
"""
import sys
import time

import torch
from transformers import Qwen3_5ForCausalLM

from chipformer.rv import core, elf, firmware, ref
from chipformer.rv.measure import DATA
from chipformer.rv.programs import program


def main():
    path, names = sys.argv[1], sys.argv[2:]
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}", flush=True)
    t = time.time()
    model = Qwen3_5ForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()
    c = core.build()
    print(f"loaded {path} + circuit ({time.time() - t:.0f} s)", flush=True)
    firmware.run(model, c, ref.Machine.from_program(program("signext")), max_new=4)  # warm-up
    for name in names:
        prog = program(name) if "-" not in name else elf.program(DATA / f"{name}.elf")
        want = ref.Machine.from_program(prog).run(10**6)
        t = time.time()
        got = firmware.run(model, c, ref.Machine.from_program(prog), max_new=len(want) + 5)
        dt = time.time() - t
        ok = got == want[1:]
        first_bad = next((i for i, (a, b) in enumerate(zip(got, want[1:])) if a != b), None)
        out = firmware.output_text(got)
        print(f"{name:14s} generate == reference: {ok} ({len(got)} generated / {len(want) - 1} "
              f"expected tokens, first mismatch {first_bad}; {dt:.1f} s, "
              f"{len(got) / dt:.2f} tok/s{', output ' + repr(out[:40]) if out else ''})", flush=True)


if __name__ == "__main__":
    main()
