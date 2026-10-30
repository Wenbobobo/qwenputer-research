"""Stage 4 GPU: free-run rv-tiny (saved checkpoint) on ROCm (bf16) and compare the generated token
stream with the reference trace (ref.Machine.from_program(...).run(10**6)). Upstream chipformer
code is used unmodified (firmware.run = plain greedy HF generate after snapshot injection).

    wsl -d Ubuntu-24.04 -- bash /mnt/c/Projects/Qcompute/gpu/run_py.sh gpu/tiny_gpu_free.py \
        [--progs rv32ui-simple,memcpy,signext] [--json gpu/out/tiny_gpu_free.json]
"""
import argparse
import json
import os
import sys
import time

RV = "/mnt/c/Projects/Qcompute/repro/qwenputer/experimental/riscv"
sys.path.insert(0, RV)

import torch  # noqa: E402
from transformers import Qwen3_5ForCausalLM  # noqa: E402

from chipformer.rv import core, elf, firmware, ref  # noqa: E402
from chipformer.rv.measure import DATA  # noqa: E402
from chipformer.rv.programs import ALL, program  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{RV}/dist/rv-tiny")
    ap.add_argument("--progs", default="rv32ui-simple,memcpy,signext")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    try:
        import fla  # noqa: F401
        kernels = "fla"
    except ImportError:
        kernels = "torch fallbacks (no fla / causal-conv1d)"
    info = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu", "kernels": kernels, "dtype": "bf16"}
    print(info, flush=True)
    t = time.time()
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    t_load = time.time() - t
    t = time.time()
    model = model.to(args.device)
    sync()
    t_dev = time.time() - t
    t = time.time()
    c = core.build()
    t_circ = time.time() - t
    n = sum(p.numel() for p in model.parameters())
    mem = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"loaded {n / 1e6:.0f}M params: from_pretrained {t_load:.0f} s, to({args.device}) "
          f"{t_dev:.0f} s, circuit build {t_circ:.0f} s, GPU mem {mem:.2f} GiB", flush=True)
    t = time.time()
    firmware.run(model, c, ref.Machine.from_program(program("signext")), max_new=4)  # warm-up
    sync()
    print(f"warm-up {time.time() - t:.1f} s", flush=True)
    rows = []
    for name in args.progs.split(","):
        prog = program(name) if name in ALL else elf.program(DATA / f"{name}.elf")
        want = ref.Machine.from_program(prog).run(10**6)
        sync()
        t = time.time()
        got = firmware.run(model, c, ref.Machine.from_program(prog), max_new=len(want) + 5)
        sync()
        dt = time.time() - t
        ok = got == want[1:]
        first_bad = next((i for i, (a, b) in enumerate(zip(got, want[1:])) if a != b), None)
        out = firmware.output_text(got)
        row = {"prog": name, "generated": len(got), "expected": len(want) - 1, "exact": ok,
               "first_mismatch": first_bad, "wall_s": round(dt, 2),
               "tok_s": round(len(got) / dt, 2), "output": out[:40]}
        rows.append(row)
        print(f"{name:14s} generate == reference: {ok} ({len(got)} generated / {len(want) - 1} "
              f"expected, first mismatch {first_bad}; {dt:.1f} s, {len(got) / dt:.2f} tok/s"
              f"{', output ' + repr(out[:40]) if out else ''})", flush=True)
    if torch.cuda.is_available():
        info["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        print(f"peak GPU mem {info['peak_mem_gib']} GiB", flush=True)
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"info": info, "load_s": [t_load, t_dev, t_circ], "rows": rows}, f, indent=1)
    sys.exit(0 if all(r["exact"] for r in rows) else 1)


if __name__ == "__main__":
    main()
