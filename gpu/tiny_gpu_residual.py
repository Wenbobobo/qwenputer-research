"""Diagnose tinycheck's residual == IR check on ROCm with the saved rv-tiny checkpoint.

Reports, per program and mode, how many (dim, position) cells differ from the IR evaluator and
the max |hf - ir|, instead of asserting. Modes:
  prefill          one chunked prefill after snapshot injection (tinycheck's mode)
  prefill-strict   same with bf16/fp16 reduced-precision matmul reductions disabled
  decode           token-by-token (the path generate uses)
  cpu-prefill      reference: the same on CPU (optional, --cpu)
"""
import argparse
import sys
import time

RV = "/mnt/c/Projects/Qcompute/repro/qwenputer/experimental/riscv"
sys.path.insert(0, RV)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import Qwen3_5ForCausalLM  # noqa: E402

from chipformer.q35 import ir  # noqa: E402
from chipformer.q35.tiny import pick_layout  # noqa: E402
from chipformer.rv import core, elf, firmware, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402
from chipformer.rv.measure import DATA  # noqa: E402
from chipformer.rv.programs import ALL, program  # noqa: E402


def hidden(model, c, prog, toks, mode):
    dev = next(model.parameters()).device
    cache = firmware.load_cache(model, c, ref.Machine.from_program(prog))
    x = torch.tensor([toks], device=dev)
    with torch.no_grad():
        if mode == "decode":
            hs = []
            for t in range(len(toks)):
                o = model(input_ids=x[:, t:t + 1], past_key_values=cache, use_cache=True,
                          output_hidden_states=True)
                hs.append(o.hidden_states[c.n_layers() - 1][0])
            h = torch.cat(hs)
        else:
            o = model(input_ids=x, past_key_values=cache, use_cache=True,
                      output_hidden_states=True)
            h = o.hidden_states[c.n_layers() - 1][0]
    return h.double().cpu().numpy()


def compare(model, c, prog, mode):
    toks = ref.Machine.from_program(prog).run(10**6)
    core.snapshot(c, ref.Machine.from_program(prog))
    want = ir.evaluate(c, toks, check=False)
    h = hidden(model, c, prog, toks, mode)
    lay = model.layout
    dims = [d for d in range(1, len(c.dims))
            if c.dims[d].level < 2 * c.n_layers() - 1 and d not in c.keep]
    cols = [lay.dim(d) for d in dims]
    diff = np.abs(h[:, cols] - want[:, dims])
    bad = diff > 0
    nbad_dims = int(bad.any(0).sum())
    first = None
    if bad.any():
        t, i = map(int, np.argwhere(bad)[0])
        first = f"pos {t} ({S.name(toks[t])}) dim {c.dims[dims[i]].name}"
    lv = {}
    for i in np.nonzero(bad.any(0))[0]:
        L = c.dims[dims[i]].level
        lv[L] = lv.get(L, 0) + 1
    names = f"levels(bad dims) {dict(sorted(lv.items()))}; min level dims " + str(sorted(
        {c.dims[dims[i]].name for i in np.nonzero(bad.any(0))[0]
         if c.dims[dims[i]].level == min(lv)})[:6]) if lv else "-"
    return len(toks), len(dims) * len(toks), int(bad.sum()), nbad_dims, float(diff.max()), first, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--progs", default="signext,memcpy")
    ap.add_argument("--modes", default="prefill,prefill-strict,decode")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    model = Qwen3_5ForCausalLM.from_pretrained(f"{RV}/dist/rv-tiny", dtype=torch.bfloat16).eval()
    c = core.build()
    model.layout = pick_layout(len(c.dims))
    devs = ["cuda"] + (["cpu"] if args.cpu else [])
    for dev in devs:
        model = model.to(dev)
        for name in args.progs.split(","):
            prog = program(name) if name in ALL else elf.program(DATA / f"{name}.elf")
            for mode in args.modes.split(","):
                strict = mode == "prefill-strict"
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not strict
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = not strict
                t = time.time()
                n, cells, nbad, nbd, mx, first, names = compare(model, c, prog, mode)
                print(f"{dev:4s} {name:10s} {mode:15s} {n} pos, {cells} cells: {nbad} cells / "
                      f"{nbd} dims differ, max|d| {mx:.3g}, first {first}, dim families {names} "
                      f"({time.time() - t:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
