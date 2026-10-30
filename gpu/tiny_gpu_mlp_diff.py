"""Isolate the ROCm-vs-CPU difference in rv-tiny layer 0's MLP (gate/up GEMMs, SiLU, product,
down GEMM) on the actual signext inputs captured at the MLP entry."""
import copy
import sys

RV = "/mnt/c/Projects/Qcompute/repro/qwenputer/experimental/riscv"
sys.path.insert(0, RV)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import Qwen3_5ForCausalLM  # noqa: E402

from chipformer.rv import core, firmware, ref  # noqa: E402
from chipformer.rv.programs import program  # noqa: E402


def stats(name, a, b):
    a, b = a.double().cpu(), b.double().cpu()
    d = (a - b).abs()
    print(f"{name:22s} differ {int((d > 0).sum())}/{d.numel()}  max|d| {d.max().item():.3g}  "
          f"nonzero(gpu) {int((a != 0).sum())} nonzero(cpu) {int((b != 0).sum())}", flush=True)


def main():
    model = Qwen3_5ForCausalLM.from_pretrained(f"{RV}/dist/rv-tiny", dtype=torch.bfloat16).eval()
    c = core.build()
    prog = program("signext")
    toks = ref.Machine.from_program(prog).run(10**6)
    mlp_cpu = copy.deepcopy(model.model.layers[0].mlp)
    model = model.cuda()
    cap = {}
    h = model.model.layers[0].mlp.register_forward_hook(
        lambda m, i, o: cap.__setitem__("x", i[0].detach()))
    cache = firmware.load_cache(model, c, ref.Machine.from_program(prog))
    cap.clear()
    with torch.no_grad():
        model(input_ids=torch.tensor([toks], device="cuda"), past_key_values=cache, use_cache=True)
    h.remove()
    x = cap["x"][0]
    mg = model.model.layers[0].mlp
    print("act", mg.act_fn, "x", tuple(x.shape), x.dtype, flush=True)
    with torch.no_grad():
        xc = x.cpu()
        g_g, g_c = mg.gate_proj(x), mlp_cpu.gate_proj(xc)
        u_g, u_c = mg.up_proj(x), mlp_cpu.up_proj(xc)
        stats("gate GEMM", g_g, g_c)
        stats("up GEMM", u_g, u_c)
        # elementwise on identical inputs
        a_g, a_c = F.silu(g_c.cuda()), F.silu(g_c)
        stats("silu (same input)", a_g, a_c)
        p_g, p_c = a_c.cuda() * u_c.cuda(), a_c * u_c
        stats("act*up (same input)", p_g, p_c)
        d_g, d_c = mg.down_proj(p_c.cuda()), mlp_cpu.down_proj(p_c)
        stats("down GEMM (same input)", d_g, d_c)
        # fp32-accumulated reference of down GEMM
        ref32 = (p_c.double() @ mlp_cpu.down_proj.weight.double().T)
        stats("down cpu vs fp64", d_c, ref32)
        stats("down gpu vs fp64", d_g, ref32)
        stats("gate gpu vs fp64", g_g, xc.double() @ mlp_cpu.gate_proj.weight.double().T)
        stats("gate cpu vs fp64", g_c, xc.double() @ mlp_cpu.gate_proj.weight.double().T)
        stats("MLP total", mg(x), mlp_cpu(xc))
        # same GEMM in fp32 on GPU, rounded to bf16
        d32 = (p_c.cuda().float() @ mg.down_proj.weight.float().T).bfloat16()
        stats("down gpu fp32->bf16", d32, d_c)


if __name__ == "__main__":
    main()
