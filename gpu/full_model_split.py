"""Full published model (Qwen3.5-4B-Qwenputer-Filesystem, dense 22 GB bf16) on a 20 GB RX 7900 XT.

accelerate is not installed in the ROCm venv, so device_map='auto' is unavailable. Instead a
manual split: the tied embedding / lm_head (484,800 x 4,608 bf16 = 4.47 GB) stays on the CPU,
every decoder layer + final norm + rotary goes to the GPU. A short prompt is prefilled and a few
greedy tokens are decoded (no long generation). Reports byte arithmetic, placement, GPU memory,
load time and per-token latency.
"""
import json
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn

MODEL = "/mnt/c/Projects/Qcompute/.data/Qwen3.5-4B-Qwenputer-Filesystem"
DEADLINE = "12:15"  # Beijing; refuse to start GPU work after this


class CpuEmbed(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.emb = emb

    def forward(self, ids):
        return self.emb(ids.cpu()).to("cuda")


class CpuHead(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, h):
        return F.linear(h.to("cpu"), self.weight)


def arithmetic():
    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))
    from safetensors import safe_open
    files = sorted(set(idx["weight_map"].values()))
    total, emb, groups = 0, 0, {}
    for f in files:
        with safe_open(f"{MODEL}/{f}", "pt") as sf:
            for k in sf.keys():
                sl = sf.get_slice(k)
                n = 1
                for s in sl.get_shape():
                    n *= s
                b = n * 2  # bf16 (checked below)
                total += b
                if "embed_tokens" in k or "lm_head" in k:
                    emb += b
                g = "mtp" if k.startswith("mtp") else ("visual" if "visual" in k else "lm")
                groups[g] = groups.get(g, 0) + b
    free, tot = torch.cuda.mem_get_info()
    print(f"weights {total / 1e9:.2f} GB = {total / 2**30:.2f} GiB (groups {groups}); "
          f"tied embedding/lm_head {emb / 1e9:.2f} GB; GPU total {tot / 2**30:.2f} GiB, "
          f"free {free / 2**30:.2f} GiB; dense fits: {total < free}; "
          f"without embedding {(total - emb) / 2**30:.2f} GiB", flush=True)
    return total, emb


def main():
    now = time.strftime("%H:%M", time.localtime(time.time() + 8 * 3600 - (-time.timezone)))
    print("Beijing time", now, flush=True)
    if now > DEADLINE:
        sys.exit("past deadline, not starting")
    arithmetic()
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    t = time.time()
    model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    print(f"loaded to CPU in {time.time() - t:.0f} s; dtype {model.dtype}; "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.3f}B params; "
          f"tied {model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()}",
          flush=True)
    t = time.time()
    emb_w = model.model.embed_tokens.weight
    model.model.embed_tokens = CpuEmbed(model.model.embed_tokens)
    model.lm_head = CpuHead(emb_w)
    for name, mod in model.model.named_children():
        if name != "embed_tokens":
            mod.to("cuda")
    torch.cuda.synchronize()
    gpu = torch.cuda.memory_allocated() / 2**30
    cpu_b = emb_w.numel() * 2 / 2**30
    print(f"placed in {time.time() - t:.0f} s: GPU {gpu:.2f} GiB (layers+norm), "
          f"CPU {cpu_b:.2f} GiB (embedding/lm_head)", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    msgs = [{"role": "user", "content": "List the files in /home and tell me what 12*34 is."}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception as e:  # noqa: BLE001
        print("chat template failed:", e)
        text = "The capital of France is"
    ids = tok(text, return_tensors="pt").input_ids
    print(f"prompt {ids.shape[1]} tokens", flush=True)
    out_ids = []
    with torch.no_grad():
        t = time.time()
        o = model(input_ids=ids.to("cuda"), use_cache=True)
        torch.cuda.synchronize()
        t_pre = time.time() - t
        cache = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
        lat = []
        for _ in range(24):
            out_ids.append(nxt)
            t = time.time()
            o = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=cache,
                      use_cache=True)
            torch.cuda.synchronize()
            lat.append(time.time() - t)
            cache = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
    peak = torch.cuda.max_memory_allocated() / 2**30
    lat_ss = sorted(lat[2:])
    med = lat_ss[len(lat_ss) // 2]
    print(f"prefill {t_pre:.2f} s; decode median {med * 1e3:.0f} ms/token = {1 / med:.1f} tok/s "
          f"(text path, CPU lm_head); peak GPU {peak:.2f} GiB", flush=True)
    print("greedy text:", repr(tok.decode(out_ids)), flush=True)
    # where does the time go: lm_head on CPU vs layers on GPU
    h = torch.randn(1, 1, model.config.hidden_size, dtype=torch.bfloat16, device="cuda")
    t = time.time()
    for _ in range(5):
        model.lm_head(h)
    print(f"CPU lm_head alone: {(time.time() - t) / 5 * 1e3:.0f} ms/token", flush=True)


if __name__ == "__main__":
    main()
