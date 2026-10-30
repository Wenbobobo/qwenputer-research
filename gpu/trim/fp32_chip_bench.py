"""C8 mixed-dtype benchmark: standard prompts plus matched H1 history sessions."""
import argparse, copy, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis/experiments/h1"))
import torch
# Limit CPU inspection/compaction thread overhead on the WSL host.
torch.set_num_threads(8)
from gpu.trim.common import CKPT
from gpu.trim.trim_chat import install
from h1_gen import recall_cases

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="1,2,5")
    ap.add_argument("--deadline", type=float, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report = {"fp32_chip": os.environ.get("TRIM_FP32_CHIP") == "1", "start": time.time()}
    torch.cuda.reset_peak_memory_stats()
    install(report)
    from chipformer.rv import chat, ref, elf, bridge as BR, store as ST
    t = time.perf_counter(); c = chat.RvChat(CKPT, device="cuda", fast=True)
    report["init_s"] = time.perf_counter() - t
    firmware = ROOT / "repro/qwenputer/experimental/riscv/tests/data/rv/calc-bridge.elf"
    def machine():
        m = ref.Machine.from_program(elf.program(firmware)); BR.park(m); return m
    audit = {}
    for i, layer in enumerate(c.lm.layers):
        audit[str(i)] = sorted({str(p.dtype) for p in layer.parameters()})
    audit["embedding"] = str(c.lm.embed_tokens.weight.dtype)
    audit["lm_head"] = str(c.model.lm_head.weight.dtype)
    report["dtype_audit"] = audit
    (out / "runtime.json").write_text(json.dumps(report, indent=2, default=str))
    def turn(label, user, snap=None, dev=None, budget=256):
        ids = c.prompt_ids([{"role":"system", "content":ST.MEM_SYSTEM},
                            {"role":"user", "content":user}], (chat.CALC_TOOL, *ST.TOOLS))
        lock = ST.StoreSession(c.T, machine(), c.store_cfg, copy.deepcopy(dev))
        lock.history(ids); c.store_snap = snap
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            r = c.generate(ids, machine(), budget, max_chip=200000, check_head=True, lock=lock)
        torch.cuda.synchronize(); wall = time.perf_counter() - t0
        rec = {"label": label, "user": user, "prompt_tokens":len(ids), "text":c.text_of(r["tokens"]),
               "wall_s":wall, "tok_s":r["text_tokens"] / r["seconds"], **r}
        with (out / "turns.jsonl").open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print("[C8]", label, "tok/s", rec["tok_s"], "lock", r["lock_error"],
              "head", r["head_mismatch"], "text", rec["text"][-140:], flush=True)
        return c.save_store(), lock.dev
    turn("Hi", "Hi", budget=12)
    turn("Hi-warm", "Hi", budget=12)
    snap, dev = turn("put", "Please put my locker code 4471-AZ in a file named locker_code.", budget=120)
    chat.save_store_file(str(out / "standard.npz"), snap)
    turn("fresh-get", "What is my locker code? Please get locker_code.",
         chat.load_store_file(str(out / "standard.npz")), dev, budget=120)
    cases = recall_cases(10, 0)
    for idx in map(int, a.cases.split(",")):
        if a.deadline and time.time() + 150 > a.deadline:
            print("[C8] deadline before case", idx, flush=True); break
        snap = dev = None
        for j, user in enumerate(cases[idx]["sessions"]):
            if a.deadline and time.time() + 45 > a.deadline:
                print("[C8] deadline before session", flush=True); break
            snap, dev = turn(f"recall-{idx}/s{j+1}", user, snap, dev)
    report["chip_cache_dtypes"] = [{"conv": str(x.conv_states[0].dtype),
        "recurrent": str(x.recurrent_states[0].dtype)} for x in c.ccache.layers[:c.n_chip]]
    report["peak_bytes"] = torch.cuda.max_memory_allocated()
    report["end"] = time.time()
    (out / "runtime.json").write_text(json.dumps(report, indent=2, default=str))
    print("[C8] END", report["peak_bytes"], flush=True)

if __name__ == "__main__":
    main()
