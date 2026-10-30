"""Single-process CPU demo for the released Qwenputer-Filesystem checkpoint.

Loads the model once, then:
  A) put a fact ("my locker code is 4471-AZ") and save the store snapshot;
  B) start a fresh session with the snapshot loaded and ask for it back;
  C) run one calc call through the in-weights RISC-V CPU.

Tailored for a 48 GB CPU box: the model is ~21 GB bf16, so keep one process.
"""
import json
import sys
import time
from pathlib import Path

from chipformer.rv import bridge as BR
from chipformer.rv import chat as CH
from chipformer.rv import elf, firmware, ref
from chipformer.rv import store as ST

MODEL = r"C:\Projects\Qcompute\.data\Qwen3.5-4B-Qwenputer-Filesystem"
FW = "tests/data/rv/calc-bridge.elf"
OUT = Path(r"C:\Projects\Qcompute\demo")


def make_machine():
    m = ref.Machine.from_program(elf.program(FW))
    BR.park(m)
    return m


def run(chat, tools, ask, max_new, lock=None, store_snap=None, tag=""):
    m = make_machine()
    msgs = [{"role": "user", "content": ask}]
    if chat.store_cfg is not None:
        msgs = [{"role": "system", "content": ST.MEM_SYSTEM}] + msgs
    ids = chat.prompt_ids(msgs, tools)
    if lock is not None:
        lock.history(ids)
    chat.store_snap = store_snap
    t0 = time.time()
    r = chat.generate(ids, m, max_new, check_head=True, verbose=False, lock=lock)
    txt = chat.text_of(r["tokens"])
    print(f"=== {tag} ({time.time()-t0:.0f}s, {r['text_tokens']} text + {r['chip_tokens']} chip, "
          f"head_mismatch {r['head_mismatch']}, lock_error {r['lock_error']}) ===", flush=True)
    print(txt, flush=True)
    return r, txt


def main():
    t0 = time.time()
    chat = CH.RvChat(MODEL, fast=False)  # CPU: HF modules, no executor
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)
    store = chat.store_cfg is not None
    tools = (CH.CALC_TOOL,) if not store else (CH.CALC_TOOL, *ST.TOOLS)
    results = {}
    if store:
        lock = ST.StoreSession(chat.T, make_machine(), chat.store_cfg)
        r, txt = run(chat, tools, "Remember that my locker code is 4471-AZ.", 120,
                     lock=lock, tag="A put")
        results["A"] = txt
        snap = chat.save_store()
        CH.save_store_file(OUT / "store.npz", snap)
        print("snapshot saved:", (OUT / "store.npz").stat().st_size, "bytes", flush=True)
        r, txt = run(chat, tools, "What's my locker code?", 120,
                     lock=None, store_snap=snap, tag="B get (fresh context)")
        results["B"] = txt
    r, txt = run(chat, tools, "Use calc: what is 123456789 * 987654321?", 300,
                 lock=None, tag="C calc")
    results["C"] = txt
    (OUT / "driver-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()