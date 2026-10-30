"""H1 in-weights condition (C5): the released Qwenputer filesystem checkpoint with its OWN
in-weights store (RvChat store heads, MEM_SYSTEM + [CALC_TOOL] + ST.TOOLS), on the Stage-6 trim
GPU runtime (gpu/trim/trim_chat.install: lossless Qwen trim + exact chip compact, fast path).

    wsl ... bash gpu/run_py.sh analysis/experiments/h1_inweights/run_inw.py --n 10 --deadline EPOCH

Protocol = upstream storebench.run_recall --cond store (StoreModel.turn), with H1 budgets:
  * each of the 4 past sessions is a FRESH conversation [MEM_SYSTEM, user session] with the store
    heads' snapshot (chat.save_store) and the lockstep reference device carried over;
  * after session 4 the snapshot is written to npz; each question is a fresh conversation
    [MEM_SYSTEM, question] whose store snapshot is LOADED FROM THAT NPZ (as a fresh process would);
  * greedy, thinking disabled (RvChat template default), session budget 256 text tokens,
    answer budget 200 text tokens (= H1 C2/C4), max_chip 200000, check_head + StoreSession lock.
Scoring: H1 strict = upstream score_fact on own_text (tool calls/responses/think removed);
lenient new-first = new value present and before any old value.
Cases: storebench recall_cases(n, seed=0) via analysis/experiments/h1/h1_gen.py (= H1 C2/C4 cases).
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis/experiments/h1"))
from h1_gen import own_text, recall_cases, score_fact  # noqa: E402

import torch  # noqa: E402
from gpu.trim.common import CKPT  # noqa: E402  (also blocks fla unless TRIM_ALLOW_FLA=1)
from gpu.trim.trim_chat import install  # noqa: E402

SESSION_BUDGET = 256
ANSWER_BUDGET = 200


def new_first(ans, q):
    t = ans.lower()
    g = t.find(q["gold"].lower())
    if g < 0:
        return False
    if q["old"] is None:
        return True
    o = t.find(q["old"].lower())
    return o < 0 or g < o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--deadline", type=float, default=0, help="epoch s; no new case after it")
    ap.add_argument("--out", default=str(HERE / "out"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "npz").mkdir(parents=True, exist_ok=True)
    qf, sf = out / "inw_questions.jsonl", out / "inw_sessions.jsonl"
    done = set()
    if qf.exists():
        done = {json.loads(x)["key"] for x in qf.read_text(encoding="utf-8").splitlines() if x}

    report = {"fla_allowed": os.environ.get("TRIM_ALLOW_FLA") == "1"}
    torch.cuda.reset_peak_memory_stats()
    install(report)
    report['runtime_mode']='no-FLA trim fast GPU (no head trim)'
    from chipformer.rv import bridge as BR, chat, elf, ref, store as ST
    t = time.perf_counter()
    c = chat.RvChat(CKPT, device="cuda", fast=True)
    report["init_s"] = time.perf_counter() - t
    report["seed"] = args.seed
    report["case_range"] = [args.start, args.n]
    report["session_budget"] = SESSION_BUDGET
    report["answer_budget"] = ANSWER_BUDGET
    firmware = ROOT / "repro/qwenputer/experimental/riscv/tests/data/rv/calc-bridge.elf"
    tools = (chat.CALC_TOOL, *ST.TOOLS)
    print("[inw] ready", {k: v for k, v in report.items() if k != "compact"}, flush=True)
    (out / "runtime.json").write_text(json.dumps(report, indent=1, default=str))

    def machine():
        m = ref.Machine.from_program(elf.program(firmware))
        BR.park(m)
        return m

    def turn(user, snap, dev, budget):
        ids = c.prompt_ids([{"role": "system", "content": ST.MEM_SYSTEM},
                            {"role": "user", "content": user}], tools)
        lock = ST.StoreSession(c.T, machine(), c.store_cfg, copy.deepcopy(dev))
        lock.history(ids)
        c.store_snap = snap
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            r = c.generate(ids, machine(), budget, max_chip=200000, check_head=True, lock=lock)
        torch.cuda.synchronize()
        text = c.text_of(r["tokens"])
        info = {"store_calls": [a[0] for a in lock.answers], "calc_runs": len(lock.runs),
                "lock_error": r["lock_error"], "head_mismatch": r["head_mismatch"],
                "text_tokens": r["text_tokens"], "chip_tokens": r["chip_tokens"],
                "seconds": round(time.time() - t0, 2), "prompt_tokens": len(ids)}
        return text, info, c.save_store(), lock.dev

    def store_view(dev):
        try:
            return {c.T.decode_bytes(k[1][:k[0]]).decode("utf-8", "replace").strip(): c.T.decode_bytes(dev.text(k) or []).decode("utf-8", "replace").strip()
                    for k in dev.names()}
        except Exception as e:  # diagnostic only
            return {"<error>": repr(e)}

    cases = recall_cases(args.n, args.seed)[args.start:]
    est = 300.0
    for case in cases:
        if all(f"{case['id']}/{j}" in done for j in range(3)):
            continue
        left = args.deadline - time.time() if args.deadline else 1e9
        if left < est * 1.15:
            print(f"[inw] deadline: {left:.0f}s left < est {est:.0f}s; stop before {case['id']}",
                  flush=True)
            break
        tc = time.time()
        snap, dev, sess = None, None, []
        for k, s_ in enumerate(case["sessions"]):
            text, info, snap, dev = turn(s_, snap, dev, SESSION_BUDGET)
            sess.append({"session": k + 1, "user": s_, "text": text, **info,
                         "store_after": store_view(dev)})
            print(f"[inw] {case['id']} s{k + 1} {info['seconds']}s calls={info['store_calls']} "
                  f"lock={info['lock_error']} hm={info['head_mismatch']}", flush=True)
        npz = out / "npz" / f"{case['id']}.npz"
        chat.save_store_file(str(npz), snap)
        with open(sf, "a", encoding="utf-8") as f:
            f.write(json.dumps({"case": case["id"], "sessions": sess}, ensure_ascii=False,
                               default=str) + "\n")
        qsnap = chat.load_store_file(str(npz))  # questions start from the saved file
        final = sess[-1]["store_after"]
        for j, q in enumerate(case["questions"]):
            key = f"{case['id']}/{j}"
            if key in done:
                continue
            text, info, _, _ = turn(q["q"], qsnap, dev, ANSWER_BUDGET)
            ans = own_text(text)
            rec = {"key": key, "cond": "C5_inweights", "kind": q["kind"], "q": q["q"],
                   "gold": q["gold"], "old": q["old"], "correct": score_fact(ans, q),
                   "new_first": new_first(ans, q), "answer": ans[:500], "text": text,
                   "info": info,
                   "gold_in_store": any(q["gold"].lower() in str(v).lower() for v in final.values()),
                   "old_in_store": (any(q["old"].lower() in str(v).lower() for v in final.values())
                                    if q["old"] else None),
                   "old_ever_stored": (any(q["old"].lower() in str(v).lower() for s in sess
                                           for v in s["store_after"].values())
                                       if q["old"] else None)}
            with open(qf, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            print(f"[inw] {key} {q['kind']} correct={rec['correct']} nf={rec['new_first']} "
                  f"{info['seconds']}s calls={info['store_calls']} ans={ans[:120]!r}", flush=True)
        dt_ = time.time() - tc
        est = dt_ if est == 300.0 else max(est, dt_)
        print(f"[inw] case {case['id']} done in {dt_:.0f}s", flush=True)
    print("[inw] END", {"max_memory_allocated_bytes": torch.cuda.max_memory_allocated()},
          flush=True)


if __name__ == "__main__":
    main()
