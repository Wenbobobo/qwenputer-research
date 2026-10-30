"""Aggregate H1 jsonl -> h1_results.json (Wilson 95% CIs, Fisher exact for C1 vs C3, C2 vs C4).
    python analysis/experiments/h1/h1_analyze.py [outdir]"""
import json
import math
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "analysis/experiments/h1/out")
# Brevity probes C1s/C3s are separate controls, excluded from primary comparisons.
# The full comparison is seed 0 for all four requested conditions plus seed 1/2 context variants.
# Brevity probes C1s/C3s are separate controls and are excluded from primary comparisons.
INCLUDE_PROBE = False
OUT = Path("analysis/experiments/h1/h1_results.json")
NAMES = {"C1": "timestamped context + most-recent-wins", "C2": "append-only host tools",
         "C3": "plain pasted context (upstream 'context')", "C4": "overwrite host tools (upstream 'host')",
         "C1s": "C1 + 'Answer in one short sentence.'", "C3s": "C3 + 'Answer in one short sentence.'"}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def fisher(a, b, c, d):  # two-sided, table [[a,b],[c,d]]
    n1, n2, k = a + b, c + d, a + c
    def hp(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)
    p0 = hp(a)
    return float("%.3g" % min(1.0, sum(hp(x) for x in range(max(0, k - n2), min(k, n1) + 1)
                              if hp(x) <= p0 * (1 + 1e-9))))


import re


def own_from_text(t):
    """Model-generated words only: drop injected user/tool turns, tool calls, think, markup."""
    t = re.sub(r"<\|im_start\|>user\n.*?<\|im_end\|>", " ", t, flags=re.S)
    t = re.sub(r"<tool_call>.*?</tool_call>", " ", t, flags=re.S)
    t = re.sub(r"<tool_response>.*?</tool_response>", " ", t, flags=re.S)
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S)
    t = re.sub(r"<\|im_start\|>assistant\n|<\|im_end\|>|<\|endoftext\|>", " ", t)
    return t.strip()


def rescore(r):
    a = own_from_text(r["text"]).lower()
    g = r["gold"].lower()
    r["answer_own"] = a
    r["correct"] = g in a and (r["old"] is None or r["old"].lower() not in a)
    r["has_new"] = g in a
    if r["old"]:
        o = r["old"].lower()
        pn, po = a.find(g), a.find(o)
        # secondary, lenient: the new value is stated and appears before any mention of the old one
        r["new_first"] = pn >= 0 and (po < 0 or pn < po)
    return r


res = {"conditions": {}, "comparisons": {}}
recs = {}
for f in sorted(D.glob("h1.C*.seed*.jsonl")):  # all seeds pooled per condition
    rs = [rescore(json.loads(x)) for x in f.read_text(encoding="utf-8").splitlines() if x]
    c = rs[0]["cond"] if rs else f.name.split(".")[1]
    if c.endswith("s"):
        continue
    recs.setdefault(c, []).extend(rs)
for c, rs in sorted(recs.items()):
    upd = [r for r in rs if r["kind"] == "update"]
    fac = [r for r in rs if r["kind"] == "fact"]
    def blk(xs, key="correct"):
        k = sum(bool(r[key]) for r in xs)
        return {"k": k, "n": len(xs), "acc": round(k / len(xs), 3) if xs else None, "wilson95": wilson(k, len(xs))}
    e = {"name": NAMES[c], "cases": len({r["key"].split("/")[0] for r in rs}),
         "changed": blk(upd), "changed_new_value_present": blk(upd, "has_new"),
         "changed_new_first_lenient": blk(upd, "new_first"),
         "facts": blk(fac), "all": blk(rs)}
    upd_wrong = [r for r in upd if not r["correct"]]
    e["changed_errors"] = {
        "old_only": sum(1 for r in upd_wrong if r["old"].lower() in r["answer_own"] and not r["has_new"]),
        "both_old_and_new": sum(1 for r in upd_wrong if r["old"].lower() in r["answer_own"] and r["has_new"]),
        "neither": sum(1 for r in upd_wrong if r["old"].lower() not in r["answer_own"] and not r["has_new"])}
    if c in ("C2", "C4"):
        first = [r for r in rs if r.get("sessions")]
        sess_of = {(r["seed"], r["key"].split("/")[0]): r["sessions"] for r in first}
        mech = {}
        for r in upd:
            ss = sess_of[(r["seed"], r["key"].split("/")[0])]
            saved = " ".join(json.dumps(cl["args"]).lower() for s in ss for cl in s["calls"]
                             if cl["fn"] in ("put", "append", "add_entry"))
            o, n = r["old"].lower() in saved, r["gold"].lower() in saved
            k = f"old_saved={o},new_saved={n}"
            mech.setdefault(k, {"n": 0, "correct": 0})
            mech[k]["n"] += 1
            mech[k]["correct"] += int(r["correct"])
        e["changed_by_what_was_saved"] = mech
        e["tool_stats"] = {
            "sessions_with_save": sum(any(cl["fn"] in ("put", "append", "add_entry") for cl in s["calls"])
                                      for r in first for s in r["sessions"]),
            "sessions_total": sum(len(r["sessions"]) for r in first),
            "question_turns_with_read": sum(any(cl["fn"] in ("get", "list", "get_entries", "list_entries")
                                                for cl in r["calls"]) for r in rs),
            "question_turns": len(rs)}
    e["per_seed_changed"] = {str(s): blk([r for r in upd if r["seed"] == s])
                             for s in sorted({r["seed"] for r in rs})}
    e["per_seed_changed_lenient"] = {str(s): blk([r for r in upd if r["seed"] == s], "new_first")
                                     for s in sorted({r["seed"] for r in rs})}
    res["conditions"][c] = e
C = res["conditions"]
for a, b in (("C1", "C3"), ("C4", "C2"), ("C1", "C4"), ("C2", "C3")):
    if a in C and b in C:
        x, y = C[a]["changed"], C[b]["changed"]
        res["comparisons"][f"{a}_vs_{b}_changed"] = {
            "diff": round(x["acc"] - y["acc"], 3),
            "fisher_p": fisher(x["k"], x["n"] - x["k"], y["k"], y["n"] - y["k"])}
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))
