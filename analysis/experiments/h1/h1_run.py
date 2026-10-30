"""H1: does the changed-fact advantage come from overwrite semantics rather than in-weights medium?

    run_py.sh analysis/experiments/h1/h1_run.py --cond C1|C2|C3|C4 --n 30 --seed 0 --out DIR

C1 timestamped context + "most recent wins"; C2 append-only host tools (add_entry/list_entries/
get_entries); C3 plain pasted context (upstream "context"); C4 overwrite host tools (upstream
"host": put/append/get/list/delete in Python). Greedy, Qwen3.5-4B bf16, thinking disabled.
Scoring = upstream score_fact on the model's own text (tool calls/responses/think removed).
"""
import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h1_gen import own_text, recall_cases, score_fact  # noqa: E402

MODEL = "/mnt/c/Projects/Qcompute/.data/Qwen3.5-4B"
PLAIN_SYSTEM = "You are a helpful assistant."
BJ = dt.timezone(dt.timedelta(hours=8))
STOP_AT = (12, 22)  # Beijing; GPU quiet hours start 12:30

# --- C4: upstream store tools minus calc (put replaces) -----------------------------------
C4_SYSTEM = ("You have a memory that keeps texts across conversations, with the tools put, "
             "append, get, list and delete. When the user "
             "tells you something about themselves (facts, preferences, changes), save it in "
             "your memory with put, one short name per fact; when a fact changes, put the new "
             "value under the same name. Before you answer a question about the user, look in "
             "your memory first (list, then get).")
C4_TOOLS = [
    {"type": "function", "function": {
        "name": "put", "description": "Store a text under a name in your memory (replaces it). "
                                      "The memory keeps it across conversations.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "a short name"},
            "text": {"type": "string", "description": "the text (up to 104 tokens)"}},
            "required": ["name", "text"]}}},
    {"type": "function", "function": {
        "name": "append", "description": "Add a text at the end of a stored text.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "text": {"type": "string"}},
            "required": ["name", "text"]}}},
    {"type": "function", "function": {
        "name": "get", "description": "The text stored under a name, exactly.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                       "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "list", "description": "The stored names.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "delete", "description": "Forget a stored name.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                       "required": ["name"]}}},
]

# --- C2: append-only log (no overwrite, no delete) -----------------------------------------
C2_SYSTEM = ("You have a memory that keeps texts across conversations, with the tools add_entry, "
             "list_entries and get_entries. The memory is append-only: entries can never be "
             "changed or deleted, and get_entries returns every entry under a name in the order "
             "they were added. When the user tells you something about themselves (facts, "
             "preferences, changes), save it in your memory with add_entry, one short name per "
             "fact; when a fact changes, add the new value under the same name. Before you answer "
             "a question about the user, look in your memory first (list_entries, then "
             "get_entries).")
C2_TOOLS = [
    {"type": "function", "function": {
        "name": "add_entry", "description": "Add a new entry (a text) under a name in your memory. "
        "Existing entries are kept; nothing is replaced. The memory keeps it across conversations.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "a short name"},
            "text": {"type": "string", "description": "the text"}},
            "required": ["name", "text"]}}},
    {"type": "function", "function": {
        "name": "list_entries", "description": "The names that have entries.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_entries", "description": "All entries stored under a name, in the order they "
        "were added.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                       "required": ["name"]}}},
]

SESSION_DATES = ["2025-03-02 18:40", "2025-04-15 09:12", "2025-06-08 20:05", "2025-09-21 13:30"]


def ctx_user(case, q, timestamped):
    if timestamped:
        past = "\n\n".join(f"[Session {k + 1} — {SESSION_DATES[k]}]\nUser: {s}"
                            for k, s in enumerate(case["sessions"]))
        return ("Here are our past conversations, each with its date and time (oldest first):\n\n"
                f"{past}\n\nIf a fact about me changed over time, the value from the most recent "
                f"session is the current one; answer with the current value.\n\nNow my question: "
                f"{q['q']}")
    past = "\n\n".join(f"[Session {k + 1}]\nUser: {s}" for k, s in enumerate(case["sessions"]))
    return f"Here are our past conversations:\n\n{past}\n\nNow my question: {q['q']}"


# --- host tools -----------------------------------------------------------------------------
class OverwriteStore:
    def __init__(self, d=None):
        self.d = dict(d or {})

    def copy(self):
        return OverwriteStore(self.d)

    def call(self, fn, a):
        n = (a.get("name") or "").strip()
        if fn == "put":
            self.d[n] = str(a.get("text", "")).strip()
            return "ok"
        if fn == "append":
            self.d[n] = (self.d.get(n, "") + " " + str(a.get("text", "")).strip()).strip()
            return "ok"
        if fn == "get":
            return self.d[n] if n in self.d else "not found"
        if fn == "list":
            return "\n".join(self.d) if self.d else "empty"
        if fn == "delete":
            return "ok" if self.d.pop(n, None) is not None else "not found"
        return f"error: unknown tool {fn}"

    def dump(self):
        return dict(self.d)


class AppendLog:
    def __init__(self, log=None):
        self.log = list(log or [])

    def copy(self):
        return AppendLog(self.log)

    def call(self, fn, a):
        n = (a.get("name") or "").strip()
        if fn == "add_entry":
            self.log.append((n, str(a.get("text", "")).strip()))
            return "ok"
        if fn == "list_entries":
            names = list(dict.fromkeys(x for x, _ in self.log))
            return "\n".join(names) if names else "empty"
        if fn == "get_entries":
            es = [t for x, t in self.log if x == n]
            return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(es)) if es else "not found"
        return f"error: unknown tool {fn} (available: add_entry, list_entries, get_entries)"

    def dump(self):
        return [list(x) for x in self.log]


TC_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FN_RE = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
PAR_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_calls(text):
    calls = []
    for blk in TC_RE.findall(text):
        m = FN_RE.search(blk)
        if m:
            calls.append((m.group(1).strip(), {k: v.strip() for k, v in PAR_RE.findall(m.group(2))}))
            continue
        try:  # json fallback
            j = json.loads(blk.strip())
            calls.append((j["name"], j.get("arguments") or {}))
        except Exception:
            calls.append(("__unparsed__", {"raw": blk[:200]}))
    return calls


# --- model ----------------------------------------------------------------------------------
class M:
    def __init__(self):
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16)
        self.model = self.model.to("cuda").eval()
        self.eos = [self.tok.convert_tokens_to_ids("<|im_end|>"),
                    self.tok.convert_tokens_to_ids("<|endoftext|>")]

    def prompt(self, system, user, tools):
        return self.tok.apply_chat_template([{"role": "system", "content": system},
                                             {"role": "user", "content": user}],
                                            tools=tools, add_generation_prompt=True,
                                            enable_thinking=False, tokenize=False)

    @torch.no_grad()
    def gen(self, text, max_new):
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  max_new_tokens=max_new, do_sample=False, eos_token_id=self.eos,
                                  pad_token_id=self.eos[0], temperature=None, top_p=None,
                                  top_k=None)
        new = out[0, ids.shape[1]:]
        return self.tok.decode(new, skip_special_tokens=False), int(new.shape[0]), ids.shape[1]

    def turn(self, system, user, tools=None, store=None, max_new=200, max_rounds=8):
        """One user turn; tool calls executed by the host store and fed back as tool responses."""
        text = self.prompt(system, user, tools)
        start = len(text)
        calls_log, ntok, segs = [], 0, []
        for _ in range(max_rounds):
            g, n, plen = self.gen(text, max_new)
            ntok += n
            g = g.replace("<|endoftext|>", "")
            if not g.endswith("<|im_end|>"):
                g_clean = g
            else:
                g_clean = g[: -len("<|im_end|>")]
            segs.append(g_clean)
            calls = parse_calls(g_clean) if tools else []
            if not calls:
                text += g
                break
            resps = []
            for fn, a in calls:
                r = store.call(fn, a) if fn != "__unparsed__" else "error: could not parse call"
                calls_log.append({"fn": fn, "args": a, "resp": r})
                resps.append(f"<tool_response>\n{r}\n</tool_response>")
            text += (g_clean.rstrip() + "<|im_end|>\n<|im_start|>user\n" + "\n".join(resps)
                     + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        own = "\n".join(x for x in (own_text(s) for s in segs) if x)
        return text[start:], calls_log, ntok, own


def past_deadline():
    now = dt.datetime.now(BJ)
    return (now.hour, now.minute) >= STOP_AT and now.hour < 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=["C1", "C2", "C3", "C4", "C1s", "C3s"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/experiments/h1/out")
    ap.add_argument("--qmax", type=int, default=200)
    ap.add_argument("--smax", type=int, default=256)
    a = ap.parse_args()
    out = Path(a.out) / f"h1.{a.cond}.seed{a.seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(x)["key"] for x in out.read_text(encoding="utf-8").splitlines() if x}
    free, total = torch.cuda.mem_get_info()
    print(json.dumps({"cuda_free_gb": round(free / 1e9, 2), "total_gb": round(total / 1e9, 2)}),
          flush=True)
    m = M()
    cases = recall_cases(a.n, a.seed)[a.start:]
    tools = {"C2": C2_TOOLS, "C4": C4_TOOLS}.get(a.cond)
    system = {"C2": C2_SYSTEM, "C4": C4_SYSTEM}.get(a.cond, PLAIN_SYSTEM)
    for case in cases:
        keys = [f"{case['id']}/{j}" for j in range(3)]
        if all(k in done for k in keys):
            continue
        if past_deadline():
            print(json.dumps({"stopped": "deadline", "at": case["id"]}), flush=True)
            break
        t0 = time.time()
        sess, store = [], None
        if tools:
            store = OverwriteStore() if a.cond == "C4" else AppendLog()
            for s in case["sessions"]:
                txt, calls, n, _ = m.turn(system, s, tools, store, a.smax)
                sess.append({"text": txt, "calls": calls, "tokens": n})
        for j, q in enumerate(case["questions"]):
            if tools:
                user, st = q["q"], store.copy()
            else:
                user, st = ctx_user(case, q, a.cond.startswith("C1")), None
                if a.cond.endswith("s"):  # brevity variant: same answer style as the tool runs
                    user += "\nAnswer in one short sentence."
            txt, calls, n, ans = m.turn(system, user, tools, st, a.qmax)
            rec = {"key": keys[j], "cond": a.cond, "seed": a.seed, "kind": q["kind"], "q": q["q"],
                   "gold": q["gold"], "old": q["old"], "correct": score_fact(ans, q),
                   "has_new": q["gold"].lower() in ans.lower(), "answer": ans[:600],
                   "text": txt, "calls": calls, "tokens": n, "user": user if j == 0 else None,
                   "store_after_sessions": store.dump() if (tools and j == 0) else None,
                   "sessions": sess if j == 0 else None}
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(json.dumps({k: rec[k] for k in ("key", "kind", "gold", "old", "correct",
                                                     "tokens")} | {"ans": ans[:120]}), flush=True)
        print(json.dumps({"case": case["id"], "sec": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
