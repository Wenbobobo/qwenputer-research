"""Dump verbatim transcripts of one case (default recall-0, seed 0) per condition -> h1_examples.md"""
import json
import sys
from pathlib import Path

D = Path("analysis/experiments/h1/out")
CASE = sys.argv[1] if len(sys.argv) > 1 else "recall-0"
out = [f"# H1 verbatim transcripts, case {CASE} (seed 0)\n"]
for c in ("C1", "C3", "C1s", "C3s", "C4", "C2"):
    f = D / f"h1.{c}.seed0.jsonl"
    if not f.exists():
        continue
    rs = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x]
    rs = [r for r in rs if r["key"].startswith(CASE + "/")]
    out.append(f"\n## {c}\n")
    if rs and rs[0].get("sessions"):
        for k, s in enumerate(rs[0]["sessions"]):
            out.append(f"\n### past session {k + 1}: model output (tool calls executed by host)\n")
            out.append("```text\n" + s["text"] + "\n```\n")
            out.append("host calls: `" + json.dumps(s["calls"], ensure_ascii=False) + "`\n")
        out.append("\nstore after 4 sessions: `" + json.dumps(rs[0]["store_after_sessions"], ensure_ascii=False) + "`\n")
    elif rs:
        out.append("\n### user prompt (question 1; questions 2-3 differ only in the last line)\n")
        out.append("```text\n" + rs[0]["user"] + "\n```\n")
    for r in rs:
        out.append(f"\n### Q{r['key'][-1]} [{r['kind']}] {r['q']}  gold={r['gold']} old={r['old']} correct={r['correct']}\n")
        out.append("```text\n" + r["text"] + "\n```\n")
Path("analysis/experiments/h1/h1_examples.md").write_text("".join(out), encoding="utf-8")
print("ok", len("".join(out)))
