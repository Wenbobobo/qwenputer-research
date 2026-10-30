"""Load the upstream synthetic multi-session recall generator (storebench.recall_cases) without
importing chipformer (which needs the RV machine). We parse storebench.py with ast and exec only the
data constants, recall_cases and score_fact, so the cases are byte-identical to upstream (seed 0).
"""
import ast
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SB = ROOT / "repro/qwenputer/experimental/riscv/chipformer/rv/storebench.py"
KEEP = {"ATTRS", "OPENERS", "FILLERS", "CLOSERS", "recall_cases", "score_fact", "own_text"}


def _load():
    tree = ast.parse(SB.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) in KEEP for t in node.targets):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in KEEP:
            body.append(node)
    mod = ast.Module(body=body, type_ignores=[])
    ns = {"random": random, "re": re}
    exec(compile(mod, str(SB), "exec"), ns)
    return ns


NS = _load()
recall_cases = NS["recall_cases"]
score_fact = NS["score_fact"]
own_text = NS["own_text"]
ATTRS = NS["ATTRS"]

if __name__ == "__main__":
    import json
    cs = recall_cases(30, 0)
    print(json.dumps(cs[0], indent=1))
    print(sum(q["kind"] == "update" for c in cs for q in c["questions"]), "update questions")
