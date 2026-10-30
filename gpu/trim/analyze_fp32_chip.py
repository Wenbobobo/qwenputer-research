"""Summarize C8 paired turn logs without rerunning GPU work."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def main():
    result = {}
    for mode in ("fp32", "bf16"):
        path = ROOT / f"gpu/out/c8_{mode}"
        if not (path / "turns.jsonl").exists():
            continue
        turns = [json.loads(x) for x in (path / "turns.jsonl").read_text().splitlines() if x]
        runtime = json.loads((path / "runtime.json").read_text())
        result[mode] = {"runtime": runtime, "turns": [
            {k: r.get(k) for k in ("label", "prompt_tokens", "text_tokens", "chip_tokens", "tok_s", "wall_s", "lock_error", "head_mismatch")}
            for r in turns], "history_divergences": sum(bool(r["lock_error"]) for r in turns if r["label"].startswith("recall-")),
            "history_turns":sum(r["label"].startswith("recall-") for r in turns)}
    if "fp32" in result and "bf16" in result:
        left = {r["label"]: r for r in result["fp32"]["turns"]}
        right = {r["label"]: r for r in result["bf16"]["turns"]}
        result["paired"] = [{"label":k,"fp32_tok_s":left[k]["tok_s"], "bf16_tok_s":right[k]["tok_s"],
            "speed_ratio":left[k]["tok_s"] / right[k]["tok_s"],
            "fp32_lock":left[k]["lock_error"], "bf16_lock":right[k]["lock_error"]} for k in left.keys() & right.keys()]
    (ROOT / "gpu/out/c8_summary.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
