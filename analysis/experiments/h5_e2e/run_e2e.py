#!/usr/bin/env python3
"""Stage-8 E3 CPU retrieval prototype.

Pipeline: BGE-small semantic router -> canonical key -> SafeStore exact get.
No Qwen reader or 4B model is loaded. Accuracy is retrieval-level only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MEM = ROOT / "analysis" / "memory_disk"
RISCV = ROOT / "repro" / "qwenputer" / "experimental" / "riscv"
for p in (MEM, RISCV):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import memory_disk as MD
import safe_store as SS

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
PREFIX = "Represent this sentence for searching relevant passages: "
COLLIDING_NAMES = (
    "the locker code for the safe room in",
    "the locker code for the safe in room",
)


def word_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def load_embedder(model_dir: Path):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir))
    model.eval()
    return tokenizer, model, torch, (time.perf_counter() - t0) * 1000.0


def embed_batch(texts, tokenizer, model, torch, max_len=512, batch_size=32):
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state[:, 0]
            normalized = torch.nn.functional.normalize(hidden, p=2, dim=1)
        chunks.append(normalized.cpu().numpy())
    if not chunks:
        return np.empty((0, 384), dtype=np.float32)
    return np.vstack(chunks)


def embed_one(text, tokenizer, model, torch, max_len=128):
    return embed_batch([text], tokenizer, model, torch, max_len=max_len, batch_size=1)[0]


def build_store(items: list[tuple[str, str]]):
    store = SS.SafeStore(seed=2035)
    records = []
    for key, text in items:
        t0 = time.perf_counter()
        result = store.put(key, text)
        records.append({
            "key": key,
            "ok": result.get("answer") == "ok\n",
            "full": bool(result.get("full")),
            "disambiguated": bool(result.get("disambiguated")),
            "ms": (time.perf_counter() - t0) * 1000.0,
        })
    return store, records


def timed_get(store, key: str):
    t0 = time.perf_counter()
    value = store.get(key)
    return value, (time.perf_counter() - t0) * 1000.0


def summarize_puts(records: list[dict]) -> dict:
    return {
        "attempted": len(records),
        "accepted": sum(1 for r in records if r["ok"]),
        "full_rejects": sum(1 for r in records if r["full"]),
        "disambiguated": sum(1 for r in records if r["disambiguated"]),
        "put_ms_p50": percentile([r["ms"] for r in records], 50),
        "put_ms_p95": percentile([r["ms"] for r in records], 95),
    }


def evaluate_group(rows, queries, tokenizer, model, torch, instructed=True):
    key_index = {row["key"]: row for row in rows}
    store, put_records = build_store([(row["key"], row["text"]) for row in rows])
    corpus_t0 = time.perf_counter()
    corpus = embed_batch(
        [row["key"] + " " + row["text"] for row in rows], tokenizer, model, torch
    )
    corpus_ms = (time.perf_counter() - corpus_t0) * 1000.0
    query_ms, get_ms, token_rows = [], [], []
    hits = {1: 0, 3: 0, 5: 0}
    rag = {1: 0, 3: 0, 5: 0}
    e2e = {1: 0, 3: 0, 5: 0}
    examples = []
    for query in queries:
        query_text = (PREFIX if instructed else "") + query["query"]
        t0 = time.perf_counter()
        vector = embed_one(query_text, tokenizer, model, torch)
        query_ms.append((time.perf_counter() - t0) * 1000.0)
        order = np.argsort(-(corpus @ vector))
        gold = query["gold_key"]
        gold_positions = [i for i, row in enumerate(rows) if row["key"] == gold]
        rank = int(np.where(order == gold_positions[0])[0][0]) + 1 if gold_positions else len(rows) + 1
        candidates = []
        for k in (1, 3, 5):
            selected = order[:k]
            selected_keys = [rows[int(i)]["key"] for i in selected]
            hits[k] += int(gold in selected_keys)
            rag[k] += int(any(
                query["gold_answer"] and query["gold_answer"] in rows[int(i)]["text"]
                for i in selected
            ))
            elapsed = 0.0
            found = False
            for selected_key in selected_keys:
                value, elapsed_one = timed_get(store, selected_key)
                elapsed += elapsed_one
                found = found or (value is not None and query["gold_answer"] in value)
            get_ms.append(elapsed)
            e2e[k] += int(found)
            if k == 5:
                candidates = [
                    {"key": rows[int(i)]["key"], "score": float(corpus[int(i)] @ vector)}
                    for i in selected
                ]
        token_rows.append({
            "query": len(word_tokens(query["query"])),
            "router": len(word_tokens(query_text)) + sum(len(word_tokens(c["key"])) for c in candidates),
            "exact": len(word_tokens(query["query"])) + len(word_tokens(query["gold_answer"])),
            "rag_k5": len(word_tokens(query["query"])) + sum(
                len(word_tokens(rows[int(i)]["text"])) for i in order[:5]
            ),
        })
        if len(examples) < 8:
            examples.append({
                "query_hash": hashlib.sha256(query["query"].encode("utf-8")).hexdigest(),
                "gold_key": gold,
                "rank": rank,
                "candidates": candidates,
                "route_hit_at_1": rank == 1,
            })
    n = len(queries)
    return {
        "n": n,
        "router_recall": {f"at_{k}": hits[k] / n for k in (1, 3, 5)},
        "rag_paste_accuracy": {f"at_{k}": rag[k] / n for k in (1, 3, 5)},
        "end_to_end_exact_get_accuracy": {f"at_{k}": e2e[k] / n for k in (1, 3, 5)},
        "tokens": {
            "query_mean": mean([row["query"] for row in token_rows]),
            "router_prompt_and_candidate_display_mean": mean([row["router"] for row in token_rows]),
            "exact_get_query_plus_answer_mean": mean([row["exact"] for row in token_rows]),
            "rag_k5_paste_mean": mean([row["rag_k5"] for row in token_rows]),
        },
        "latency_ms": {
            "bge_query_embed_p50": percentile(query_ms, 50),
            "bge_query_embed_p95": percentile(query_ms, 95),
            "store_get_p50": percentile(get_ms, 50),
            "store_get_p95": percentile(get_ms, 95),
            "corpus_embed_total": corpus_ms,
        },
        "store": summarize_puts(put_records),
        "examples": examples,
    }


def evaluate_longmemeval(items, tokenizer, model, torch):
    aggregate = {
        "router_hits": {1: 0, 3: 0, 5: 0},
        "rag_hits": {1: 0, 3: 0, 5: 0},
        "e2e_hits": {1: 0, 3: 0, 5: 0},
        "query_ms": [], "get_ms": [], "token_rows": [], "put_records": [],
    }
    examples = []
    corpus_ms = 0.0
    for item in items:
        rows = []
        for session_id, session in zip(item["haystack_session_ids"], item["haystack_sessions"]):
            text = " ".join(turn["role"] + ": " + turn["content"] for turn in session)
            rows.append({"key": session_id, "text": text})
        answer = item.get("answer")
        if isinstance(answer, list):
            answer = " ".join(str(part) for part in answer)
        answer = "" if answer is None else str(answer)
        answer_ids = item.get("answer_session_ids", [])
        if not answer_ids:
            continue
        result = evaluate_group(
            rows,
            [{"query": item["question"], "gold_key": answer_ids[0], "gold_answer": answer}],
            tokenizer, model, torch, instructed=True,
        )
        for k in (1, 3, 5):
            aggregate["router_hits"][k] += int(result["router_recall"][f"at_{k}"] > 0)
            aggregate["rag_hits"][k] += int(result["rag_paste_accuracy"][f"at_{k}"] > 0)
            aggregate["e2e_hits"][k] += int(result["end_to_end_exact_get_accuracy"][f"at_{k}"] > 0)
        aggregate["query_ms"].append(result["latency_ms"]["bge_query_embed_p50"])
        aggregate["get_ms"].append(result["latency_ms"]["store_get_p50"])
        aggregate["token_rows"].append(result["tokens"])
        aggregate["put_records"].append(result["store"])
        corpus_ms += result["latency_ms"]["corpus_embed_total"]
        if len(examples) < 8:
            examples.append(result["examples"][0])
    n = len(aggregate["query_ms"])
    def token_mean(field):
        return mean([row[field] for row in aggregate["token_rows"]])
    return {
        "n": n,
        "session_pool": "per_question",
        "router_recall": {f"at_{k}": aggregate["router_hits"][k] / n for k in (1, 3, 5)},
        "rag_paste_accuracy": {f"at_{k}": aggregate["rag_hits"][k] / n for k in (1, 3, 5)},
        "end_to_end_exact_get_accuracy": {f"at_{k}": aggregate["e2e_hits"][k] / n for k in (1, 3, 5)},
        "tokens": {
            "query_mean": token_mean("query_mean"),
            "router_prompt_and_candidate_display_mean": token_mean("router_prompt_and_candidate_display_mean"),
            "exact_get_query_plus_answer_mean": token_mean("exact_get_query_plus_answer_mean"),
            "rag_k5_paste_mean": token_mean("rag_k5_paste_mean"),
        },
        "latency_ms": {
            "bge_query_embed_p50": percentile(aggregate["query_ms"], 50),
            "bge_query_embed_p95": percentile(aggregate["query_ms"], 95),
            "store_get_p50": percentile(aggregate["get_ms"], 50),
            "store_get_p95": percentile(aggregate["get_ms"], 95),
            "corpus_embed_total": corpus_ms,
        },
        "store": {
            "attempted": sum(row["attempted"] for row in aggregate["put_records"]),
            "accepted": sum(row["accepted"] for row in aggregate["put_records"]),
            "full_rejects": sum(row["full_rejects"] for row in aggregate["put_records"]),
            "disambiguated": sum(row["disambiguated"] for row in aggregate["put_records"]),
        },
        "examples": examples,
    }


def collision_test():
    store = SS.SafeStore(seed=2035)
    first = store.put(COLLIDING_NAMES[0], "value A")
    before = store.get(COLLIDING_NAMES[0])
    second = store.put(COLLIDING_NAMES[1], "value B")
    after_first = store.get(COLLIDING_NAMES[0])
    after_second = store.get(COLLIDING_NAMES[1])
    return {
        "names": list(COLLIDING_NAMES),
        "raw_keys_equal": store.raw_key(COLLIDING_NAMES[0]) == store.raw_key(COLLIDING_NAMES[1]),
        "first_disambiguated": bool(first["disambiguated"]),
        "second_disambiguated": bool(second["disambiguated"]),
        "before_second_put": before,
        "after_second_put": {"first": after_first, "second": after_second},
        "logical_names": store.list(),
        "isolated": after_first == "value A\n" and after_second == "value B\n",
    }


def main():
    parser = argparse.ArgumentParser(description="Stage-8 H5 end-to-end CPU retrieval benchmark")
    parser.add_argument("--out", default="analysis/experiments/h5_e2e/results.json")
    parser.add_argument("--n-long", type=int, default=100)
    args = parser.parse_args()

    model_dir = ROOT / ".data" / "bge-small-en-v1.5"
    tokenizer, model, torch, load_ms = load_embedder(model_dir)
    synthetic_path = ROOT / "analysis" / "experiments" / "h5" / "dataset.json"
    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    synthetic_rows = [{"key": row["name"], "text": row["text"]} for row in synthetic["facts"]]
    synthetic_queries = [
        {"query": query["query"], "gold_key": query["target"], "gold_answer": query["answer"]}
        for query in synthetic["queries"]
    ]
    synthetic_result = evaluate_group(
        synthetic_rows, synthetic_queries, tokenizer, model, torch, instructed=True
    )

    long_path = ROOT / ".data" / "longmemeval" / "longmemeval_oracle"
    long_result = None
    if long_path.exists():
        long_items = json.loads(long_path.read_text(encoding="utf-8"))[:args.n_long]
        long_result = evaluate_longmemeval(long_items, tokenizer, model, torch)

    result = {
        "experiment": "stage8-h5-e2e",
        "device": "cpu",
        "gpu_used": False,
        "reader_model_loaded": False,
        "embedding": {
            "model": "BAAI/bge-small-en-v1.5",
            "path": str(model_dir),
            "pooling": "cls",
            "normalization": "l2_unit",
            "instruction_prefix": PREFIX,
            "model_load_ms": load_ms,
        },
        "pipeline": "query -> bge-small cosine top-k -> canonical key -> SafeStore exact get -> containment audit",
        "synthetic": synthetic_result,
        "longmemeval": long_result,
        "collision_test": collision_test(),
        "data_availability": {
            "synthetic_dataset": synthetic_path.exists(),
            "longmemeval_oracle": long_path.exists(),
            "longmemeval_path": str(long_path),
            "reference_store": MD.DEFAULT_MACHINE_ELF.exists(),
        },
        "limitations": [
            "No Qwen3.5-4B reader was loaded; every accuracy figure is retrieval-level containment.",
            "LongMemEval is evaluated per question against that question's own session pool, matching the Stage-5 pool construction.",
            "Long session bodies can exceed memory-disk v0 per-file capacity; store acceptance and full rejects are reported explicitly.",
            "When a question has multiple answer sessions, the first answer_session_id is the canonical gold key.",
        ],
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
