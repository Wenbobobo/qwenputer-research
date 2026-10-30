"""H3: token-cost crossing point of list-based recall (store) vs RAG (k=5, c=100).
Uses the reference store protocol (common.py) and the Qwen3.5 tokenizer only.
"""
import random

from common import (Config, call_tokens, get_response_tokens, list_response_tokens,
                    name_key_from_tokens, encode, load_tokenizer, write_section, Q_TCO, Q_TRO)

tok = load_tokenizer()
cfg = Config()

CONTENT = ("The user lives in Shanghai, works as a data engineer on weekdays, prefers green tea "
           "over coffee, speaks Mandarin at home and English at work, has a dog named Mochi who is "
           "nine years old, and plans a trip to Kyoto in the spring of next year before the "
           "conference in Berlin in the autumn, so the schedule should be checked every month.")
content_all = encode(tok, CONTENT)
assert len(content_all) >= 40, len(content_all)
LENS = [10, 20, 40]

# RAG baseline (token count only): query q + k*c; q assumed 20 (as in docs/research/parametric-memory.md),
# and the measured length of a typical question is reported too.
K, C_RAG, Q_ASSUMED = 5, 100, 20
RAG = Q_ASSUMED + K * C_RAG
QUESTION = "What city does the user live in, and what is their dog's name?"
q_measured = len(encode(tok, QUESTION))

# realistic names: the snake/short/bench corpus of H2 (no adversarial prefix names)
import h2_names  # noqa: E402  (corpus definition shared with h2_alias.py)
pool = [nm for nm in h2_names.corpus if h2_names.cat[nm] != "adversarial_prefix"]
ntoks = {nm: encode(tok, nm) for nm in pool}
keys = {nm: name_key_from_tokens(ntoks[nm]) for nm in pool}
assert len(set(keys.values())) == len(pool), "realistic pool must have distinct keys"

def printed_name(nm):
    """What list prints for a name: the key's first min(length, NMAX) tokens."""
    n, toks = keys[nm]
    return list(toks[:min(n, 6)])

# --- get(name) ------------------------------------------------------------------------------
NAME = "user_city"
name_toks = ntoks[NAME]
get_call = call_tokens("get", name_toks)
get_rows = []
for L in LENS:
    resp = get_response_tokens(content_all[:L])
    get_rows.append({
        "content_tokens": L,
        "qwen_call_tokens": len(get_call),
        "tool_response_wrapper_tokens": len(resp) - L,
        "tool_response_total_tokens": len(resp),
        "get_total_context_tokens": len(get_call) + len(resp),
        "chip_tokens_expected_hit_LF0.1": 2.05,
        "chip_tokens_note": "1 SP probe + 1 SG per chunk (best case 2; H2 realistic hit probes give 2.05-2.49)",
    })
call_breakdown = {
    "tool_call_tokens_total": len(get_call),
    "wrapper_and_op_tokens": len(get_call) - len(name_toks),
    "name_tokens": len(name_toks),
}

# --- list() with n names -----------------------------------------------------------------------
list_call = call_tokens("list")
rng = random.Random(11)
NS = [1, 5, 10, 20, 50, 100, 200, 400]
rows = []
for n in NS:
    costs_list, costs_listget = {L: [] for L in LENS}, {L: [] for L in LENS}
    printed_totals = []
    for trial in range(200):
        sel = rng.sample(pool, n)
        names_out = [printed_name(nm) for nm in sel]
        resp = list_response_tokens(names_out)
        printed_totals.append(len(resp))
        target = sel[0]
        for L in LENS:
            tot_list = len(list_call) + len(resp)
            costs_list[L].append(tot_list)
            get_c = call_tokens("get", ntoks[target])
            tot_get = len(get_c) + len(get_response_tokens(content_all[:L]))
            costs_listget[L].append(tot_list + tot_get)
    mean = lambda xs: round(sum(xs) / len(xs), 2)
    rows.append({
        "n_names": n,
        "list_call_tokens": len(list_call),
        "list_tool_response_tokens_mean": mean(printed_totals),
        "list_call_plus_response_tokens_mean": mean(costs_list[20]),
        "list_plus_get_L10_mean": mean(costs_listget[10]),
        "list_plus_get_L20_mean": mean(costs_listget[20]),
        "list_plus_get_L40_mean": mean(costs_listget[40]),
        "rag_context_tokens": RAG,
        "list_only_minus_rag": round(mean(costs_list[20]) - RAG, 2),
        "list_plus_get_L20_minus_rag": round(mean(costs_listget[20]) - RAG, 2),
    })

# per-name slope (printed tokens per name in list output, excluding the SL chips)
per_name = [len(printed_name(nm)) for nm in pool]
slope = sum(per_name) / len(per_name)

# crossing points: the first n where the list-based recall costs more than RAG
def first_cross(kind, L):
    for n in range(1, 1001):
        resp_mean = []
        r2 = random.Random(100 + n)
        for trial in range(60):
            sel = r2.sample(pool, n)
            resp = list_response_tokens([printed_name(nm) for nm in sel])
            tot = len(list_call) + len(resp)
            if kind == "list_plus_get":
                tot += len(call_tokens("get", ntoks[sel[0]])) + len(get_response_tokens(content_all[:L]))
            resp_mean.append(tot)
        if sum(resp_mean) / len(resp_mean) >= RAG:
            return n
    return None

crossings = {}
for L in LENS:
    crossings["list_plus_get_L%d" % L] = first_cross("list_plus_get", L)
crossings["list_only"] = first_cross("list_only", 0)
# analytic: list_only(n) ~ A + slope*n, RAG constant
A_list = len(list_call) + 3 + 8  # header and tail
crossings["analytic_list_only_approx"] = round((RAG - A_list) / slope, 1)
crossings["analytic_list_plus_get_L20_approx"] = round(
    (RAG - (A_list + len(get_call) + 3 + 8 + 20)) / slope, 1)

out = {
    "description": "H3 token-cost crossing: list-based recall vs RAG (k=5, c=100)",
    "protocol_source": "repro/qwenputer/experimental/riscv/chipformer/rv/store.py (call_tokens, respond list/get)",
    "tokenizer": "Qwen3.5 tokenizer.json (tokens counted as the store writes them)",
    "name_example": NAME,
    "name_example_tokens": len(name_toks),
    "get_call_breakdown": call_breakdown,
    "get_costs": get_rows,
    "list_call_tokens": len(list_call),
    "list_wrapper_tokens_no_names": len(list_response_tokens([])),
    "list_rows": rows,
    "realistic_name_pool_size": len(pool),
    "printed_tokens_per_name_mean": round(slope, 3),
    "parametric_memory_doc_assumption_tokens_per_name": 4,
    "rag": {"k": K, "c": C_RAG, "query_tokens_assumed": Q_ASSUMED, "rag_context_tokens": RAG,
            "measured_example_question_tokens": q_measured,
            "measured_example_question": QUESTION},
    "crossing_n_star": crossings,
    "doc_prediction_range": [100, 150],
}
write_section("H3", out)
import json
print(json.dumps(out, ensure_ascii=False, indent=1))
