"""H2: name-key / alias statistics for the reference store (see common.py for the cited protocol).

Run: cd repo root; <venv>/Scripts/python.exe -I analysis/experiments/h2_alias.py
"""
import random
from collections import defaultdict

from common import (Config, Directory, NMAX, name_key_from_tokens, encode, load_tokenizer,
                    write_section, Q_NL)

tok = load_tokenizer()
cfg = Config()
N_SLOTS = cfg.n_slots

# ---------------- corpus (2000 names) ----------------
rng = random.Random(1)
A = ["user", "dog", "cat", "home", "work", "car", "pet", "team", "project", "favorite", "my",
     "wife", "kid", "house", "phone", "bank", "trip", "birth", "school", "coffee"]
B = ["city", "name", "job", "age", "color", "food", "email", "lang", "timezone", "address",
     "birthday", "hobby", "salary", "role", "company", "country", "plan", "budget", "pref",
     "owner", "size", "model", "brand", "status", "note", "goal", "id", "port", "key", "phone"]
C = ["first", "last", "main", "backup", "old", "new", "alt", "default", "current", "pref", "v2",
     "2", "3"]
base_names = ["city", "job", "name", "age", "email", "phone", "dog", "cat", "color", "food",
              "hobby", "pet", "lang", "timezone", "birthday", "address", "car", "team", "role",
              "company", "school", "country", "salary", "plan", "budget", "owner", "size", "model",
              "brand", "status", "note", "goal", "id", "port", "key", "wife", "kid", "house",
              "work", "bank", "trip", "coffee", "gender", "height", "weight", "zodiac",
              "religion", "income", "passport", "license", "diet", "allergy", "hometown",
              "nickname", "twitter", "shoe", "eye", "hair", "music", "sport", "drink", "movie", "book"]
forms = ["{x}", "my_{x}", "user_{x}", "{x}_2", "the_{x}", "{x}_pref", "{x}_value", "{x}_now",
         "current_{x}", "new_{x}"]
bench = []
for x in base_names:
    for fm in forms:
        bench.append(fm.format(x=x))
bench_set = set(bench)
snake, seen = [], set(bench_set)
while len(snake) < 2000 - 200 - 400 - len(bench):
    k = rng.choice([2, 2, 3])
    nm = "_".join([rng.choice(A)] + [rng.choice(B + C) for _ in range(k - 1)])
    if nm not in seen:
        seen.add(nm)
        snake.append(nm)
short = ["s%d" % i for i in range(1, 201)]
adv = []
words = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta", "zeta", "eta",
         "lambda", "mu", "nu", "xi", "pi", "rho", "tau", "phi", "chi", "psi", "north", "south",
         "east", "west", "red", "blue", "green", "black", "white", "gold", "silver", "iron",
         "copper", "lead", "tin", "zinc", "nickel", "cobalt", "ruby", "jade"]
for g, w in enumerate(words):  # 40 groups; members share a long prefix (> 6 tokens)
    prefix = "%s_%s_quarterly_review_meeting_notes_for_" % (w, words[(g + 7) % 40])
    for j in range(10):
        adv.append(prefix + "item%d" % (j + 1))
corpus = snake + short + adv + bench
assert len(corpus) == 2000 and len(set(corpus)) == 2000, (len(corpus), len(set(corpus)))
cat = {}
for nm in snake: cat[nm] = "snake"
for nm in short: cat[nm] = "short_sN"
for nm in adv: cat[nm] = "adversarial_prefix"
for nm in bench: cat[nm] = "bench_short"

toks = {nm: encode(tok, nm) for nm in corpus}
keys = {nm: name_key_from_tokens(toks[nm]) for nm in corpus}
slots = {nm: cfg.slot_of(keys[nm]) for nm in corpus}

# ---------------- corpus-level alias structure ----------------
groups = defaultdict(list)
for nm in corpus:
    groups[keys[nm]].append(nm)
aliased = {nm for g in groups.values() if len(g) > 1 for nm in g}
alias_by_cat = {}
for c in ["snake", "short_sN", "adversarial_prefix", "bench_short"]:
    members = [nm for nm in corpus if cat[nm] == c]
    alias_by_cat[c] = {"names": len(members),
                       "aliased_names": sum(1 for nm in members if nm in aliased),
                       "alias_rate": round(sum(1 for nm in members if nm in aliased) / len(members), 4)}
corpus_stats = {
    "names": len(corpus),
    "distinct_keys": len(groups),
    "aliased_names": len(aliased),
    "alias_rate_corpus": round(len(aliased) / len(corpus), 4),
    "largest_key_group": max(len(g) for g in groups.values()),
    "by_category": alias_by_cat,
    "name_token_length_hist": {str(k): sum(1 for nm in corpus if len(toks[nm]) == k)
                               for k in sorted(set(len(toks[nm]) for nm in corpus))},
    "names_over_5_tokens_truncated_in_list": sum(1 for nm in corpus if len(toks[nm]) + 1 > NMAX),
    "slot_distinct_used_by_all_keys": len(set(slots[nm] for nm in corpus)),
}

# ---------------- load-factor experiments ----------------
TRIALS = 50
MISS_SAMPLE = 300
res_lf = {}
POOLS = [("all_2000", corpus), ("realistic_no_adversarial", [nm for nm in corpus if cat[nm] != "adversarial_prefix"])]
for pool_name, pool in POOLS:
    res_lf[pool_name] = {}
    for lf in [0.1, 0.25, 0.5]:
        N = round(lf * N_SLOTS)
        r = random.Random(int(lf * 1000))
        hit_probes, miss_probes, ins_probes = [], [], []
        alias_frac, wrong_get_frac, overwrites, false_hits, miss_n = [], [], [], [], 0
        for t in range(TRIALS):
            sub = r.sample(pool, N)
            rest = [nm for nm in pool if nm not in set(sub)]
            d = Directory(cfg)
            ow = 0
            for nm in sub:
                p, st = d.put(keys[nm])
                ins_probes.append(p)
                if st == "replaced":
                    ow += 1
            # hit probes (each name looked up once, as get would)
            for nm in sub:
                _, p, found = d.probe(keys[nm])
                hit_probes.append(p)
            # alias: names in sub whose key is shared by another sub name
            sub_groups = defaultdict(list)
            for nm in sub:
                sub_groups[keys[nm]].append(nm)
            n_alias = sum(len(g) for g in sub_groups.values() if len(g) > 1)
            alias_frac.append(n_alias / N)
            wrong_get_frac.append((N - len(sub_groups)) / N)  # gets that would return another name's text
            overwrites.append(ow)
            # misses: names outside sub
            sample = r.sample(rest, MISS_SAMPLE)
            for nm in sample:
                _, p, found = d.probe(keys[nm])
                miss_probes.append(p)
                if found:
                    false_hits_add = 1
                    false_hits.append(1)
                miss_n += 1
        alpha = N / N_SLOTS
        res_lf[pool_name][str(lf)] = {
            "names_inserted": N,
            "alpha": alpha,
            "trials": TRIALS,
            "alias_rate_within_subset": round(sum(alias_frac) / TRIALS, 5),
            "wrong_get_rate_within_subset": round(sum(wrong_get_frac) / TRIALS, 5),
            "overwrites_per_trial_mean": round(sum(overwrites) / TRIALS, 4),
            "insert_probes_mean": round(sum(ins_probes) / len(ins_probes), 4),
            "hit_probes_mean": round(sum(hit_probes) / len(hit_probes), 4),
            "hit_probes_max": max(hit_probes),
            "miss_probes_mean": round(sum(miss_probes) / len(miss_probes), 4),
            "miss_probes_max": max(miss_probes),
            "false_hit_rate_on_miss": round(len(false_hits) / miss_n, 5),
            "get_chip_tokens_hit_mean": round(sum(hit_probes) / len(hit_probes) + 1, 4),
            "get_chip_tokens_miss_mean": round(sum(miss_probes) / len(miss_probes), 4),
            "analytic_hit_probes_linear_probing": round((1 + 1 / (1 - alpha)) / 2, 4),
            "analytic_miss_probes_linear_probing": round((1 + 1 / (1 - alpha) ** 2) / 2, 4),
        }

# ---------------- exhaustion: insert the whole corpus in random order ----------------
r = random.Random(7)
ex = []
for t in range(TRIALS):
    order = corpus[:]
    r.shuffle(order)
    d = Directory(cfg)
    created = replaced = full = 0
    for nm in order:
        _, st = d.put(keys[nm])
        if st == "created": created += 1
        elif st == "replaced": replaced += 1
        elif st == "full": full += 1
    ex.append((created, replaced, full))
exhaust = {
    "trials": TRIALS,
    "created_mean": round(sum(e[0] for e in ex) / TRIALS, 2),
    "overwritten_by_alias_mean": round(sum(e[1] for e in ex) / TRIALS, 2),
    "rejected_full_mean": round(sum(e[2] for e in ex) / TRIALS, 2),
    "note": "512-name cap (n_names) is reached after 512 distinct-key creates; alias puts overwrite instead of creating",
}

# ---------------- are benchmark-style short names safe? ----------------
bench_safe = {"bench_names": len(bench)}
bn_alias = [nm for nm in bench if nm in aliased]
bn_trunc = [nm for nm in bench if len(toks[nm]) + 1 > NMAX]
bench_safe["aliased_with_corpus"] = len(bn_alias)
bench_safe["aliased_examples"] = bn_alias[:10]
bench_safe["over_5_tokens"] = len(bn_trunc)
bench_safe["collide_with_snake_or_short"] = sum(
    1 for nm in bench if any(keys[o] == keys[nm] and o != nm for o in corpus if cat[o] != "bench_short"))
bench_safe["short_names_tokens_mean"] = round(sum(len(toks[nm]) for nm in bench) / len(bench), 3)
bench_safe["slot_collisions_bench_vs_corpus_mean_probe"] = round(
    sum(1 for nm in bench if sum(1 for o in corpus if slots[o] == slots[nm]) > 1) / len(bench), 4)

out = {
    "description": "H2 name-key/alias statistics (reference store.py protocol: 6 tokens + length, linear probing, 1024 slots, 512 names)",
    "corpus": corpus_stats,
    "load_factors": res_lf,
    "exhaustion_full_corpus_insert": exhaust,
    "benchmark_short_names": bench_safe,
}
write_section("H2", out)
import json
print(json.dumps(out, ensure_ascii=False, indent=1))
