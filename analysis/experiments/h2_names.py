"""Corpus of 2000 names shared by H2 and H3 (copied from h2_alias.py)."""
import random

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


cat = {}
for nm in snake: cat[nm] = "snake"
for nm in short: cat[nm] = "short_sN"
for nm in adv: cat[nm] = "adversarial_prefix"
for nm in bench: cat[nm] = "bench_short"
