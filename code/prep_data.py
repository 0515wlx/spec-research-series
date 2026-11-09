"""Prepare prompt pool (train) and MATH-500 eval prompts, with dedup scan."""
import json, os, re, hashlib, random, collections
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from datasets import load_dataset

D = "/path/to/spec-research/data"
os.makedirs(D, exist_ok=True)

def norm(t):
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\u4e00-\u9fff ]", "", t)
    return t.strip()

def h(t):
    return hashlib.md5(norm(t).encode()).hexdigest()

def ngrams(t, n=13):
    w = norm(t).split()
    return {" ".join(w[i:i+n]) for i in range(max(0, len(w)-n+1))}

# ---------- MATH-500 ----------
m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
ev = [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"],
       "subject": r["subject"], "level": r["level"]} for r in m500]
with open(f"{D}/math500.jsonl", "w") as f:
    for r in ev:
        f.write(json.dumps(r) + "\n")
print("math500", len(ev))

# ---------- train prompts ----------
pool = []
cm = load_dataset("hendrycks/competition_math", split="train")
for r in cm:
    pool.append({"src": "competition_math", "problem": r["problem"], "solution": r["solution"]})
print("competition_math", len(pool))
n0 = len(pool)
gs = load_dataset("openai/gsm8k", "main", split="train")
for r in gs:
    pool.append({"src": "gsm8k", "problem": r["question"], "solution": r["answer"]})
print("gsm8k", len(pool) - n0)

n0 = len(pool)
try:
    or1 = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    keep = 0
    for r in or1:
        if not r.get("is_reasoning_complete", True):
            continue
        gens = r.get("generations") or []
        sol = gens[0] if gens else (r.get("solution") or "")
        pool.append({"src": "openr1", "problem": r["problem"], "solution": sol})
        keep += 1
        if keep >= 30000:
            break
except Exception as e:
    print("openr1 failed", e)
print("openr1", len(pool) - n0)

# ---------- dedup scan ----------
ev_hash = {h(r["problem"]) for r in ev}
ev_ng = []
for r in ev:
    ev_ng.append(ngrams(r["problem"]))
seen, uniq, dup_exact, dup_eval = set(), [], 0, 0
for r in pool:
    key = h(r["problem"])
    if key in seen:
        dup_exact += 1
        continue
    seen.add(key)
    if key in ev_hash:
        dup_eval += 1
        continue
    g = ngrams(r["problem"])
    if g and any(len(g & e) / max(1, len(g)) > 0.8 for e in ev_ng):
        dup_eval += 1
        continue
    uniq.append(r)
print(f"pool={len(pool)} exact_dup={dup_exact} overlap_with_eval={dup_eval} unique={len(uniq)}")

# per-source unique counts + rough intra-source dup ratio
cnt = collections.Counter(r["src"] for r in uniq)
print("unique by source", cnt)

# ngram-level dup estimate within the deduped pool (sample 5000)
samp = random.Random(0).sample(uniq, min(5000, len(uniq)))
gseen, dup_ng = set(), 0
for r in samp:
    g = ngrams(r["problem"])
    if g and len(g & gseen) / len(g) > 0.9:
        dup_ng += 1
    gseen |= g
print(f"near-dup ratio in sample: {dup_ng}/{len(samp)}")

with open(f"{D}/train_pool.jsonl", "w") as f:
    for r in uniq:
        f.write(json.dumps(r) + "\n")
print("saved", len(uniq))
