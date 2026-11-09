"""Build train prompt pool from NuminaMath-CoT (modelscope) with dedup scan."""
import json, os, re, hashlib, random, collections
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

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
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


ev = [json.loads(l) for l in open(f"{D}/math500.jsonl")]
ev_hash = {h(r["problem"]) for r in ev}
ev_ng = [ngrams(r["problem"]) for r in ev]

from modelscope.msdatasets import MsDataset
ds = MsDataset.load("AI-MO/NuminaMath-CoT", split="train", use_streaming=True)

pool = []
for i, r in enumerate(ds):
    p = r.get("problem") or ""
    if len(p) < 20:
        continue
    pool.append({"src": r.get("source", "numina"), "problem": p, "solution": r.get("solution", "")})
    if len(pool) >= 40000:
        break
print("raw pool", len(pool), "src dist", collections.Counter(x["src"] for x in pool).most_common(8))

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
print(f"pool={len(pool)} exact_dup={dup_exact} ({dup_exact/len(pool):.1%}) overlap_with_eval={dup_eval} unique={len(uniq)}")

# near-dup estimate on the deduped pool
samp = random.Random(0).sample(uniq, min(5000, len(uniq)))
gseen, dup_ng = set(), 0
for r in samp:
    g = ngrams(r["problem"])
    if g and len(g & gseen) / len(g) > 0.9:
        dup_ng += 1
    gseen |= g
print(f"near-dup ratio in sample: {dup_ng}/{len(samp)}")

# coverage of eval-like topics: avg length
ls = [len(r["problem"]) for r in uniq]
print("problem chars mean/median", sum(ls) / len(ls), sorted(ls)[len(ls) // 2])

random.Random(1).shuffle(uniq)
with open(f"{D}/train_pool.jsonl", "w") as f:
    for r in uniq:
        f.write(json.dumps(r) + "\n")
print("saved", len(uniq))
