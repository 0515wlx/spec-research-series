"""Covariate controls for the raw-OOD residual.

Question: is the raw-OOD rejection residual just a high-entropy / long-tail /
rare-token / punctuation effect?  We use existing token-level caches and the
target sequences to build token frequency, bigram/trigram novelty, rank, entropy,
margin, and format flags, then:

  * stratify q by each covariate and compare raw OOD vs in-domain;
  * standardize raw OOD q to the in-domain covariate distribution;
  * fit match ~ domain + covariates and report the adjusted OOD coefficient
    with a sequence-level bootstrap CI.
"""
from __future__ import annotations

import json
import os
import string
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from infra.model_loader import load_tokenizer

OUT = "out/covariate_controls"
os.makedirs(OUT, exist_ok=True)

SETS = {
    "MATH-500": ("math500", "out/accept", "math500_"),
    "GSM8K": ("gsm8k", "out/accept", "gsm8k_"),
    "HumanEval": ("humaneval", "out/accept", "humaneval_"),
    "WikiText raw": ("wikitext", "out/accept", "wikitext_"),
    "FineWeb raw": ("fineweb", "out/accept", "fineweb_"),
}
DRAFTS = ["qwen3.5-2b", "qwen3.5-27b"]
IN_DOMAIN = ["MATH-500", "GSM8K", "HumanEval"]
OOD = ["WikiText raw", "FineWeb raw"]
COVS = ["ent_t", "margin", "log_rank", "log_freq", "log_bigram", "log_trigram"]


def token_flags(s: str):
    t = s.strip()
    punct = bool(t) and all(ch in string.punctuation or ch in " \n\t\r" for ch in t)
    space = (t == "")
    digit = bool(t) and t.isdigit()
    return punct, space, digit


def build_token_stats(data_path: str, tok, cache: dict):
    rows = [json.loads(l) for l in open(data_path)]
    uni = Counter()
    bi = Counter()
    tri = Counter()
    for r in rows:
        ids = r["output_ids"]
        uni.update(ids)
        bi.update(zip(ids[:-1], ids[1:]))
        tri.update(zip(ids[:-2], ids[1:-1], ids[2:]))

    def s_of(i):
        if i not in cache:
            cache[i] = tok.decode([i])
        return cache[i]

    out = []
    for r in rows:
        ids = r["output_ids"]
        freq = np.empty(len(ids), dtype=np.float32)
        big = np.empty(len(ids), dtype=np.float32)
        trig = np.empty(len(ids), dtype=np.float32)
        punct = np.zeros(len(ids), dtype=np.int8)
        space = np.zeros(len(ids), dtype=np.int8)
        digit = np.zeros(len(ids), dtype=np.int8)
        for j, i in enumerate(ids):
            freq[j] = uni[i]
            big[j] = bi[(ids[j - 1], i)] if j > 0 else uni[i]
            trig[j] = tri[(ids[j - 2], ids[j - 1], i)] if j > 1 else big[j]
            p, sp, d = token_flags(s_of(i))
            punct[j] = int(p); space[j] = int(sp); digit[j] = int(d)
        out.append({"freq": freq, "bigram": big, "trigram": trig,
                    "punct": punct, "space": space, "digit": digit})
    return out


def load_frame(label, draft, tok, cache):
    domain, accept_dir, prefix = SETS[label]
    jp = os.path.join(accept_dir, f"{prefix}{draft}.json")
    npz = os.path.join(accept_dir, f"{prefix}{draft}.npz")
    d = json.load(open(jp))
    stats = build_token_stats(d["data"], tok, cache)
    z = np.load(npz, allow_pickle=True)
    match, ent, margin, rank = z["match"], z["ent_t"], z["margin"], z["rank"]
    sid, tids = z["seq_id"], z["target_ids"]
    frames = []
    for i in range(int(sid.max()) + 1):
        m = sid == i
        n = min(int(m.sum()), len(stats[i]["freq"]))
        if n == 0:
            continue
        frames.append(pd.DataFrame({
            "domain": label, "draft": draft, "seq_id": i,
            "match": match[m][:n].astype(np.int8),
            "ent_t": ent[m][:n], "margin": margin[m][:n],
            "rank": rank[m][:n],
            "freq": stats[i]["freq"][:n], "bigram": stats[i]["bigram"][:n],
            "trigram": stats[i]["trigram"][:n],
            "punct": stats[i]["punct"][:n], "space": stats[i]["space"][:n],
            "digit": stats[i]["digit"][:n],
        }))
    df = pd.concat(frames, ignore_index=True)
    df["log_rank"] = np.log1p(df["rank"])
    df["log_freq"] = np.log1p(df["freq"])
    df["log_bigram"] = np.log1p(df["bigram"])
    df["log_trigram"] = np.log1p(df["trigram"])
    return df


def stratified(df, cov, bins=5):
    x = df[cov].to_numpy()
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    b = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        m = b == k
        if m.sum() == 0:
            continue
        rows.append({"bin": k, "n": int(m.sum()),
                     "q": float(1 - df.loc[m, "match"].mean())})
    return rows


def standardize(df_in, df_ood, cov, bins=5):
    edges = np.quantile(np.concatenate([df_in[cov], df_ood[cov]]),
                        np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    def bidx(x):
        return np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    bi = bidx(df_in[cov].to_numpy())
    bo = bidx(df_ood[cov].to_numpy())
    p_in = np.array([(bi == k).mean() for k in range(len(edges) - 1)])
    q_ood = []
    for k in range(len(edges) - 1):
        m = bo == k
        q_ood.append(float(1 - df_ood.loc[m, "match"].mean()) if m.any() else np.nan)
    q_ood = np.asarray(q_ood)
    ok = ~np.isnan(q_ood)
    return {
        "q_ood_raw": float(1 - df_ood["match"].mean()),
        "q_in_raw": float(1 - df_in["match"].mean()),
        "q_ood_standardized": float(np.sum(p_in[ok] * q_ood[ok]) / p_in[ok].sum()),
        "bins": [{"p_in": float(p_in[k]), "q_ood": (None if np.isnan(q_ood[k]) else float(q_ood[k]))}
                 for k in range(len(edges) - 1)],
    }


def fit_adjusted(df, n_boot=60, sample=120000, seed=0):
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["ood"] = d["domain"].isin(OOD).astype(np.int8)
    feats = COVS + ["punct", "space", "digit"]
    # stratified subsample by domain to keep fit fast
    parts = []
    for dom, g in d.groupby("domain"):
        if len(g) > sample // d["domain"].nunique():
            g = g.sample(sample // d["domain"].nunique(), random_state=seed)
        parts.append(g)
    d = pd.concat(parts, ignore_index=True)
    X = d[feats].to_numpy(dtype=np.float64)
    y = d["match"].to_numpy(dtype=np.int8)
    X = StandardScaler().fit_transform(X)
    X2 = np.column_stack([d["ood"].to_numpy(), X])
    base = LogisticRegression(max_iter=1000, C=1.0).fit(X2, y)
    coef = float(base.coef_[0][0])
    # sequence bootstrap, using index groups for speed
    vals = []
    groups = {s: g.index.to_numpy() for s, g in d.groupby("seq_id")}
    seqs = np.asarray(list(groups))
    for _ in range(n_boot):
        take = rng.choice(seqs, size=len(seqs), replace=True)
        idx = np.concatenate([groups[s] for s in take])
        dd = d.loc[idx]
        Xb = np.column_stack([dd["ood"].to_numpy(),
                              StandardScaler().fit_transform(dd[feats].to_numpy())])
        yb = dd["match"].to_numpy()
        try:
            f = LogisticRegression(max_iter=500, C=1.0).fit(Xb, yb)
            vals.append(float(f.coef_[0][0]))
        except Exception:
            pass
    vals = np.asarray(vals)
    return {"coef_ood": coef,
            "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
            "odds_ratio": float(np.exp(coef)),
            "n_boot": int(len(vals)), "n_tokens": int(len(d))}


def main():
    tok = load_tokenizer("qwen3.5-35b-a3b")
    cache = {}
    frames = []
    for label in SETS:
        for draft in DRAFTS:
            print("load", label, draft, flush=True)
            frames.append(load_frame(label, draft, tok, cache))
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_parquet(os.path.join(OUT, "tokens.parquet"))

    out = {"stratified": {}, "standardized": {}, "adjusted": {}}
    for draft in DRAFTS:
        d = all_df[all_df["draft"] == draft]
        for cov in COVS + ["punct", "space", "digit"]:
            out["stratified"][f"{draft}/{cov}"] = {
                dom: stratified(d[d["domain"] == dom], cov)
                for dom in IN_DOMAIN + OOD
            }
        for ood in OOD:
            for ind in IN_DOMAIN:
                di = d[d["domain"] == ind]
                do = d[d["domain"] == ood]
                out["standardized"][f"{draft}/{ood}/{ind}"] = {
                    cov: standardize(di, do, cov) for cov in COVS
                }
        out["adjusted"][draft] = fit_adjusted(d)

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)

    lines = ["# Covariate controls for raw-OOD rejection", ""]
    lines += ["## Adjusted OOD coefficient (match ~ OOD + covariates)", "",
              "| draft | OOD coef | 95% CI | odds ratio | n tokens |",
              "|---|---:|---|---:|---:|"]
    for draft, v in out["adjusted"].items():
        lines.append(f"| {draft} | {v['coef_ood']:+.3f} | "
                     f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | "
                     f"{v['odds_ratio']:.3f} | {v['n_tokens']} |")
    lines += ["", "## Standardized OOD q (matched to in-domain covariate distribution)", "",
              "| draft | OOD | reference | covariate | q OOD raw | q OOD std | q in-domain |",
              "|---|---|---|---|---:|---:|---:|"]
    for key, covs in out["standardized"].items():
        draft, ood, ind = key.split("/")
        for cov, v in covs.items():
            lines.append(f"| {draft} | {ood} | {ind} | {cov} | {v['q_ood_raw']:.4f} | "
                         f"{v['q_ood_standardized']:.4f} | {v['q_in_raw']:.4f} |")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved", OUT)


if __name__ == "__main__":
    main()
