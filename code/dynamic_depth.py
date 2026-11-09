"""Dynamic depth from confidence/entropy clustering.

Question: if depth gamma is chosen per-token from the draft's confidence (a proxy
for entropy), does the larger teacher-QAT draft benefit more than MoE8?  And how
does a large draft compare with native MTP at long gamma?

Policies:
  fixed          : one gamma for all tokens
  cluster        : confidence quantile bins -> one gamma per bin (learned on train)
  sequential     : draft while confidence >= threshold (up to gamma_max)
  oracle-cluster : same as cluster but gamma per bin fitted on the TEST labels

5-fold CV by sequence.  Metric = speedup = accepted tokens / (1 + gamma*c).
"""
import argparse, os
import numpy as np

D = "/path/to/spec-research/out"
GRID = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]


def load_data(name):
    if name in ("dense", "mtp"):
        d = np.load(os.path.join(D, "composite2.npz"))
        tgt, seq = d["tgt"], d["seq"]
        pred = d["d_pred"] if name == "dense" else d["m_pred"]
        conf = d["d_conf"] if name == "dense" else d["m_conf"]
        match = pred == tgt
    else:
        fn = os.path.join(D, f"features_{name}.npz")
        if not os.path.exists(fn):
            return None
        d = np.load(fn)
        match, conf, seq = d["match"].astype(bool), d["maxp"], d["seq"]
    ms, cs = [], []
    for s in range(int(seq.max()) + 1):
        idx = np.where(seq == s)[0]
        ms.append(match[idx]); cs.append(conf[idx])
    return ms, cs


def run_policy(ms, cs, policy, c, gamma_max=32):
    tot_acc = tot_cost = blocks = prop = accd = gsum = 0
    mode = policy[0]
    for m, cf in zip(ms, cs):
        t, N = 0, len(m)
        while t < N:
            if mode == "fixed":
                g = policy[1]
            elif mode == "cluster":
                g = policy[1][np.searchsorted(policy[2], cf[t], side="right")]
            elif mode == "seq":
                g = 0
                for i in range(min(gamma_max, N - t)):
                    if cf[t + i] < policy[1]:
                        break
                    g += 1
            else:
                raise ValueError(mode)
            g = min(g, N - t)
            L = 0
            while L < g and m[t + L]:
                L += 1
            tot_acc += L + 1
            tot_cost += 1 + g * c
            prop += g; accd += L; gsum += g; blocks += 1
            t += L + 1
    return {"speedup": tot_acc / max(tot_cost, 1),
            "tau": tot_acc / max(blocks, 1),
            "alpha": accd / max(prop, 1),
            "gamma": gsum / max(blocks, 1),
            "acc": tot_acc, "cost": tot_cost, "blocks": blocks,
            "prop": prop, "accd": accd, "gsum": gsum}


def fit_fixed(ms, cs, c):
    return max(GRID, key=lambda g: run_policy(ms, cs, ("fixed", g), c)["speedup"])


def fit_cluster(ms, cs, c, K=4):
    tr_conf = np.concatenate(cs)
    edges = np.quantile(tr_conf, np.linspace(0, 1, K + 1)[1:-1])
    gs = [fit_fixed(ms, cs, c)] * K
    for _ in range(2):
        for j in range(K):
            best = (gs[j], -1.0)
            for g in GRID:
                trial = list(gs); trial[j] = g
                s = run_policy(ms, cs, ("cluster", trial, edges), c)["speedup"]
                if s > best[1]:
                    best = (g, s)
            gs[j] = best[0]
    return gs, edges


def fit_seq(ms, cs, c):
    tr_conf = np.concatenate(cs)
    ths = np.quantile(tr_conf, [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
    best = (None, None, -1.0)
    for th in ths:
        for gm in (8, 16, 32, 48):
            s = run_policy(ms, cs, ("seq", th), c, gamma_max=gm)["speedup"]
            if s > best[2]:
                best = (th, gm, s)
    return best[0], best[1]


def cv(ms, cs, c, K=4):
    n = len(ms)
    folds = [list(range(i, n, 5)) for i in range(5)]
    keys = ("fixed", "cluster", "seq", "oracle")
    agg = {k: {"acc": 0, "cost": 0, "blocks": 0, "prop": 0, "accd": 0, "gsum": 0} for k in keys}
    gammas = {"fixed": [], "cluster": [], "seq": []}
    for f in range(5):
        te = folds[f]
        tr = [i for i in range(n) if i not in te]
        ms_tr = [ms[i] for i in tr]; cs_tr = [cs[i] for i in tr]
        ms_te = [ms[i] for i in te]; cs_te = [cs[i] for i in te]
        gfix = fit_fixed(ms_tr, cs_tr, c)
        gs, edges = fit_cluster(ms_tr, cs_tr, c, K)
        th, gm = fit_seq(ms_tr, cs_tr, c)
        gs_o, edges_o = fit_cluster(ms_te, cs_te, c, K)
        for key, pol, gmax in [("fixed", ("fixed", gfix), 32),
                               ("cluster", ("cluster", gs, edges), 32),
                               ("seq", ("seq", th), gm),
                               ("oracle", ("cluster", gs_o, edges_o), 32)]:
            s = run_policy(ms_te, cs_te, pol, c, gamma_max=gmax)
            for k in ("acc", "cost", "blocks", "prop", "accd", "gsum"):
                agg[key][k] += s[k]
        gammas["fixed"].append(gfix)
        gammas["cluster"].append(list(map(int, gs)))
        gammas["seq"].append((round(float(th), 4), gm))
    out = {}
    for k, r in agg.items():
        out[k] = {"speedup": r["acc"] / max(r["cost"], 1),
                  "tau": r["acc"] / max(r["blocks"], 1),
                  "alpha": r["accd"] / max(r["prop"], 1),
                  "gamma": r["gsum"] / max(r["blocks"], 1)}
    out["gammas"] = gammas
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    a = ap.parse_args()
    models = [("dense", 0.099), ("moe8", 0.099), ("mtp", 0.03), ("teacher_qat", 0.268)]
    for name, c in models:
        data = load_data(name)
        if data is None:
            print(f"\n== {name}: features not found, skip")
            continue
        ms, cs = data
        print(f"\n== {name}  c={c}  nseq={len(ms)}  match={np.mean([m.mean() for m in ms]):.4f}")
        print("  fixed gamma curve (gamma speedup tau):")
        for g in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]:
            s = run_policy(ms, cs, ("fixed", g), c)
            print(f"    {g:2d}  {s['speedup']:.3f}  {s['tau']:.2f}")
        r = cv(ms, cs, c, a.K)
        print("  CV results:")
        for k in ["fixed", "cluster", "seq", "oracle"]:
            print(f"    {k:8s} speedup={r[k]['speedup']:.3f} tau={r[k]['tau']:.2f} "
                  f"avg_gamma={r[k]['gamma']:.1f}")
        gain = r["cluster"]["speedup"] / r["fixed"]["speedup"] - 1
        gains = r["seq"]["speedup"] / r["fixed"]["speedup"] - 1
        print(f"    cluster gain vs fixed: {gain*100:+.1f}%   sequential gain: {gains*100:+.1f}%")
        print(f"    fold fixed gammas: {r['gammas']['fixed']}")
        print(f"    fold cluster gammas: {r['gammas']['cluster']}")
        print(f"    fold seq (thr,gmax): {r['gammas']['seq']}")


if __name__ == "__main__":
    main()
