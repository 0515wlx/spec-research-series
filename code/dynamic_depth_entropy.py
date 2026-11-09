"""2D entropy clustering: [draft maxp, previous-token teacher entropy] -> gamma."""
import sys, os
import numpy as np
from scipy.cluster.vq import kmeans2
sys.path.insert(0, os.path.dirname(__file__))
from dynamic_depth import load_data, run_policy, GRID

D = "/path/to/spec-research/out"


def get_tent(seq):
    d = np.load(os.path.join(D, "composite2.npz"))
    t = d["t_ent"]; s = d["seq"]
    out = np.zeros_like(t, dtype=np.float32)
    for si in range(int(s.max()) + 1):
        idx = np.where(s == si)[0]
        e = t[idx]
        out[idx[0]] = e[0]
        out[idx[1:]] = e[:-1]
    return out


def fit_ids(ms, cf, ids, c, K, mode="ids"):
    class P:
        pass
    gs = [8] * K
    # coordinate descent on training sequences
    def sim(g):
        tot_acc = tot_cost = 0
        for m, cf_, id_ in zip(ms, cf, ids):
            t, N = 0, len(m)
            while t < N:
                gg = min(g[id_[t]], N - t)
                L = 0
                while L < gg and m[t + L]:
                    L += 1
                tot_acc += L + 1; tot_cost += 1 + gg * c; t += L + 1
        return tot_acc / max(tot_cost, 1)
    for _ in range(2):
        for j in range(K):
            best = (gs[j], -1.0)
            for gg in GRID:
                trial = list(gs); trial[j] = gg
                s = sim(trial)
                if s > best[1]:
                    best = (gg, s)
            gs[j] = best[0]
    return gs


def cv_entropy(ms, cf, tent, c, K=4):
    # standardize features on all tokens (unsupervised)
    X = np.stack([np.concatenate(cf), tent], 1).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    cent, lab = kmeans2(X, K, minit="++", seed=0)
    # split labels back to sequences
    ids, off = [], 0
    for cf_ in cf:
        n = len(cf_); ids.append(lab[off:off + n]); off += n
    n_seq = len(ms)
    folds = [list(range(i, n_seq, 5)) for i in range(5)]
    acc = cost = 0
    for f in range(5):
        te = folds[f]; tr = [i for i in range(n_seq) if i not in te]
        gs = fit_ids([ms[i] for i in tr], [cf[i] for i in tr], [ids[i] for i in tr], c, K)
        for i in te:
            m, cf_, id_ = ms[i], cf[i], ids[i]
            t = 0; N = len(m)
            while t < N:
                gg = min(gs[id_[t]], N - t)
                L = 0
                while L < gg and m[t + L]:
                    L += 1
                acc += L + 1; cost += 1 + gg * c; t += L + 1
    return acc / max(cost, 1), gs


if __name__ == "__main__":
    for name, c in [("dense", 0.099), ("moe8", 0.099), ("mtp", 0.03), ("teacher_qat", 0.268)]:
        data = load_data(name)
        if data is None:
            continue
        ms, cf = data
        tent = get_tent(np.concatenate([np.full(len(x), i) for i, x in enumerate(cf)]))
        sp1, g1 = cv_entropy(ms, cf, tent, c, K=4)
        sp2, g2 = cv_entropy(ms, cf, tent, c, K=8)
        print(f"{name:12s} c={c:.3f}  2D-K4 speedup={sp1:.3f} gammas={list(map(int,g1))}  "
              f"2D-K8 speedup={sp2:.3f} gammas={list(map(int,g2))}")
