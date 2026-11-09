"""Depth law: alpha(gamma) linear? conditional p_k? Markov/boundary-selection check."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/path/to/spec-research/out"
d = np.load(os.path.join(D, "composite2.npz"))
tgt, seq = d["tgt"], d["seq"]


def get_ms(pred):
    out = []
    for s in range(int(seq.max()) + 1):
        idx = np.where(seq == s)[0]
        out.append(np.asarray(pred[idx] == tgt[idx], dtype=bool))
    return out


def alpha(ms, gammas):
    out = {}
    for g in gammas:
        acc = prop = 0
        for x in ms:
            t, N = 0, len(x)
            while t < N:
                k = 0
                while k < g and t + k < N and x[t + k]:
                    k += 1
                acc += k
                prop += min(g, N - t)
                t += k + 1
        out[g] = acc / max(prop, 1)
    return out


def survival(ms, kmax=64):
    S = {}
    for k in range(1, kmax + 1):
        tot = cnt = 0
        for x in ms:
            if len(x) < k:
                continue
            ok = np.convolve(x.astype(np.int8), np.ones(k, dtype=np.int8), "valid") == k
            tot += int(ok.sum()); cnt += len(ok)
        S[k] = tot / cnt
    return S


def markov_sim(a, b, N=1_500_000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(N, dtype=bool)
    p = b / (1 - a + b)
    x[0] = rng.random() < p
    u = rng.random(N)
    for i in range(1, N):
        x[i] = u[i] < (a if x[i - 1] else b)
    return x


def alpha_flat(x, gammas):
    out = {}
    for g in gammas:
        t, N = 0, len(x); acc = prop = 0
        while t < N:
            k = 0
            while k < g and t + k < N and x[t + k]:
                k += 1
            acc += k; prop += min(g, N - t); t += k + 1
        out[g] = acc / max(prop, 1)
    return out


gammas = np.arange(1, 33)
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
ax = axes[0, 0]
for name, pred, c in [("dense", d["d_pred"], "#1f77b4"), ("MTP", d["m_pred"], "#ff7f0e")]:
    ms = get_ms(pred)
    A = alpha(ms, gammas)
    p = sum(int(x.sum()) for x in ms) / sum(len(x) for x in ms)
    gs = np.array([1, 2, 3, 4, 5, 6]); ys = np.array([A[g] for g in gs])
    coef = np.polyfit(gs, ys, 1)
    ax.plot(gammas, [A[g] for g in gammas], "o", ms=3, color=c, label=f"{name} empirical")
    ax.plot(gammas, np.polyval(coef, gammas), "--", color=c, lw=1.0,
            label=f"{name} linear γ≤6 (slope {coef[0]:.4f})")
    ax.plot(gammas, p * (1 - p ** gammas) / (gammas * (1 - p)), ":", color=c, lw=1.0)
ax.set_xlabel("γ"); ax.set_ylabel("block acceptance α(γ)")
ax.set_title("(a) α(γ): linear only near γ≲6, then concave")
ax.grid(alpha=0.3); ax.legend(fontsize=6.5)

ax = axes[0, 1]
for name, pred, c in [("dense", d["d_pred"], "#1f77b4"), ("MTP", d["m_pred"], "#ff7f0e")]:
    ms = get_ms(pred); S = survival(ms, 64)
    pk = {1: S[1]}
    for k in range(2, 65):
        pk[k] = S[k] / S[k - 1]
    ax.plot(list(pk), [pk[k] for k in pk], "-", color=c, lw=1.6, label=f"{name} p_k")
    ax.axhline(S[1], ls=":", color=c, lw=0.9)
ax.set_xlabel("depth k"); ax.set_ylabel("conditional P(accept_k | first k-1 accepted)")
ax.set_title("(b) conditional acceptance rises with depth (selection)")
ax.set_xscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=7)

ax = axes[1, 0]
for name, pred, c in [("dense", d["d_pred"], "#1f77b4"), ("MTP", d["m_pred"], "#ff7f0e")]:
    ms = get_ms(pred); S = survival(ms, 64); p = S[1]
    ks = np.array(sorted(S))
    ax.plot(ks, [S[k] / p ** k for k in ks], "-", color=c, lw=1.6, label=f"{name} S(k)/p^k")
ax.set_xlabel("k"); ax.set_ylabel("S(k) / p^k")
ax.set_title("(c) long runs are heavier-tailed than i.i.d.")
ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=7)

ax = axes[1, 1]
for name, pred, c in [("dense", d["d_pred"], "#1f77b4"), ("MTP", d["m_pred"], "#ff7f0e")]:
    ms = get_ms(pred); A = alpha(ms, gammas)
    p = sum(int(x.sum()) for x in ms) / sum(len(x) for x in ms)
    # transition probs
    cc, ci = [], []
    for y in ms:
        cc.append(y[1:][y[:-1]].mean()); ci.append(y[1:][~y[:-1]].mean())
    a, b = float(np.mean(cc)), float(np.mean(ci))
    x = markov_sim(a, b)
    Am = alpha_flat(x, gammas)
    ax.plot(gammas, [A[g] for g in gammas], "o", ms=3, color=c, label=f"{name} empirical")
    ax.plot(gammas, [Am[g] for g in gammas], "-", color=c, lw=1.2,
            label=f"{name} Markov(C|C={a:.3f},C|I={b:.3f})")
ax.set_xlabel("γ"); ax.set_ylabel("α(γ)")
ax.set_title("(d) first-order correctness Markov + block starts")
ax.grid(alpha=0.3); ax.legend(fontsize=6.5)

fig.tight_layout()
fig.savefig(os.path.join(D, "fig_depth_law.png"), dpi=160)
print("saved", os.path.join(D, "fig_depth_law.png"))
