"""OOD tokens and long-gamma speculative decoding: is a small OOD rate catastrophic?

Part A: use the WikiText teacher-agree sequences to simulate block acceptance.
Part B: inject synthetic OOD tokens into MATH-500 match sequences (iid / clustered).
Part C: scaling/benefit model: tau_inf vs OOD rate q.
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/path/to/spec-research/out"
MODELS = [
    ("dense-distill", "wikitext_dense_distill_s1.npz", None, None, 85.50),
    ("dense-RL", "wikitext_dense_rl.npz", "composite2.npz", "d_pred", 87.01),
    ("dense-QAT-W4", "wikitext_dense_qat_w4.npz", None, None, 86.63),
    ("MoE8", "wikitext_moe8_keep8_distill.npz", "features_moe8.npz", "match", 88.32),
    ("MoE16", "wikitext_moe16_keep8_distill.npz", None, None, 88.08),
    ("MTP", "wikitext_mtp.npz", "composite2.npz", "m_pred", 95.74),
    ("teacher-QAT-W4", "wikitext_teacher_qat.npz", "features_teacher_qat.npz", "match", 96.97),
]


def simulate(match_seqs, gammas):
    out = {}
    for g in gammas:
        acc = prop = 0; lens = []
        for m in match_seqs:
            t, N = 0, len(m)
            while t < N:
                k = 0
                while k < g and t + k < N and m[t + k]:
                    k += 1
                acc += k; prop += min(g, N - t); lens.append(k + 1); t += k + 1
        out[int(g)] = {"alpha": acc / max(prop, 1), "tau": float(np.mean(lens)),
                       "acc": acc, "prop": prop, "steps": len(lens)}
    return out


def wikitest_seqs(name, fn):
    td = np.load(os.path.join(D, "wikitext_teacher.npz"))
    t1 = td["top_idx"][:, :, 0]
    d = np.load(os.path.join(D, fn))
    pred = d["pred"].reshape(t1.shape)
    match = pred == t1
    return [match[i] for i in range(match.shape[0])]


def math_seqs(fn, key):
    d = np.load(os.path.join(D, fn))
    if key == "match":
        m = d["match"].astype(bool)
    else:
        m = d[key] == d["tgt"]
    seq = d["seq"]
    return [np.asarray(m[seq == s], dtype=bool) for s in range(int(seq.max()) + 1)]


def inject(ms, q, run=1, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for m in ms:
        m = m.copy()
        N = len(m)
        n_runs = int(round(q * N / run))
        starts = rng.integers(0, max(N - run, 1), size=n_runs)
        for s in starts:
            m[s:s + run] = False
        out.append(m)
    return out


def main():
    gammas = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]
    print("== Part A: WikiText teacher-agree speculative decoding")
    wres = {}
    for name, wfn, mfn, mkey, math in MODELS:
        ms = wikitest_seqs(name, wfn)
        tau = simulate(ms, gammas)
        q = 1 - np.mean([m.mean() for m in ms])
        # run-length / clustering
        cc = [m[1:][m[:-1]].mean() for m in ms if len(m) > 2]
        ci = [m[1:][~m[:-1]].mean() for m in ms if len(m) > 2]
        wres[name] = {"q": float(q), "tau": {g: tau[g]["tau"] for g in gammas},
                      "P(C|C)": float(np.mean(cc)), "P(C|I)": float(np.mean(ci)),
                      "math": math}
        print(f"  {name:18s} q={q:.3f}  tau8={tau[8]['tau']:.2f}  tau32={tau[32]['tau']:.2f} "
              f"tau_inf~{tau[64]['tau']:.2f}  P(C|C)={np.mean(cc):.3f} P(C|I)={np.mean(ci):.3f}")

    print("\n== Part B: inject OOD into MATH-500 match sequences")
    inj = {}
    for name, wfn, mfn, mkey, math in MODELS:
        if mfn is None:
            continue
        base = math_seqs(mfn, mkey)
        base_tau = simulate(base, [32, 64])[32]["tau"]
        rows = {}
        for run in [1, 3, 5, 10]:
            for q in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]:
                seed = sum(map(ord, name)) * 10000 + run * 100 + int(q * 100)
                ms = base if q == 0 else inject(base, q, run=run, seed=seed)
                t = simulate(ms, [32, 64])
                rows[f"{run}:{q}"] = {"tau32": t[32]["tau"], "tau64": t[64]["tau"]}
        inj[name] = {"base_tau32": base_tau, "rows": rows}
        print(f"  {name:18s} base tau32={base_tau:.2f}")
        for run in [1, 3, 5, 10]:
            vals = [rows[f"{run}:{q}"]["tau32"] for q in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]]
            print(f"    run={run:2d} tau32 q=0/1/2/5/10/20%: " + " ".join(f"{v:.1f}" for v in vals))

    # fit 1/tau_inf = c + s*q
    qv = np.array([wres[n]["q"] for n in wres])
    tv = np.array([wres[n]["tau"][64] for n in wres])
    A = np.stack([np.ones_like(qv), qv], 1)
    coef, *_ = np.linalg.lstsq(A, 1 / tv, rcond=None)
    pred = A @ coef
    r2 = 1 - np.sum((1 / tv - pred) ** 2) / np.sum((1 / tv - (1 / tv).mean()) ** 2)
    print(f"\n== Part C: 1/tau_inf = {coef[0]:.5f} + {coef[1]:.5f}*q   R2={r2:.6f}")
    print(f"   => p_I ~= {1-coef[0]:.5f}; tau_inf ~= 1/(q + {coef[0]:.5f})")
    for q in [0.01, 0.02, 0.05, 0.10, 0.167, 0.20, 0.30]:
        print(f"   q={q:.3f}  tau_inf~{1/(q+coef[0]):.2f}")

    # plot: 2x2
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    ax = axes[0, 0]
    qs = np.linspace(0.005, 0.75, 300)
    ax.plot(qs, 1 / (qs + coef[0]), "k-", lw=1.5, label=r"$1/(q+\epsilon)$")
    for name in wres:
        ax.scatter(wres[name]["q"], wres[name]["tau"][64], s=35)
        ax.annotate(name, (wres[name]["q"], wres[name]["tau"][64]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("teacher-disagreement / OOD rate q")
    ax.set_ylabel(r"$\tau_\infty$ (γ=64)")
    ax.set_title(f"(a) long-γ τ ≈ 1/q  (R²={r2:.4f})")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[0, 1]
    for name in wres:
        xs = [g for g in gammas if g <= 32]
        ys = [wres[name]["tau"][g] for g in xs]
        ax.plot(xs, ys, "o-", ms=2.5, lw=1.2, label=f"{name} (q={wres[name]['q']:.2f})")
    ax.set_xlabel("γ"); ax.set_ylabel("τ on WikiText")
    ax.set_title("(b) WikiText long-γ saturates at ~1/q")
    ax.grid(alpha=0.3); ax.legend(fontsize=6)

    ax = axes[1, 0]
    for name in inj:
        rows = inj[name]["rows"]; base = inj[name]["base_tau32"]
        for run, ls in [(1, "-"), (3, "--"), (5, ":"), (10, "-.")]:
            qs2 = [0, 1, 2, 5, 10, 20]
            ys = [rows[f"{run}:{q/100}"]["tau32"] / base for q in qs2]
            ax.plot(qs2, ys, ls, marker="o", ms=2.5, lw=1.1,
                    label=f"{name} run={run}")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("injected OOD rate q (%)")
    ax.set_ylabel(r"$\tau(32)/\tau_{base}(32)$")
    ax.set_title("(c) a few % OOD halves long-γ τ")
    ax.grid(alpha=0.3); ax.legend(fontsize=5.5, ncol=2)

    ax = axes[1, 1]
    ax.loglog(qv, tv, "o", ms=5, label="WikiText models")
    qq = np.logspace(-2.3, 0, 200)
    ax.loglog(qq, 1 / qq, "k--", lw=1.0, label=r"$1/q$")
    for target in [5, 10, 20, 30]:
        ax.axhline(target, color="grey", ls=":", lw=0.8)
        ax.text(0.012, target * 1.05, f"q={1/target:.3f}", fontsize=6, color="grey")
    ax.set_xlabel("OOD rate q")
    ax.set_ylabel(r"$\tau_\infty$")
    ax.set_title("(d) scaling wall: need q<5% for τ>20")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_ood_catastrophe.png"), dpi=160)
    print("saved", os.path.join(D, "fig_ood_catastrophe.png"))
    json.dump({"wikitext": wres, "inject": inj,
               "fit_tau_inf": {"c": float(coef[0]), "slope": float(coef[1]), "R2": float(r2)}},
              open(os.path.join(D, "ood_catastrophe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
