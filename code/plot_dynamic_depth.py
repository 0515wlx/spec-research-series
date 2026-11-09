"""Plot dynamic-depth / entropy clustering results and the equal-cost comparison."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from dynamic_depth import load_data, run_policy, fit_fixed, cv

D = "/path/to/spec-research/out"
MODELS = [("dense", 0.099, "#1f77b4"), ("moe8", 0.099, "#2ca02c"),
          ("mtp", 0.03, "#ff7f0e"), ("teacher_qat", 0.268, "#d62728")]
GAMMAS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))

# (a) fixed-gamma speedup at each model's own c
ax = axes[0, 0]
for name, c, col in MODELS:
    data = load_data(name)
    if data is None:
        continue
    ms, cf = data
    sp = [run_policy(ms, cf, ("fixed", g), c)["speedup"] for g in GAMMAS]
    ax.plot(GAMMAS, sp, "o-", ms=3, color=col, lw=1.5, label=f"{name} (c={c})")
ax.set_xlabel("fixed γ"); ax.set_ylabel("speedup τ/(1+γc)")
ax.set_title("(a) fixed depth at each model's cost")
ax.grid(alpha=0.3); ax.legend(fontsize=7)

# (b) dynamic gain (cluster) vs fixed
ax = axes[0, 1]
names, gains, oracles = [], [], []
for name, c, col in MODELS:
    data = load_data(name)
    if data is None:
        continue
    ms, cf = data
    r = cv(ms, cf, c, K=4)
    names.append(name)
    gains.append((r["cluster"]["speedup"] / r["fixed"]["speedup"] - 1) * 100)
    oracles.append((r["oracle"]["speedup"] / r["fixed"]["speedup"] - 1) * 100)
x = np.arange(len(names))
ax.bar(x - 0.18, gains, 0.36, label="confidence-cluster", color="#4c72b0")
ax.bar(x + 0.18, oracles, 0.36, label="oracle cluster", color="#dd8452")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("speedup gain vs fixed (%)")
ax.set_title("(b) dynamic-depth gain: weaker drafts gain more")
ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7)

# (c) tau advantage of teacher QAT over MTP grows with gamma
ax = axes[1, 0]
qat = load_data("teacher_qat"); mtp = load_data("mtp")
tau_q = [run_policy(*qat, ("fixed", g), 0.268)["tau"] for g in GAMMAS]
tau_m = [run_policy(*mtp, ("fixed", g), 0.03)["tau"] for g in GAMMAS]
ax.plot(GAMMAS, tau_q, "o-", ms=3, color="#d62728", label="teacher QAT τ(γ)")
ax.plot(GAMMAS, tau_m, "s-", ms=3, color="#ff7f0e", label="MTP τ(γ)")
ax2 = ax.twinx()
ax2.plot(GAMMAS, [a / b for a, b in zip(tau_q, tau_m)], "k:", lw=1.2)
ax2.set_ylabel("QAT / MTP τ ratio (dotted)")
ax.set_xlabel("γ"); ax.set_ylabel("τ")
ax.set_title("(c) τ advantage of large draft grows with γ")
ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")

# (d) equal-cost speedup (best fixed gamma per c)
ax = axes[1, 1]
cs = np.array([0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.27])
for name, _, col in MODELS:
    data = load_data(name)
    if data is None:
        continue
    ms, cf = data
    ys = []
    for c in cs:
        g = fit_fixed(ms, cf, c)
        ys.append(run_policy(ms, cf, ("fixed", g), c)["speedup"])
    ax.plot(cs, ys, "o-", ms=3, color=col, lw=1.5, label=name)
ax.set_xlabel("per-token draft cost c")
ax.set_ylabel("best fixed-γ speedup")
ax.set_title("(d) equal-cost: large draft wins at long γ")
ax.grid(alpha=0.3); ax.legend(fontsize=7)
ax.axvspan(0.13, 0.20, color="grey", alpha=0.12, label="W2/W3 QAT range")

fig.tight_layout()
fig.savefig(os.path.join(D, "fig_dynamic_depth.png"), dpi=160)
print("saved", os.path.join(D, "fig_dynamic_depth.png"))
