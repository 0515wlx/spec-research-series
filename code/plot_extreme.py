"""Plot long-gamma acceptance for the extreme drafts."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/path/to/spec-research/out"
S = [
    ("dense-distill", "eval_dense_distill_full.json", "#888888", "--"),
    ("dense-RL", "eval_dense_rl_full.json", "#1f77b4", "-"),
    ("dense-ctrl (1.17M)", "eval_dense_ctrl_full.json", "#888888", ":"),
    ("bf16-ctrl (1.17M)", "eval_dense_bf16ctrl_full.json", "#1f77b4", ":"),
    ("PTQ-W4", "eval_ptq_w4_full.json", "#d62728", ":"),
    ("QAT-W4", "eval_dense_qat_w4_full.json", "#d62728", "-"),
    ("MoE8", "eval_moe8_keep8_full.json", "#2ca02c", "-"),
    ("MoE16", "eval_moe16_keep8_full.json", "#9467bd", "-"),
]
GAMMAS = [1, 2, 3, 5, 8, 12, 16, 24, 32]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for tag, fn, c, ls in S:
    d = json.load(open(os.path.join(D, fn)))
    taus = [d["per_gamma"][str(g)]["tau"] for g in GAMMAS]
    ax.plot(GAMMAS, taus, ls, color=c, marker="o", ms=3.5, lw=1.8, label=tag)
ax.set_xlabel("draft length γ")
ax.set_ylabel("mean accepted length τ (tokens/step)")
ax.set_xscale("log")
ax.set_xticks(GAMMAS)
ax.set_xticklabels([str(g) for g in GAMMAS])
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(D, "fig_tau.png"), dpi=160)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
KS = [1, 2, 3, 5, 8, 16, 32, 64]
for tag, fn, c, ls in S:
    if fn == "eval_ptq_w4_full.json":
        continue
    d = json.load(open(os.path.join(D, fn)))
    sv = d.get("survival", {})
    ys = [sv[str(k)] for k in KS]
    ax.plot(KS, ys, ls, color=c, marker="o", ms=3.5, lw=1.8, label=tag)
ax.set_xlabel("k (consecutive draft tokens)")
ax.set_ylabel("P(all k tokens accepted)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(KS)
ax.set_xticklabels([str(k) for k in KS])
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(D, "fig_survival.png"), dpi=160)
print("saved", os.path.join(D, "fig_tau.png"), os.path.join(D, "fig_survival.png"))
