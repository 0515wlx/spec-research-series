"""Plot the original-teacher W4 PTQ/QAT draft vs the dense reference and the ideal."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/path/to/spec-research/out"
GAMMAS = [1, 2, 3, 5, 8, 12, 16, 24, 32, 48]


def get(fn):
    return json.load(open(os.path.join(D, fn)))


ptq500 = get("eval_teacher_ptq_w4_500.json")
ptq100 = get("eval_teacher_ptq_w4_100.json")
qat100 = get("eval_teacher_qat_w4_100.json")
dense = get("eval_moe8_keep8_full.json")

fig, ax = plt.subplots(figsize=(6.6, 4.3))
ax.plot(GAMMAS, [g + 1 for g in GAMMAS], "k--", lw=1.2, label="ideal (bf16 teacher)")
ax.plot([g for g in GAMMAS if str(g) in ptq500["per_gamma"]],
        [ptq500["per_gamma"][str(g)]["tau"] for g in GAMMAS if str(g) in ptq500["per_gamma"]],
        color="#d62728", marker="o", ms=3.5, lw=1.8, label="teacher PTQ-W4 (500)")
ax.plot([g for g in GAMMAS if str(g) in ptq100["per_gamma"]],
        [ptq100["per_gamma"][str(g)]["tau"] for g in GAMMAS if str(g) in ptq100["per_gamma"]],
        color="#d62728", ls=":", marker="o", ms=3, lw=1.6, label="teacher PTQ-W4 (100)")
ax.plot([g for g in GAMMAS if str(g) in qat100["per_gamma"]],
        [qat100["per_gamma"][str(g)]["tau"] for g in GAMMAS if str(g) in qat100["per_gamma"]],
        color="#ff7f0e", marker="s", ms=3.5, lw=1.9, label="teacher QAT-W4 (100)")
ax.plot([g for g in GAMMAS if str(g) in dense["per_gamma"]],
        [dense["per_gamma"][str(g)]["tau"] for g in GAMMAS if str(g) in dense["per_gamma"]],
        color="#2ca02c", marker="^", ms=3.5, lw=1.6, label="dense MoE8 (reference)")
ax.set_xlabel("draft length γ")
ax.set_ylabel("mean accepted length τ")
ax.set_xscale("log")
ax.set_xticks(GAMMAS)
ax.set_xticklabels([str(g) for g in GAMMAS])
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(D, "fig_teacher_qat_tau.png"), dpi=160)

fig, ax = plt.subplots(figsize=(6.6, 4.3))
KS = [1, 2, 3, 5, 8, 16, 32, 64]
for tag, d, c, ls, mk in [
        ("teacher PTQ-W4 (500)", ptq500, "#d62728", "-", "o"),
        ("teacher QAT-W4 (100)", qat100, "#ff7f0e", "-", "s"),
        ("dense MoE8", dense, "#2ca02c", "-", "^")]:
    sv = d.get("survival", {})
    ax.plot(KS, [sv[str(k)] for k in KS], ls, color=c, marker=mk, ms=3.5, lw=1.8, label=tag)
ax.plot(KS, [1.0] * len(KS), "k--", lw=1.2, label="ideal (bf16 teacher)")
ax.set_xlabel("k (consecutive draft tokens)")
ax.set_ylabel("P(all k tokens accepted)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(KS)
ax.set_xticklabels([str(k) for k in KS])
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(D, "fig_teacher_qat_survival.png"), dpi=160)
print("saved teacher qat figures")
