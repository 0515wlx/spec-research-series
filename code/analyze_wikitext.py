"""WikiText distribution-shift summary and plots."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/path/to/spec-research/out"
MODELS = [
    ("dense-distill", "wikitext_dense_distill_s1.npz", 85.50, "#1f77b4"),
    ("dense-RL", "wikitext_dense_rl.npz", 87.01, "#4c72b0"),
    ("dense-QAT-W4", "wikitext_dense_qat_w4.npz", 86.63, "#9467bd"),
    ("MoE8", "wikitext_moe8_keep8_distill.npz", 88.32, "#2ca02c"),
    ("MoE16", "wikitext_moe16_keep8_distill.npz", 88.08, "#8c564b"),
    ("MTP", "wikitext_mtp.npz", 95.74, "#ff7f0e"),
    ("teacher-QAT-W4", "wikitext_teacher_qat.npz", 96.97, "#d62728"),
]


def main():
    td = np.load(os.path.join(D, "wikitext_teacher.npz"))
    t_nll = float(td["nll"].mean())
    t1 = td["top_idx"][:, :, 0].reshape(-1)
    rows = []
    for name, fn, math, col in MODELS:
        p = os.path.join(D, fn)
        if not os.path.exists(p):
            continue
        d = np.load(p)
        rows.append({"name": name, "math": math, "col": col, "d": d,
                     "nll": float(d["nll"].mean()), "match": float(d["match"].mean()),
                     "teacher_agree": float((d["pred"].reshape(-1) == t1).mean()),
                     "tv": float(d["tv"].mean()), "kl": float(d["kl_td"].mean()),
                     "ent": float(d["ent"].mean())})
    print(f"teacher NLL {t_nll:.4f}")
    print(f"{'model':18s} {'NLL':>6s} {'excess':>7s} {'top1':>6s} {'agree':>6s} {'TV':>6s} {'KL_TD':>6s} {'ent':>6s} {'MATH':>6s}")
    for r in rows:
        print(f"{r['name']:18s} {r['nll']:6.3f} {r['nll']-t_nll:7.3f} "
              f"{r['match']*100:5.1f}% {r['teacher_agree']*100:5.1f}% {r['tv']:6.3f} "
              f"{r['kl']:6.3f} {r['ent']:6.3f} {r['math']:5.1f}%")

    tv = np.array([r["tv"] for r in rows]); math = np.array([r["math"] for r in rows]) / 100
    wt = np.array([r["match"] for r in rows]); nll = np.array([r["nll"] for r in rows])
    kl = np.array([r["kl"] for r in rows])

    def lin(x, y):
        A = np.stack([np.ones_like(x), x], 1); c, *_ = np.linalg.lstsq(A, y, rcond=None)
        pr = A @ c; r2 = 1 - np.sum((y - pr) ** 2) / np.sum((y - y.mean()) ** 2)
        return c, r2, pr

    c_tv, r2_tv, pr_tv = lin(tv, math)
    c_kl, r2_kl, pr_kl = lin(np.sqrt(kl), math)
    c_nll, r2_nll, pr_nll = lin(np.log(nll), math)
    print(f"\nMATH = {c_tv[0]:.4f}{c_tv[1]:+.4f}*TV  R2={r2_tv:.4f}")
    print(f"MATH = {c_kl[0]:.4f}{c_kl[1]:+.4f}*sqrt(KL)  R2={r2_kl:.4f}")
    print(f"MATH = {c_nll[0]:.4f}{c_nll[1]:+.4f}*log(NLL)  R2={r2_nll:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    ax = axes[0, 0]
    for r in rows:
        ax.scatter(r["tv"], r["math"], color=r["col"], s=35)
        ax.annotate(r["name"], (r["tv"], r["math"]), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points")
    xs = np.linspace(0.1, 0.75, 50)
    ax.plot(xs, (c_tv[0] + c_tv[1] * xs) * 100, "k--", lw=1.0,
            label=f"fit R²={r2_tv:.2f}")
    ax.set_xlabel("mean TV(teacher || draft) on WikiText")
    ax.set_ylabel("MATH-500 top-1 match (%)")
    ax.set_title("(a) WikiText TV predicts math acceptance")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[0, 1]
    c_wt, r2_wt, _ = lin(tv, wt)
    for r in rows:
        ax.scatter(r["tv"], r["match"] * 100, color=r["col"], s=35)
        ax.annotate(r["name"], (r["tv"], r["match"] * 100), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points")
    ax.plot(xs, (c_wt[0] + c_wt[1] * xs) * 100, "k--", lw=1.0, label=f"fit R²={r2_wt:.2f}")
    ax.set_xlabel("mean TV on WikiText")
    ax.set_ylabel("WikiText top-1 match (%)")
    ax.set_title("(b) same shift predicts WikiText top-1")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[1, 0]
    for r in rows:
        d = r["d"]; qs = np.quantile(d["tv"], np.linspace(0, 1, 11))
        xs2, ys2 = [], []
        for i in range(10):
            sel = (d["tv"] >= qs[i]) & (d["tv"] <= qs[i + 1] if i == 9 else d["tv"] < qs[i + 1])
            if sel.sum():
                xs2.append(d["tv"][sel].mean()); ys2.append(d["match"][sel].mean() * 100)
        ax.plot(xs2, ys2, "o-", ms=2.5, lw=1.1, color=r["col"], label=r["name"])
    ax.set_xlabel("TV bin mean")
    ax.set_ylabel("top-1 match (%)")
    ax.set_title("(c) per-token: TV bins → top-1")
    ax.grid(alpha=0.3); ax.legend(fontsize=5.5)

    ax = axes[1, 1]
    for r in rows:
        ax.scatter(r["nll"], r["math"], color=r["col"], s=35)
        ax.annotate(r["name"], (r["nll"], r["math"]), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points")
    xs2 = np.linspace(1.7, 6.0, 50)
    ax.plot(xs2, (c_nll[0] + c_nll[1] * np.log(xs2)) * 100, "k--", lw=1.0,
            label=f"fit R²={r2_nll:.2f}")
    ax.set_xlabel("WikiText NLL (nats/token)")
    ax.set_ylabel("MATH-500 top-1 match (%)")
    ax.set_title("(d) NLL also predicts acceptance")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_wikitext_shift.png"), dpi=160)
    print("saved", os.path.join(D, "fig_wikitext_shift.png"))

    json.dump({"teacher": {"nll": t_nll, "ppl": float(np.exp(t_nll))},
               "models": [{k: v for k, v in r.items() if k != "d"} for r in rows],
               "fits": {"math_tv": [c_tv.tolist(), r2_tv],
                        "math_sqrt_kl": [c_kl.tolist(), r2_kl],
                        "math_log_nll": [c_nll.tolist(), r2_nll],
                        "wt_tv": [c_wt.tolist(), r2_wt]}},
              open(os.path.join(D, "wikitext_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
