"""Statistics used in the paper: profile-likelihood CIs, LOOCV, paired
bootstrap tests, real-loop prompt-level CIs, and top-k / entropy diagnostics.

All inputs are existing result files under ``out/``; this script does not run
any model.  Outputs ``out/paper_stats/stats.json`` and a markdown summary.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra.metrics import simulate_blocks  # noqa: E402

OUT = "out/paper_stats"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- models
def floor(D, p):
    qinf, A, alpha = p
    return qinf + A * np.power(D, -alpha)


def power(D, p):
    A, alpha = p
    return A * np.power(D, -alpha)


def loglin(D, p):
    a, b = p
    return a + b * np.log10(D)


def invlog(D, p):
    a, b = p
    return a + b / np.log10(D + 1.0)


MODELS = {
    "floor": (floor, [0.05, 0.2, 0.3], ([0, 0, 0], [1, 10, 10])),
    "power": (power, [0.2, 0.3], ([0, 0], [10, 10])),
    "loglinear": (loglin, [0.1, 0.0], ([-np.inf, -np.inf], [np.inf, np.inf])),
    "invlog": (invlog, [0.1, 0.0], ([-np.inf, -np.inf], [np.inf, np.inf])),
}


def fit_one(name, D, q):
    fn, p0, (lo, hi) = MODELS[name]
    res = least_squares(lambda p: fn(D, p) - q, p0, bounds=(lo, hi), max_nfev=100000)
    pred = fn(D, res.x)
    rss = float(np.sum((q - pred) ** 2))
    n, k = len(q), len(res.x)
    aic = n * np.log(max(rss / n, 1e-18)) + 2 * k
    bic = n * np.log(max(rss / n, 1e-18)) + k * np.log(n)
    r2 = 1 - rss / float(np.sum((q - q.mean()) ** 2))
    return {"params": [float(x) for x in res.x], "pred": pred.tolist(),
            "rss": rss, "r2": float(r2), "aic": float(aic), "bic": float(bic)}


def profile_ci(D, q, level=0.95, grid=600):
    """Profile-likelihood CI for q_inf in the floor model."""
    D = np.asarray(D, float)
    q = np.asarray(q, float)
    best = fit_one("floor", D, q)
    rss_min = best["rss"]
    n, k = len(q), 3
    sigma2 = rss_min / max(n - k, 1)
    thresh = rss_min + chi2.ppf(level, 1) * sigma2

    qmax = max(q.max() * 1.25, q.max() + 0.05)
    qs = np.linspace(0.0, qmax, grid)
    rss = []
    for qinf in qs:
        def resid(p):
            A, alpha = p
            return qinf + A * np.power(D, -alpha) - q
        try:
            r = least_squares(resid, [max(q.max(), 1e-3), 0.3],
                              bounds=([0, 0], [10, 10]), max_nfev=20000)
            rss.append(float(np.sum(resid(r.x) ** 2)))
        except Exception:
            rss.append(np.inf)
    rss = np.asarray(rss)
    ok = qs[rss <= thresh]
    if len(ok) == 0:
        return [None, None]
    return [float(ok.min()), float(ok.max())]


def loocv(D, q):
    D = np.asarray(D, float)
    q = np.asarray(q, float)
    out = {}
    for name in MODELS:
        errs = []
        for i in range(len(D)):
            mask = np.ones(len(D), bool)
            mask[i] = False
            try:
                f = fit_one(name, D[mask], q[mask])
                pred = MODELS[name][0](D[i:i + 1], np.asarray(f["params"]))[0]
                errs.append(float(pred - q[i]))
            except Exception:
                errs.append(np.nan)
        errs = np.asarray(errs, float)
        out[name] = {
            "rmse": float(np.sqrt(np.nanmean(errs ** 2))),
            "max_abs": float(np.nanmax(np.abs(errs))),
            "errors": errs.tolist(),
        }
    return out


def bootstrap_qinf(D, q_per_seq, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    D = np.asarray(D, float)
    vals = []
    for _ in range(n_boot):
        qs = []
        for arr in q_per_seq:
            arr = np.asarray(arr, float)
            qs.append(arr[rng.integers(0, len(arr), len(arr))].mean())
        try:
            vals.append(fit_one("floor", D, np.asarray(qs))["params"][0])
        except Exception:
            pass
    if not vals:
        return {"mean": None, "ci95": [None, None], "n": 0}
    vals = np.asarray(vals)
    return {"mean": float(vals.mean()),
            "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
            "median": float(np.median(vals)), "n": int(len(vals))}


def paired_bootstrap(q1, q2, n_boot=2000, seed=0):
    """Paired prompt-level bootstrap of q1 - q2 (same sequence order)."""
    a = np.asarray(q1, float)
    b = np.asarray(q2, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d.append(float(a[idx].mean() - b[idx].mean()))
    d = np.asarray(d)
    return {"diff": float(a.mean() - b.mean()),
            "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "p_lt0": float((d < 0).mean()), "n": n}


# ---------------------------------------------------------------- load fits
def load_fit(path):
    return json.load(open(path))


def add_fit_block(entry):
    D = np.asarray(entry["D"], float)
    q = np.asarray(entry["q"], float)
    fits = {name: fit_one(name, D, q) for name in MODELS}
    best_alt = min((fits[n]["bic"], n) for n in ["power", "loglinear", "invlog"])
    fits["_delta_bic_floor_vs_best_alt"] = float(best_alt[0] - fits["floor"]["bic"])
    fits["_delta_aic_floor_vs_best_alt"] = float(
        min(fits[n]["aic"] for n in ["power", "loglinear", "invlog"]) - fits["floor"]["aic"])
    fits["_best_alt"] = best_alt[1]
    entry["fits2"] = fits
    entry["profile_ci"] = profile_ci(D, q)
    entry["loocv"] = loocv(D, q)
    if entry.get("q_per_seq"):
        entry["bootstrap2"] = bootstrap_qinf(D, entry["q_per_seq"])
    return entry


def main():
    stats = {"domains": {}, "pairs": {}, "loop": {}, "topk": {}, "entropy": {}}

    # label -> (fit path, fit key, accept dir, filename prefix)
    sources = {
        "MATH-500": ("out/fit/capacity/fit.json", "math500", "out/accept", "math500_"),
        "GSM8K": ("out/fit/capacity/fit.json", "gsm8k", "out/accept", "gsm8k_"),
        "HumanEval": ("out/fit/capacity/fit.json", "humaneval", "out/accept", "humaneval_"),
        "MT-Bench": ("out/fit/capacity/fit.json", "mtbench", "out/accept", "mtbench_"),
        "WikiText-2 (raw)": ("out/fit/capacity/fit.json", "wikitext", "out/accept", "wikitext_"),
        "FineWeb-Edu (raw)": ("out/fit/fineweb/fit.json", "fineweb", "out/accept", "fineweb_"),
        "WikiText-2 (generated)": ("out/fit/generated_ood/fit.json", "gen", "out/accept",
                                   "wikitext_gen_"),
        "FineWeb-Edu (generated)": ("out/fit/fineweb_gen/fit.json", "gen", "out/accept",
                                    "fineweb_gen_"),
        "MATH-500 (Qwen3.6)": ("out/fit/repl_math500/fit.json", "math500",
                               "out/accept/repl", "repl_math500_"),
        "WikiText-2 raw (Qwen3.6)": ("out/fit/repl_wikitext/fit.json", "wikitext",
                                     "out/accept/repl", "repl_wikitext_"),
        "FineWeb-Edu raw (Qwen3.6)": ("out/fit/repl_fineweb/fit.json", "fineweb",
                                      "out/accept/repl", "repl_fineweb_"),
    }

    def load_qseq(entry, accept_dir, prefix):
        if entry.get("q_per_seq"):
            return entry["q_per_seq"]
        seqs = []
        for model in entry.get("models", []):
            p = os.path.join(accept_dir, f"{prefix}{model}.json")
            if not os.path.exists(p):
                return None
            seqs.append(json.load(open(p)).get("q_per_seq"))
        return seqs if all(s is not None for s in seqs) else None

    for label, (path, key, accept_dir, prefix) in sources.items():
        entry = load_fit(path)[key]
        entry["q_per_seq"] = load_qseq(entry, accept_dir, prefix)
        entry = add_fit_block(entry)
        stats["domains"][label] = entry

    # paired adjacent-capacity tests from q_per_seq
    for label, (path, key, accept_dir, prefix) in sources.items():
        e = load_fit(path)[key]
        e["q_per_seq"] = load_qseq(e, accept_dir, prefix)
        if not e.get("q_per_seq"):
            continue
        D = e["D"]
        pairs = {}
        for i in range(len(D) - 1):
            pairs[f"{D[i]:g}v{D[i+1]:g}"] = paired_bootstrap(
                e["q_per_seq"][i], e["q_per_seq"][i + 1])
        stats["pairs"][label] = pairs

    # pooled OOD fit: shared A and alpha, one intercept per raw OOD set
    ood_labels = ["WikiText-2 (raw)", "FineWeb-Edu (raw)", "WikiText-2 raw (Qwen3.6)",
                  "FineWeb-Edu raw (Qwen3.6)"]
    D = np.asarray(stats["domains"][ood_labels[0]]["D"], float)
    Y = np.concatenate([np.asarray(stats["domains"][k]["q"], float) for k in ood_labels])
    Dcat = np.tile(D, len(ood_labels))

    def pooled_pred(p):
        qinf, A, alpha = p[:len(ood_labels)], p[-2], p[-1]
        return np.concatenate([qinf[i] + A * Dcat[i * len(D):(i + 1) * len(D)] ** (-alpha)
                               for i in range(len(ood_labels))])

    def pooled_resid(p):
        return pooled_pred(p) - Y

    p0 = [0.12] * len(ood_labels) + [0.2, 0.4]
    lo = [0.0] * len(ood_labels) + [0.0, 0.0]
    hi = [1.0] * len(ood_labels) + [10.0, 10.0]
    r = least_squares(pooled_resid, p0, bounds=(lo, hi), max_nfev=100000)
    rss = float(np.sum(pooled_resid(r.x) ** 2))
    n, k = len(Y), len(r.x)
    sigma2 = rss / max(n - k, 1)
    thresh = rss + chi2.ppf(0.95, 1) * sigma2
    pooled = {"params": [float(x) for x in r.x],
              "r2": float(1 - rss / np.sum((Y - Y.mean()) ** 2)),
              "intercepts": {}}
    for j, label in enumerate(ood_labels):
        grid = np.linspace(0.0, 0.3, 300)
        vals = []
        for q in grid:
            def res2(p, j=j, q=q):
                qinf = list(p[:len(ood_labels)])
                qinf[j] = q
                A, alpha = p[-2], p[-1]
                return np.concatenate([qinf[i] + A * Dcat[i * len(D):(i + 1) * len(D)] ** (-alpha)
                                       for i in range(len(ood_labels))]) - Y
            p0b = list(r.x)
            p0b[j] = q
            try:
                rr = least_squares(res2, p0b, bounds=(lo, hi), max_nfev=20000)
                vals.append(float(np.sum(res2(rr.x) ** 2)))
            except Exception:
                vals.append(np.inf)
        ok = grid[np.asarray(vals) <= thresh]
        pooled["intercepts"][label] = {
            "q_inf": float(r.x[j]),
            "profile_ci95": ([float(ok.min()), float(ok.max())] if len(ok) else [None, None]),
        }
    stats["pooled_ood"] = pooled

    # real loop prompt-level CIs and paired teacher/real ratio
    for path in sorted(glob.glob("out/loop/*.json")):
        base = os.path.basename(path).replace(".json", "")
        d = json.load(open(path))
        dom, draft, gamma = d["domain"], d["draft"], d["gamma"]
        if "pilot" in base or d.get("n", 0) < 5:
            continue
        ids = [r["id"] for r in d["rows"]]
        real = np.asarray([r["tau"] for r in d["rows"]], float)
        npz = f"out/accept/{dom}_{draft}.npz"
        teacher = []
        if os.path.exists(npz):
            z = np.load(npz, allow_pickle=True)
            match, sid = z["match"], z["seq_id"]
            for i in ids:
                m = match[sid == i]
                teacher.append(simulate_blocks(m, [gamma])[gamma]["tau"])
        teacher = np.asarray(teacher, float)
        rng = np.random.default_rng(0)
        real_b = [real[rng.integers(0, len(real), len(real))].mean() for _ in range(5000)]
        ratio = real / teacher if len(teacher) == len(real) else None
        ratio_b = ([float(np.mean(ratio[rng.integers(0, len(ratio), len(ratio))]))
                    for _ in range(5000)] if ratio is not None else None)
        stats["loop"][base] = {
            "domain": dom, "draft": draft, "gamma": gamma, "n": len(real),
            "real_tau": float(real.mean()),
            "real_ci95": [float(np.quantile(real_b, 0.025)), float(np.quantile(real_b, 0.975))],
            "teacher_tau_per_prompt": float(teacher.mean()) if len(teacher) else None,
            "teacher_tau_concat": d.get("tau"),
            "ratio_per_prompt": float(np.mean(ratio)) if ratio is not None else None,
            "ratio_ci95": ([float(np.quantile(ratio_b, 0.025)), float(np.quantile(ratio_b, 0.975))]
                           if ratio_b is not None else None),
        }

    # top-k recall among rejected tokens + entropy/margin stratified q
    for dom in ["math500", "gsm8k", "humaneval", "mtbench", "wikitext", "fineweb",
                "wikitext_gen"]:
        for draft in ["qwen3.5-2b", "qwen3.5-27b"]:
            p = f"out/accept/{dom}_{draft}.npz"
            if not os.path.exists(p):
                continue
            z = np.load(p, allow_pickle=True)
            match, rank = z["match"], z["rank"]
            rej = ~match
            if not rej.any():
                continue
            rec = {f"top{k}": float((rank[rej] < k).mean()) for k in [1, 4, 9, 49, 255]}
            stats["topk"][f"{dom}/{draft}"] = {
                "q": float(rej.mean()), "n_rej": int(rej.sum()), **rec}
            # entropy / margin quintiles
            for field in ["ent_t", "margin"]:
                x = z[field]
                try:
                    edges = np.quantile(x, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    bins = np.clip(np.digitize(x, edges[1:-1]), 0, 4)
                    qs = [float((~match[bins == b]).mean()) for b in range(5)]
                    stats["entropy"][f"{dom}/{draft}/{field}"] = {
                        "edges": [float(v) for v in edges], "q_by_quintile": qs}
                except Exception:
                    pass

    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)

    # markdown summary
    lines = ["# Paper statistics (from existing caches)", ""]
    lines += ["## Fit diagnostics", "",
              "| set | q_inf | boot 95% | profile 95% | ΔBIC vs best alt | LOO RMSE floor | LOO RMSE best alt |",
              "|---|---:|---|---|---:|---:|---:|"]
    def fmt(x, nd=4):
        return "n/a" if x is None else f"{x:.{nd}f}"

    for label, e in stats["domains"].items():
        qinf = e["fits2"]["floor"]["params"][0]
        b = e.get("bootstrap2") or {}
        ci = b.get("ci95") or [None, None]
        p = e.get("profile_ci") or [None, None]
        lf = e["loocv"]["floor"]["rmse"]
        best = e["fits2"]["_best_alt"]
        lb = e["loocv"][best]["rmse"]
        lines.append(f"| {label} | {qinf:.4f} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
                     f"[{fmt(p[0])}, {fmt(p[1])}] | "
                     f"{e['fits2']['_delta_bic_floor_vs_best_alt']:.2f} | {lf:.4f} | {lb:.4f} |")
    lines += ["", "## Pooled raw-OOD fit (shared A, alpha; one intercept per set)", ""]
    p = stats["pooled_ood"]
    lines.append(f"R2={p['r2']:.4f}, shared A={p['params'][-2]:.4f}, alpha={p['params'][-1]:.4f}")
    for k, v in p["intercepts"].items():
        lines.append(f"- {k}: q_inf={v['q_inf']:.4f}, profile 95% CI "
                     f"[{fmt(v['profile_ci95'][0])}, {fmt(v['profile_ci95'][1])}]")
    lines += ["", "## Adjacent-capacity paired bootstrap (q_i - q_{i+1})", ""]
    for label, pairs in stats["pairs"].items():
        lines.append(f"**{label}**")
        for k, v in pairs.items():
            lines.append(f"- {k}: {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}], P(diff<0)={v['p_lt0']:.3f}")
        lines.append("")
    lines += ["## Real loop prompt-level", "",
              "| loop | n | real tau | 95% CI | teacher (per-prompt) | ratio | ratio 95% CI |",
              "|---|---:|---:|---|---:|---:|---|"]
    for k, v in stats["loop"].items():
        lines.append(f"| {k} | {v['n']} | {v['real_tau']:.3f} | "
                     f"[{v['real_ci95'][0]:.3f}, {v['real_ci95'][1]:.3f}] | "
                     f"{v['teacher_tau_per_prompt']:.3f} | {v['ratio_per_prompt']:.3f} | "
                     f"[{v['ratio_ci95'][0]:.3f}, {v['ratio_ci95'][1]:.3f}] |")
    lines += ["", "## Top-k recall among rejected tokens", "",
              "| cell | q | top-1 | top-5 | top-10 | top-50 | top-256 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in stats["topk"].items():
        lines.append(f"| {k} | {v['q']:.4f} | {v['top1']:.3f} | {v['top4']:.3f} | "
                     f"{v['top9']:.3f} | {v['top49']:.3f} | {v['top255']:.3f} |")
    lines += ["", "## q by target-entropy quintile", ""]
    for k, v in stats["entropy"].items():
        if k.endswith("ent_t"):
            lines.append(f"- {k}: " + ", ".join(f"{x:.3f}" for x in v["q_by_quintile"]))
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nsaved", os.path.join(OUT, "stats.json"))


if __name__ == "__main__":
    main()
