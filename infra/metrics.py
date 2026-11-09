"""Token-level acceptance / divergence metrics and block simulations."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-12


@torch.no_grad()
def metrics_full(draft_logits: torch.Tensor, target_logits: torch.Tensor, target_ids: torch.Tensor) -> dict:
    """Exact full-vocab metrics.  Inputs are [T, V] / [T] on the same device."""
    draft_logits = draft_logits.float()
    target_logits = target_logits.float()
    logp_d = F.log_softmax(draft_logits, dim=-1)
    logp_t = F.log_softmax(target_logits, dim=-1)
    p = logp_t.exp()
    q = logp_d.exp()

    tv = 0.5 * (p - q).abs().sum(-1)
    kl_td = (p * (logp_t - logp_d)).sum(-1)
    kl_dt = (q * (logp_d - logp_t)).sum(-1)
    m = 0.5 * (p + q)
    logm = m.clamp_min(EPS).log()
    js = 0.5 * (p * (logp_t - logm)).sum(-1) + 0.5 * (q * (logp_d - logm)).sum(-1)
    hell = torch.sqrt((1.0 - (p * q).clamp_min(0).sqrt().sum(-1)).clamp_min(0))

    top2_logp = logp_t.topk(2, dim=-1).values
    top2_p = top2_logp.exp()
    margin = top2_p[:, 0] - top2_p[:, 1]

    pred_t = target_logits.argmax(-1)
    pred_d = draft_logits.argmax(-1)
    match = pred_t == pred_d
    target_logp = logp_t.gather(-1, pred_t.unsqueeze(-1)).squeeze(-1)
    rank = (draft_logits > draft_logits.gather(-1, pred_t.unsqueeze(-1))).sum(-1)
    ent_t = -(p * logp_t).sum(-1)
    ent_d = -(q * logp_d).sum(-1)

    return {
        "match": match,
        "tv": tv,
        "kl_td": kl_td,
        "kl_dt": kl_dt,
        "js": js,
        "hell": hell,
        "margin": margin,
        "rho": 2.0 * tv / margin.clamp_min(EPS),
        "pred_t": pred_t,
        "pred_d": pred_d,
        "target_ids": target_ids,
        "rank": rank,
        "ent_t": ent_t,
        "ent_d": ent_d,
        "p0": torch.zeros_like(tv),
        "q0": torch.zeros_like(tv),
    }


@torch.no_grad()
def metrics_topk(
    draft_logits: torch.Tensor,
    top_idx: torch.Tensor,
    top_logp: torch.Tensor,
    p0_t: torch.Tensor,
    target_argmax: torch.Tensor,
    target_ids: torch.Tensor,
    margin_t: Optional[torch.Tensor] = None,
    ent_t: Optional[torch.Tensor] = None,
) -> dict:
    """Metrics against a cached target top-k distribution (+ residual bucket).

    Shapes: draft_logits [T,V], top_idx [T,K], top_logp [T,K], p0_t [T], target_argmax [T].
    """
    draft_logits = draft_logits.float()
    logp_d = F.log_softmax(draft_logits, dim=-1)
    logp_d_k = logp_d.gather(-1, top_idx)
    q_k = logp_d_k.exp()
    q0 = (1.0 - q_k.sum(-1)).clamp_min(EPS)
    p_k = top_logp.float().exp()
    p0 = p0_t.float().clamp_min(EPS)

    tv = 0.5 * ((p_k - q_k).abs().sum(-1) + (p0 - q0).abs())
    kl_td = (p_k * (top_logp.float() - logp_d_k.clamp_min(-50))).sum(-1) + p0 * (p0.log() - q0.log())
    kl_dt = (q_k * (logp_d_k - top_logp.float())).sum(-1) + q0 * (q0.log() - p0.log())

    m_k = 0.5 * (p_k + q_k)
    m0 = 0.5 * (p0 + q0)
    ml = m_k.clamp_min(EPS).log()
    js = 0.5 * (p_k * (top_logp.float() - ml)).sum(-1) + 0.5 * (p0 * (p0.log() - m0.log()))
    js += 0.5 * (q_k * (logp_d_k - ml)).sum(-1) + 0.5 * (q0 * (q0.log() - m0.log()))

    pred_d = draft_logits.argmax(-1)
    match = pred_d == target_argmax
    rank = (draft_logits > draft_logits.gather(-1, target_argmax.unsqueeze(-1))).sum(-1)
    ent_d = -(logp_d.exp() * logp_d).sum(-1)

    if margin_t is None:
        margin_t = (p_k[:, 0] - p_k[:, 1]).clamp_min(EPS)
    if ent_t is None:
        ent_t = -(p_k * top_logp.float()).sum(-1)

    # Logit-space perturbation diagnostics.  For the target's top-2 tokens the
    # log-probability margin is an exact function of the target and draft logits
    # (the log-normalisers cancel in the difference).  ``d_margin_log`` is the
    # perturbation of that decision margin; ``err_k`` is the mean absolute
    # log-probability error on the target top-k support.
    tgt_margin_log = top_logp.float()[:, 0] - top_logp.float()[:, 1]
    draft_margin_log = logp_d_k[:, 0] - logp_d_k[:, 1]
    d_margin_log = draft_margin_log - tgt_margin_log
    err_k = (logp_d_k - top_logp.float()).abs().mean(-1)

    return {
        "match": match,
        "tv": tv,
        "kl_td": kl_td,
        "kl_dt": kl_dt,
        "js": js,
        "hell": torch.zeros_like(tv),
        "margin": margin_t,
        "rho": 2.0 * tv / margin_t.clamp_min(EPS),
        "margin_log": tgt_margin_log,
        "d_margin_log": d_margin_log,
        "abs_d_margin_log": d_margin_log.abs(),
        "err_k": err_k,
        "pred_d": pred_d,
        "target_ids": target_ids,
        "rank": rank,
        "ent_t": ent_t,
        "ent_d": ent_d,
        "p0": p0,
        "q0": q0,
    }


def simulate_blocks(match: np.ndarray, gammas: list[int]) -> dict:
    """Greedy block simulation on a binary acceptance sequence."""
    match = np.asarray(match, dtype=bool)
    N = len(match)
    out = {}
    for g in gammas:
        t = 0
        acc = prop = steps = 0
        lens = []
        while t < N:
            L = 0
            while L < g and t + L < N and match[t + L]:
                L += 1
            acc += L
            prop += min(g, N - t)
            lens.append(L + 1)
            steps += 1
            t += L + 1
        out[int(g)] = {
            "alpha": acc / max(prop, 1),
            "tau": float(np.mean(lens)) if lens else 1.0,
            "accepted": acc,
            "proposed": prop,
            "steps": steps,
        }
    return out


def survival(match: np.ndarray, ks: list[int]) -> dict:
    """S(k) = Pr[the next k drafts are all accepted], averaged over positions."""
    match = np.asarray(match, dtype=np.int8)
    out = {}
    for k in ks:
        if len(match) < k:
            out[int(k)] = None
            continue
        ok = np.convolve(match, np.ones(k, dtype=np.int8), "valid") == k
        out[int(k)] = float(ok.mean())
    return out


def clustering(match: np.ndarray) -> dict:
    """P(C|C), P(C|I) and block-start P(C|I)."""
    m = np.asarray(match, dtype=bool)
    if len(m) < 3:
        return {"P(C|C)": None, "P(C|I)": None, "block_start_p": None}
    cc = m[1:][m[:-1]].mean() if m[:-1].any() else None
    ci = m[1:][~m[:-1]].mean() if (~m[:-1]).any() else None
    return {"P(C|C)": float(cc) if cc is not None else None,
            "P(C|I)": float(ci) if ci is not None else None,
            "block_start_p": float(ci) if ci is not None else None}


def summarize(match: np.ndarray, gammas: list[int], ks: list[int]) -> dict:
    match = np.asarray(match, dtype=bool)
    q = 1.0 - float(match.mean()) if len(match) else None
    return {
        "tokens": int(len(match)),
        "q": q,
        "match_rate": float(match.mean()) if len(match) else None,
        "tau_inf": (1.0 / q) if q and q > 0 else None,
        "survival": survival(match, ks),
        "per_gamma": simulate_blocks(match, gammas),
        **clustering(match),
    }


def _aic_bic(y: np.ndarray, yhat: np.ndarray, k: int) -> tuple[float, float]:
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    rss = max(rss, 1e-18)
    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    return aic, bic


def fit_models(D: np.ndarray, q: np.ndarray) -> dict:
    """Fit floor / power / log-linear models.  D is capacity in billions of params."""
    from scipy.optimize import curve_fit

    D = np.asarray(D, dtype=float)
    q = np.asarray(q, dtype=float)
    res = {}

    def floor(D, qinf, A, alpha):
        return qinf + A * np.power(D, -alpha)

    def power(D, A, alpha):
        return A * np.power(D, -alpha)

    def loglin(D, a, b):
        return a + b * np.log10(D)

    def invlog(D, a, b):
        return a + b / np.log10(D + 1.0)

    fits = [
        ("floor", floor, [max(q.min() * 0.5, 0.0), max(q.max(), 1e-3), 0.3], (0, 0, 0), (1, 10, 5)),
        ("power", power, [max(q.max(), 1e-3), 0.3], (0, 0), (10, 5)),
        ("loglinear", loglin, [q.mean(), 0.0], (-np.inf, -np.inf), (np.inf, np.inf)),
        ("invlog", invlog, [q.mean(), 0.0], (-np.inf, -np.inf), (np.inf, np.inf)),
    ]
    for name, fn, p0, lo, hi in fits:
        try:
            popt, _ = curve_fit(fn, D, q, p0=p0, bounds=(lo, hi), maxfev=100000)
            yhat = fn(D, *popt)
            r2 = 1 - np.sum((q - yhat) ** 2) / np.sum((q - q.mean()) ** 2)
            aic, bic = _aic_bic(q, yhat, len(popt))
            res[name] = {"params": [float(x) for x in popt], "r2": float(r2),
                         "aic": aic, "bic": bic, "pred": yhat.tolist()}
        except Exception as e:
            res[name] = {"error": repr(e)}
    best = min((v for v in res.values() if "bic" in v), key=lambda v: v["bic"], default=None)
    res["_best_by_bic"] = best
    return res


def bootstrap_floor(D: np.ndarray, q_per_seq: list[np.ndarray], n_boot: int = 500, seed: int = 0) -> dict:
    """Bootstrap q_inf by resampling sequences within each capacity point."""
    from scipy.optimize import curve_fit

    rng = np.random.default_rng(seed)
    D = np.asarray(D, dtype=float)
    vals = []
    for _ in range(n_boot):
        qs = []
        for arr in q_per_seq:
            arr = np.asarray(arr, dtype=float)
            idx = rng.integers(0, len(arr), size=len(arr))
            qs.append(arr[idx].mean())
        qs = np.asarray(qs)
        try:
            popt, _ = curve_fit(
                lambda d, qinf, A, a: qinf + A * np.power(d, -a),
                D, qs, p0=[max(qs.min() * 0.5, 0.0), max(qs.max(), 1e-3), 0.3],
                bounds=([0, 0, 0], [1, 10, 5]), maxfev=50000,
            )
            vals.append(float(popt[0]))
        except Exception:
            continue
    if not vals:
        return {"q_inf_mean": None, "q_inf_ci95": None, "n": 0}
    vals = np.asarray(vals)
    return {
        "q_inf_mean": float(vals.mean()),
        "q_inf_ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
        "n": int(len(vals)),
    }
