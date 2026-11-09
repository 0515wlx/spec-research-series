"""Depth law: is acceptance linear in depth, or diminishing returns?

Outputs the table used in REPORT §14:
  - p (marginal token match), alpha(1), alpha(5), local slope, -p(1-p)/2
  - stretched-exponential beta (fit to survival), S(32)/p^32, p_64/p_1
  - linear-vs-nonlinear fit comparison, conditional p_k, Markov check
"""
import json, os
import numpy as np
from scipy.optimize import least_squares
from scipy.special import betaln

D = "/path/to/spec-research/out"


def load_composite():
    d = np.load(os.path.join(D, "composite2.npz"))
    tgt, seq = d["tgt"], d["seq"]
    out = {}
    for name, pred in [("dense", d["d_pred"]), ("mtp", d["m_pred"])]:
        ms = []
        for s in range(int(seq.max()) + 1):
            idx = np.where(seq == s)[0]
            ms.append(np.asarray(pred[idx] == tgt[idx], dtype=bool))
        out[name] = ms
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
        if cnt == 0:
            break
        S[k] = tot / cnt
    return S


def alpha_matches(ms, gammas):
    out = {}
    for g in gammas:
        acc = prop = 0
        for x in ms:
            t, N = 0, len(x)
            while t < N:
                k = 0
                while k < g and t + k < N and x[t + k]:
                    k += 1
                acc += k; prop += min(g, N - t); t += k + 1
        out[g] = acc / max(prop, 1)
    return out


def load_json(fn):
    d = json.load(open(os.path.join(D, fn)))
    S = {int(k): v for k, v in d.get("survival", {}).items()}
    A = {int(k): v["alpha"] for k, v in d["per_gamma"].items()}
    return d["pos_match_rate"], S, A


def fit_stretched(S):
    ks = np.array(sorted(S), dtype=float)
    y = np.log(np.array([S[k] for k in ks]))
    def res(th):
        lam, beta = np.exp(th)
        return (ks / lam) ** beta + y
    r = least_squares(res, np.log([10.0, 0.8]),
                      bounds=(np.log([1e-3, 1e-3]), np.log([1e4, 5.0])))
    return float(np.exp(r.x[0])), float(np.exp(r.x[1]))


def fit_linear_pk(S):
    """S(k)=prod(p0-b*i); if the best b is ~0, linear per-token decay is rejected."""
    ks = np.arange(1, len(S) + 1, dtype=float)
    y = np.array([S[int(k)] for k in ks])
    def s_lin(th):
        p0, b = th
        return np.cumprod(np.clip(p0 - b * ks, 1e-6, 1.0))
    r = least_squares(lambda th: np.log(np.maximum(s_lin(th), 1e-12)) - np.log(y),
                      [y[0], 1e-3], bounds=([0.5, 0.0], [1.0, 0.1]))
    return float(r.x[0]), float(r.x[1])


def r2(y, yh):
    y = np.asarray(y); yh = np.asarray(yh)
    return 1 - np.sum((y - yh) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)


def markov_sim(a, b, N=1_500_000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(N, dtype=bool)
    x[0] = rng.random() < b / (1 - a + b)
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


def main():
    comp = load_composite()
    rows = []
    print(f"{'model':22s} {'p':>7s} {'a1':>7s} {'a5':>7s} {'slope':>7s} "
          f"{'-p(1-p)/2':>10s} {'beta':>6s} {'S32/p32':>8s} {'p64/p1':>7s}")
    for name, ms in comp.items():
        S = survival(ms, 64)
        p = sum(int(x.sum()) for x in ms) / sum(len(x) for x in ms)
        A = alpha_matches(ms, [1, 5])
        _, beta = fit_stretched(S)
        pk1, pk64 = S[1], (S[64] / S[32]) ** (1 / 32)
        rows.append((name, p, A[1], A[5], beta, S, pk1, pk64))
    for name, fn in [("dense-RL(500)", "eval_dense_rl_full.json"),
                     ("MoE8(500)", "eval_moe8_keep8_full.json"),
                     ("teacher PTQ-W4(500)", "eval_teacher_ptq_w4_500.json"),
                     ("teacher QAT-W4(100)", "eval_teacher_qat_w4_100.json")]:
        p, S, A = load_json(fn)
        if not S:
            continue
        _, beta = fit_stretched(S)
        pk1, pk64 = S[1], (S[64] / S[32]) ** (1 / 32)
        rows.append((name, p, A.get(1, np.nan), A.get(5, np.nan), beta, S, pk1, pk64))
    for name, p, a1, a5, beta, S, pk1, pk64 in rows:
        slope = (a5 - a1) / 4
        print(f"{name:22s} {p:7.4f} {a1:7.4f} {a5:7.4f} {slope:7.4f} "
              f"{-p*(1-p)/2:10.4f} {beta:6.3f} {S[32]/p**32:8.2f} {pk64/pk1:7.3f}")

    print("\nlinear-vs-nonlinear fits (dense / MTP):")
    for name, ms in comp.items():
        S = survival(ms, 64)
        ks = np.array(sorted(S), dtype=float)
        y = np.array([S[int(k)] for k in ks])
        p0, b = fit_linear_pk(S)
        y_lin = np.cumprod(np.clip(p0 - b * ks, 1e-6, 1.0))
        p_exp = float(np.exp(np.sum(ks * np.log(y)) / np.sum(ks * ks)))
        y_exp = p_exp ** ks
        lam, beta = fit_stretched(S)
        y_st = np.exp(-(ks / lam) ** beta)
        print(f"  {name}: exp R2={r2(y,y_exp):.4f}  linear-pk R2={r2(y,y_lin):.4f} "
              f"(b={b:.2e})  stretched R2={r2(y,y_st):.4f} (lambda={lam:.2f}, beta={beta:.3f})")

    print("\nconditional p_k (sliding survival):")
    for name, ms in comp.items():
        S = survival(ms, 64)
        pk = {1: S[1]}
        for k in range(2, 65):
            pk[k] = S[k] / S[k - 1]
        print("  " + name, " ".join(f"{k}:{pk[k]:.3f}" for k in (1, 2, 4, 8, 16, 32, 64)))

    print("\nMarkov check (block starts after failures):")
    gammas = list(range(1, 33))
    for name, ms in comp.items():
        cc = [x[1:][x[:-1]].mean() for x in ms if len(x) > 2]
        ci = [x[1:][~x[:-1]].mean() for x in ms if len(x) > 2]
        a, b = float(np.mean(cc)), float(np.mean(ci))
        x = markov_sim(a, b)
        Am = alpha_flat(x, gammas)
        A = alpha_matches(ms, gammas)
        print(f"  {name}: P(C|C)={a:.4f} P(C|I)={b:.4f}  "
              f"emp a(32)={A[32]:.4f}  markov a(32)={Am[32]:.4f}")


if __name__ == "__main__":
    main()
