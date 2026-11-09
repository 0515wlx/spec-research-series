"""How much can a confidence-gated dynamic draft budget buy? (uses saved routing/confidence + match data)"""
import json
import numpy as np

d = np.load('/path/to/spec-research/out/routing_analysis.npz')
match = d['match'].astype(bool)
conf = d['d_maxp']
N = len(match)
print('tokens', N, 'match', match.mean())


def run(policy, gammas, c, v, oracle=False):
    t = 0
    tok = cost = steps = 0
    drafted = 0
    while t < N:
        g = policy(t)
        L = 0
        while L < g and t + L < N and match[t + L]:
            L += 1
        drafted += g
        cost += g * c + v
        tok += L + 1
        steps += 1
        t += L + 1
    return dict(tokens=tok, cost=cost, steps=steps, drafted=drafted,
                tps_ratio=tok / cost, avg_g=drafted / max(steps, 1),
                tau=tok / max(steps, 1))


for name, c, v in [('measured stack (c=.35,v=2.4)', 0.35, 2.4), ('optimised serving (c=.10,v=1.2)', 0.10, 1.2)]:
    print('\n==', name)
    base = None
    for g in [2, 3, 4, 5, 6, 8, 12]:
        r = run(lambda t, g=g: g, g, c, v)
        if base is None:
            base = r
        print(f'  fixed gamma={g:2d}: tau={r["tau"]:.3f}  tok/cost={r["tps_ratio"]:.3f}  (rel {r["tps_ratio"]/base["tps_ratio"]:.3f})')
    # confidence-gated policies
    def pol(t):
        p = conf[t]
        if p < 0.7:
            return 2
        if p < 0.9:
            return 4
        if p < 0.98:
            return 6
        return 8
    r = run(pol, None, c, v)
    print(f'  gated (p<.7:2, <.9:4, <.98:6, else 8): tau={r["tau"]:.3f} avg_g={r["avg_g"]:.2f} '
          f'tok/cost={r["tps_ratio"]:.3f} (rel {r["tps_ratio"]/base["tps_ratio"]:.3f})')
    # oracle gamma: pick gamma with best tok/cost given the future
    def oracle_pick(t):
        best, bg = -1, 1
        for g in [1, 2, 3, 4, 5, 6, 8, 12]:
            L = 0
            while L < g and t + L < N and match[t + L]:
                L += 1
            val = (L + 1) / (g * c + v)
            if val > best:
                best, bg = val, g
        return bg
    r = run(oracle_pick, None, c, v)
    print(f'  oracle gamma: tau={r["tau"]:.3f} avg_g={r["avg_g"]:.2f} tok/cost={r["tps_ratio"]:.3f} '
          f'(rel {r["tps_ratio"]/base["tps_ratio"]:.3f})')
    # gated on first-token confidence only, with per-step adaptive gamma by expected value
    def pol_ev(t):
        p = conf[t]
        best, bg = (1) / (1 * c + v), 1
        for g in [2, 3, 4, 5, 6, 8, 12]:
            val = (1 + sum(p ** k for k in range(1, g))) / (g * c + v)
            if val > best:
                best, bg = val, g
        return bg
    r = run(pol_ev, None, c, v)
    print(f'  EV-optimal gamma from p1: tau={r["tau"]:.3f} avg_g={r["avg_g"]:.2f} '
          f'tok/cost={r["tps_ratio"]:.3f} (rel {r["tps_ratio"]/base["tps_ratio"]:.3f})')
