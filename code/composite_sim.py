"""Acceptance-length simulation for dense / MTP / composite(gated) drafting."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

d = np.load('/path/to/spec-research/out/composite2.npz')
tgt, dp, dc = d['tgt'], d['d_pred'], d['d_conf'].astype(float)
mp, mc = d['m_pred'], d['m_conf'].astype(float)
tent, seq = d['t_ent'].astype(float), d['seq']
N = len(tgt)
md = dp == tgt
mm = mp == tgt
print(f'dense acc {md.mean():.4f}  mtp acc {mm.mean():.4f}  union {(md|mm).mean():.4f}')


def lag(x, k, fill=np.nan):
    y = np.full(N, fill, dtype=float)
    y[k:] = x[:-k]
    for s in np.unique(seq):
        idx = np.where(seq == s)[0]
        y[idx[:k]] = fill
    return y


ag = (dp == mp).astype(float)
feats = {'d_conf': dc, 'm_conf': mc, 'diff': dc - mc, 'agree': ag,
         'd_conf_l1': lag(dc, 1), 'm_conf_l1': lag(mc, 1),
         'tent_l1': lag(tent, 1), 'rej_l1': lag((~md).astype(float), 1)}
keys = list(feats)
X = np.column_stack([np.nan_to_num(feats[k]) for k in keys])
sq = np.unique(seq)
tr = np.isin(seq, sq[: int(0.8 * len(sq))])
te = ~tr
y = (mm & ~md).astype(int)
sc = StandardScaler().fit(X[tr])
clf = LogisticRegression(max_iter=3000).fit(sc.transform(X[tr]), y[tr])
p = np.full(N, np.nan)
p[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
# threshold chosen on train
ptr = clf.predict_proba(sc.transform(X[tr]))[:, 1]
best = (0, 0.5)
for thr in np.linspace(0.05, 0.95, 19):
    pick = np.where(ptr > thr, mp[tr], dp[tr])
    acc = (pick == tgt[tr]).mean()
    if acc > best[0]:
        best = (acc, thr)
thr = best[1]
print(f'gate thr {thr:.2f} (train acc {best[0]:.4f})')
pick_m = np.where(np.isnan(p), False, p > thr)
comp = np.where(pick_m, mp, dp)


def sim(match, gamma):
    tok = steps = 0
    lens = []
    t = 0
    while t < N:
        if seq[t] != seq[max(t - 1, 0)] and t > 0:
            pass
        L = 0
        while L < gamma and t + L < N and seq[t + L] == seq[t] and match[t + L]:
            L += 1
        tok += L + 1
        lens.append(L + 1)
        steps += 1
        t += L + 1
    return np.mean(lens)


for name, m in [('dense', md), ('mtp', mm), ('composite(gate)', (comp == tgt)),
                ('oracle union', (md | mm))]:
    out = ' '.join(f'tau({g})={sim(m, g):.2f}' for g in [1, 3, 5, 8])
    print(f'{name:18s} acc {(m.mean()):.4f}  {out}')

# same on the held-out split only
print('\nheld-out split only:')
for name, m in [('dense', md), ('mtp', mm), ('composite(gate)', (comp == tgt))]:
    mt = m.copy()
    mt[tr] = True  # count everything but report separately
    # recompute tau restricted to held-out sequences
    tok = steps = 0
    lens = []
    t = 0
    while t < N:
        if not te[t]:
            t += 1
            continue
        L = 0
        while L < 8 and t + L < N and seq[t + L] == seq[t] and m[t + L]:
            L += 1
        lens.append(L + 1)
        t += L + 1
    print(f'  {name:18s} acc {m[te].mean():.4f}  tau_heldout(8)={np.mean(lens):.2f}')
