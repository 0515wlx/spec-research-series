"""Branch-point accounting + anticipation analysis."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

d = np.load('/path/to/spec-research/out/branch_tokens_full.npz')
tmax, margin, tent = d['tmax'].astype(np.float64), d['margin'].astype(np.float64), d['tent'].astype(np.float64)
dmax, dent = d['dmax'].astype(np.float64), d['dent'].astype(np.float64)
match = d['match'].astype(bool)
seq = d['seq']
rej = ~match
N = len(match)
print(f'tokens {N}  match {match.mean():.4f}  reject {rej.mean():.4f}')

defs = {
    'teacher_tmax<0.50': tmax < 0.50,
    'teacher_tmax<0.90': tmax < 0.90,
    'teacher_tmax<0.99': tmax < 0.99,
    'teacher_ent>q90': tent > np.quantile(tent, 0.90),
    'teacher_ent>q80': tent > np.quantile(tent, 0.80),
    'top1-top2_margin<0.10': margin < 0.10,
}
print('\n=== A. rejected-token composition ===')
print(f"{'definition':24s} {'baserate':>9} {'P(rej|B)':>9} {'P(rej|~B)':>10} {'share of rej that is B':>23}")
for name, B in defs.items():
    nb = B.mean()
    pr_b = rej[B].mean() if B.sum() else float('nan')
    pr_nb = rej[~B].mean() if (~B).sum() else float('nan')
    share = (rej & B).sum() / max(rej.sum(), 1)
    print(f'{name:24s} {nb:9.3f} {pr_b:9.3f} {pr_nb:10.3f} {share:23.3f}')

# "hard branch" = both teacher and draft are unsure
print('\n=== B. branch points defined by teacher AND draft uncertainty ===')
hard = (tmax < 0.9) & (dmax < 0.9)
print(f'base rate {hard.mean():.3f}  P(rej|hard) {rej[hard].mean():.3f}  '
      f'share of all rejects {((rej & hard).sum()/rej.sum()):.3f}')

print('\n=== C. run structure of branch points (tmax<0.9) ===')
B = defs['teacher_tmax<0.90']
same = 0; tot = 0
for s in np.unique(seq):
    m = seq == s
    b = B[m]
    if len(b) < 2:
        continue
    same += int((b[1:] & b[:-1]).sum()); tot += int(b[:-1].sum())
print(f'P(branch[t] | branch[t-1]) = {same/max(tot,1):.3f}   (base rate {B.mean():.3f}, lift {same/max(tot,1)/max(B.mean(),1e-9):.2f}x)')
runs = []
for s in np.unique(seq):
    b = B[seq == s]
    r = 0
    for x in b:
        if x:
            r += 1
        elif r:
            runs.append(r); r = 0
    if r:
        runs.append(r)
print(f'branch runs: n={len(runs)} mean_len={np.mean(runs):.2f} p50={np.median(runs):.0f} '
      f'p90={np.quantile(runs,0.9):.0f} max={max(runs)}')

print('\n=== D. anticipation: predict branch[t] from data before t ===')
# build lagged features within each sequence
feats = {}
def lag(x, k, fill):
    y = np.full_like(x, fill, dtype=np.float64)
    y[k:] = x[:-k]
    # mask sequence starts
    for s in np.unique(seq):
        m = np.where(seq == s)[0]
        y[m[:k]] = fill
    return y
feats['tent_lag1'] = lag(tent, 1, np.nan)
feats['tent_lag2'] = lag(tent, 2, np.nan)
feats['tent_lag3'] = lag(tent, 3, np.nan)
feats['tmax_lag1_inv'] = 1 - lag(tmax, 1, np.nan)
feats['dent_lag1'] = lag(dent, 1, np.nan)
feats['dmax_lag1_inv'] = 1 - lag(dmax, 1, np.nan)
feats['rej_lag1'] = lag(rej.astype(float), 1, np.nan)
# rolling stats over last 4
roll = np.full(N, np.nan); rollstd = np.full(N, np.nan)
for s in np.unique(seq):
    idx = np.where(seq == s)[0]
    x = tent[idx]
    for j in range(1, len(idx)):
        w = x[max(0, j - 4):j]
        roll[idx[j]] = w.mean(); rollstd[idx[j]] = w.std()
feats['tent_roll4_mean'] = roll
feats['tent_roll4_std'] = rollstd
# token-class features of the previous token
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/path/to/models/Qwen3.6-35B-A3B')
prev_raw = d['prev']
prev = np.full(N, -1, dtype=np.int64)
_off = 0
for _s in np.unique(seq):
    _idx = np.where(seq == _s)[0]
    _n = len(_idx)
    prev[_idx[:-1]] = prev_raw[_off:_off + _n - 1]
    _off += _n - 1
_pv = prev.copy(); _pv[_pv < 0] = 0
strs = tok.convert_ids_to_tokens(_pv.tolist())
for _i in np.where(prev < 0)[0]:
    strs[_i] = None
def cls(s):
    if s is None: return [0]*6
    return [int('\n' in s), int(s.strip().isdigit()), int(any(c in s for c in '=+-*/^')),
            int('\\' in s), int(s.startswith('Ġ') and len(s.strip()) > 0), int(len(s.strip()) == 0)]
C = np.array([cls(s) for s in strs], dtype=float)
names = ['is_newline', 'is_digit', 'is_operator', 'is_latex', 'is_word_start', 'is_empty']
for i, n in enumerate(names):
    feats[n] = C[:, i]
pos = np.zeros(N)
for s in np.unique(seq):
    idx = np.where(seq == s)[0]
    pos[idx] = np.arange(len(idx))
feats['pos'] = pos

y = defs['teacher_tmax<0.90'].astype(int)
ok = ~np.isnan(feats['tent_lag1'])
print(f'({ok.sum()} tokens with valid lag)')
for k, v in feats.items():
    vv = v[ok]
    if np.isnan(vv).any():
        vv = np.nan_to_num(vv)
    print(f'  AUC({k}) = {roc_auc_score(y[ok], vv):.3f}')

X = np.column_stack([np.nan_to_num(feats[k][ok]) for k in feats])
ys = y[ok]
sq = seq[ok]
tr = np.isin(sq, np.unique(sq)[: int(0.8 * len(np.unique(sq)))])
sc = StandardScaler().fit(X[tr])
clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), ys[tr])
auc = roc_auc_score(ys[~tr], clf.predict_proba(sc.transform(X[~tr]))[:, 1])
print(f'  LOGREG on all {X.shape[1]} lag features: held-out AUC = {auc:.3f}')
# with only lag-1 features
keys = ['tent_lag1', 'tmax_lag1_inv', 'dent_lag1', 'dmax_lag1_inv', 'rej_lag1',
        'is_newline', 'is_digit', 'is_operator', 'is_latex']
X2 = np.column_stack([np.nan_to_num(feats[k][ok]) for k in keys])
sc2 = StandardScaler().fit(X2[tr])
clf2 = LogisticRegression(max_iter=2000).fit(sc2.transform(X2[tr]), ys[tr])
print(f'  LOGREG lag-1 only: held-out AUC = {roc_auc_score(ys[~tr], clf2.predict_proba(sc2.transform(X2[~tr]))[:, 1]):.3f}')

print('\n=== E. P(branch) by previous token (top 15 frequent) ===')
import collections
cnt = collections.Counter(strs)
tab = []
for s, c in cnt.most_common(200):
    if c < 150:
        continue
    sel = np.array([x == s for x in strs])
    tab.append((s.replace('Ġ', ' '), c, B[sel].mean(), rej[sel].mean()))
for t in tab[:15]:
    print(f'  prev={t[0]!r:16s} n={t[1]:6d}  P(branch)={t[2]:.3f}  P(reject)={t[3]:.3f}')
