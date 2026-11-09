"""Branch-point predictor + composite (dense + MTP) gating analysis."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

d = np.load('/path/to/spec-research/out/composite2.npz')
tgt, dp, dc = d['tgt'], d['d_pred'], d['d_conf'].astype(float)
mp, mc = d['m_pred'], d['m_conf'].astype(float)
tent, tmax, seq, prev = d['t_ent'].astype(float), d['t_max'].astype(float), d['seq'], d['prev']
N = len(tgt)
md = dp == tgt
mm = mp == tgt
rej = ~md
print(f'tokens {N}  dense acc {md.mean():.4f}  mtp acc {mm.mean():.4f}')
print(f'union (at least one right) {(md|mm).mean():.4f}  both right {(md&mm).mean():.4f}  '
      f'both wrong {(~md&~mm).mean():.4f}')
print(f'disagreement rate {(dp!=mp).mean():.4f}')

# ---------- features available at draft time ----------
def lag(x, k, fill=np.nan):
    y = np.full(N, fill, dtype=float)
    y[k:] = x[:-k]
    for s in np.unique(seq):
        idx = np.where(seq == s)[0]
        y[idx[:k]] = fill
    return y
agree = (dp == mp).astype(float)
feats = {
    'd_conf': dc, 'm_conf': mc, 'conf_diff': dc - mc, 'agree': agree,
    'd_conf_lag1': lag(dc, 1), 'm_conf_lag1': lag(mc, 1),
    't_ent_lag1': lag(tent, 1), 't_ent_lag2': lag(tent, 2),
    'rej_lag1': lag(rej.astype(float), 1),
}
roll = np.full(N, np.nan)
for s in np.unique(seq):
    idx = np.where(seq == s)[0]
    x = tent[idx]
    for j in range(1, len(idx)):
        roll[idx[j]] = x[max(0, j - 4):j].mean()
feats['t_ent_roll4'] = roll
# prev token class
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/path/to/models/Qwen3.6-35B-A3B')
strs = tok.convert_ids_to_tokens(prev.tolist())
feats['is_newline'] = np.array([('\n' in s) for s in strs], float)
feats['is_word_start'] = np.array([s.startswith('Ġ') for s in strs], float)
feats['is_digit'] = np.array([s.strip().isdigit() for s in strs], float)
pos = np.zeros(N)
for s in np.unique(seq):
    idx = np.where(seq == s)[0]
    pos[idx] = np.arange(len(idx))
feats['pos'] = pos

keys = list(feats.keys())
X = np.column_stack([np.nan_to_num(feats[k]) for k in keys])
sq = np.unique(seq)
tr = np.isin(seq, sq[: int(0.8 * len(sq))])
te = ~tr

def fit_eval(y, keyset=None, model='lr'):
    ks = keyset or keys
    Xi = np.column_stack([np.nan_to_num(feats[k]) for k in ks])
    sc = StandardScaler().fit(Xi[tr])
    if model == 'lr':
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(Xi[tr]), y[tr])
        p = clf.predict_proba(sc.transform(Xi[te]))[:, 1]
    else:
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06)
        clf.fit(Xi[tr], y[tr])
        p = clf.predict_proba(Xi[te])[:, 1]
    return roc_auc_score(y[te], p), p

print('\n=== 1. branch-point predictor (target: teacher top1 < 0.9) ===')
y_branch = (tmax < 0.9).astype(int)
print(f'base rate {y_branch.mean():.3f}')
for name, ks in [('lag features only', ['d_conf_lag1', 'm_conf_lag1', 't_ent_lag1', 't_ent_lag2',
                                        'rej_lag1', 't_ent_roll4', 'is_newline', 'is_word_start',
                                        'is_digit', 'pos']),
                 ('current draft confidences', ['d_conf', 'm_conf', 'conf_diff', 'agree']),
                 ('everything', keys)]:
    auc_lr, _ = fit_eval(y_branch, ks, 'lr')
    auc_gb, _ = fit_eval(y_branch, ks, 'gb')
    print(f'  {name:28s} LR AUC {auc_lr:.3f}   GBDT AUC {auc_gb:.3f}')

print('\n=== 2. composite gating (only matters where they disagree) ===')
dis = dp != mp
print(f'disagreement positions {dis.sum()}  dense right there {md[dis].mean():.3f}  '
      f'mtp right there {mm[dis].mean():.3f}')
print(f'oracle pick on disagreement would give acc {((md|mm)[dis]).mean()*dis.mean()+ (~dis).mean()*md[~dis].mean():.4f}')
# simple rule: higher confidence wins
pick_m = mc > dc
pred = np.where(pick_m, mp, dp)
print(f'rule "higher confidence wins": acc {(pred==tgt).mean():.4f}')
# oracle
print(f'oracle (either right): {(md|mm).mean():.4f}')
# gate: train to choose mtp when it is right & dense wrong
y = (mm & ~md).astype(int)
for name, ks, mdl in [('conf only', ['d_conf', 'm_conf', 'conf_diff', 'agree'], 'lr'),
                      ('all feats', keys, 'lr'), ('all feats', keys, 'gb')]:
    auc, p = fit_eval(y, ks, mdl)
    print(f'  gate({name},{mdl}) AUC {auc:.3f}')
# threshold sweep with the all-feats GBDT gate -> realised accuracy
auc, p = fit_eval(y, keys, 'gb')
p_full = np.full(N, np.nan); p_full[te] = p
best = (0, 0)
for thr in np.linspace(0.05, 0.95, 19):
    pick = np.where(p_full > thr, mp, dp)
    acc = (pick[te] == tgt[te]).mean()
    if acc > best[0]:
        best = (acc, thr)
print(f'  best gate accuracy on held-out: {best[0]:.4f} (thr={best[1]:.2f}); '
      f'always-dense held-out {(md[te]).mean():.4f}, oracle {(md|mm)[te].mean():.4f}')
