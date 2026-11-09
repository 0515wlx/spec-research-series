"""Is a draft-token rejection explained by the teacher routing to cold / non-merged experts?

For every generated token we compute routing features from the teacher's 40 routers and join them
with the dense draft's per-token match/mismatch, then measure mismatch rate vs routing coldness.
"""
import sys, os, json, argparse, time
import numpy as np
import torch
sys.path.insert(0, 'code')
from draft import load_dense
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

T = '/path/to/models/Qwen3.6-35B-A3B'
D = '/path/to/spec-research/data'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=80)
    ap.add_argument('--out', default='out/routing_analysis.npz')
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(T)
    draft, _ = load_dense('ckpt/dense_distill_s1', dtype=torch.bfloat16, device='cuda:1')
    draft.eval()
    teacher = AutoModelForCausalLM.from_pretrained(T, dtype=torch.bfloat16, device_map={'': 0},
                                                   low_cpu_mem_usage=True).eval()
    nL = len(teacher.model.layers)
    st = np.load(f'{D}/calib_stats.npz')
    usage = st['usage']                     # (nL, 256)
    top8 = st['top8']                       # (nL, 8)
    merged_mask = np.zeros((nL, 256), dtype=bool)
    for i in range(nL):
        merged_mask[i, top8[i]] = True
    q25 = np.quantile(usage, 0.25, axis=1, keepdims=True)
    q50 = np.quantile(usage, 0.50, axis=1, keepdims=True)
    cold25 = usage <= q25
    cold50 = usage <= q50

    caps = {i: {} for i in range(nL)}

    def mk(i):
        def hook(mod, inp, out):
            _, w, idx = out
            caps[i]['idx'] = idx.detach()
            caps[i]['w'] = w.detach().float()
        return hook
    hs = [teacher.model.layers[i].mlp.gate.register_forward_hook(mk(i)) for i in range(nL)]

    recs = [json.loads(l) for l in open(f'{D}/teacher_math500_nothink.jsonl')][:a.n]
    feats = []
    t0 = time.time()
    with torch.no_grad():
        for ri, r in enumerate(recs):
            inp, out = r['input_ids'], r['output_ids']
            while out and out[-1] == 248044:
                out = out[:-1]
            ids = torch.tensor(inp + out, dtype=torch.long)
            s = len(inp)
            # dense draft teacher-forced match
            lg = draft(input_ids=ids.cuda()[None].to('cuda:1'), use_cache=False).logits[0]
            pred = lg[s - 1: len(ids) - 1].argmax(-1).cpu()
            match = (pred == ids[s:]).numpy()
            # teacher forward
            to = teacher(input_ids=ids.cuda()[None], use_cache=False)
            bit = 0
            # note: hooks with generation cache; top-k idx are (T,8)
            idxs = torch.stack([caps[i]['idx'][s - 1: len(ids) - 1] for i in range(nL)])
            ws = torch.stack([caps[i]['w'][s - 1: len(ids) - 1] for i in range(nL)])
            mm = torch.from_numpy(merged_mask).to(idxs.device)
            c25 = torch.from_numpy(cold25).to(idxs.device)
            c50 = torch.from_numpy(cold50).to(idxs.device)
            Gi = idxs.shape[1]
            m8 = mm.gather(1, idxs.reshape(nL, -1)).reshape(nL, Gi, 8)        # in merged top8?
            w_out = (ws * (~m8)).sum(-1)                                       # (nL,G)
            k25 = c25.gather(1, idxs.reshape(nL, -1)).reshape(nL, Gi, 8)
            w_cold25 = (ws * k25).sum(-1)
            k50 = c50.gather(1, idxs.reshape(nL, -1)).reshape(nL, Gi, 8)
            w_cold50 = (ws * k50).sum(-1)
            ent = -(ws.clamp_min(1e-9).log() * ws).sum(-1)                     # entropy over top8
            maxw = ws.max(-1).values
            tl = to.logits[0, s - 1: len(ids) - 1].float()
            lp = torch.log_softmax(tl, -1)
            tent = -(lp.exp() * lp).sum(-1)
            tmax = lp.max(-1).values.exp()
            feats.append(dict(
                match=match,
                w_out=w_out.mean(0).cpu().numpy(),
                w_out_max=w_out.max(0).values.cpu().numpy(),
                w_cold25=w_cold25.mean(0).cpu().numpy(),
                w_cold50=w_cold50.mean(0).cpu().numpy(),
                ent=ent.mean(0).cpu().numpy(),
                maxw=maxw.mean(0).cpu().numpy(),
                t_ent=tent.cpu().numpy(),
                t_maxp=tmax.cpu().numpy(),
                n_out=(~m8).float().mean(dim=(0, 2)).cpu().numpy(),
                d_maxp=lg[s - 1: len(ids) - 1].float().log_softmax(-1).max(-1).values.exp().cpu().numpy(),
                d_ent=-(lg[s - 1: len(ids) - 1].float().log_softmax(-1).exp()
                        * lg[s - 1: len(ids) - 1].float().log_softmax(-1)).sum(-1).cpu().numpy(),
            ))
            if (ri + 1) % 10 == 0:
                print('seq', ri + 1, f'{time.time()-t0:.0f}s', flush=True)
    for h in hs:
        h.remove()
    keys = ['match', 'w_out', 'w_out_max', 'w_cold25', 'w_cold50', 'ent', 'maxw', 't_ent', 't_maxp',
            'n_out', 'd_maxp', 'd_ent']
    out = {k: np.concatenate([f[k] for f in feats]) for k in keys}
    np.savez(a.out, **out)
    print('saved', len(out['match']), 'tokens')

    # ---------------- analysis ----------------
    m = out['match'].astype(bool)
    print(f'\noverall match {m.mean():.4f}  mismatch {(~m).mean():.4f}  n={len(m)}')
    for k in ['w_out', 'w_cold25', 'w_cold50', 'ent', 't_ent', 'd_ent', 'd_maxp', 'n_out']:
        v = out[k]
        qs = np.quantile(v, np.linspace(0, 1, 11))
        qs = np.unique(qs)
        print(f'\n[{k}] mismatch rate by decile (low->high):')
        rows = []
        for i in range(len(qs) - 1):
            sel = (v >= qs[i]) & (v <= qs[i + 1] if i == len(qs) - 2 else v < qs[i + 1])
            if sel.sum() < 50:
                continue
            rows.append((round(float(qs[i]), 4), round(float(qs[i + 1]), 4), int(sel.sum()),
                         round(float((~m[sel]).mean()), 4)))
        for r in rows:
            print('   ', r)
        # AUC (mismatch vs feature)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(~m, v)
        print(f'    AUC(feature -> mismatch) = {auc:.3f}')
    pure = out['n_out'] < 1e-6
    print(f'\ntokens with zero non-merged routing: {pure.sum()} ({pure.mean():.1%}), '
          f'mismatch {1-m[pure].mean():.4f} vs rest {1-m[~pure].mean():.4f}')
    from sklearn.metrics import roc_auc_score
    print('\n--- does routing add anything beyond teacher uncertainty? ---')
    te = out['t_ent']
    qs = np.quantile(te, np.linspace(0, 1, 6))
    for i in range(5):
        sel = (te >= qs[i]) & (te < qs[i + 1] if i < 4 else te <= qs[i + 1])
        if (~m[sel]).sum() in (0, sel.sum()):
            continue
        auc_r = roc_auc_score(~m[sel], out['w_out'][sel])
        auc_d = roc_auc_score(~m[sel], out['d_maxp'][sel]) if (~m[sel]).sum() > 5 else float('nan')
        print(f'  teacher-entropy bin {i+1}: n={sel.sum():6d} mismatch={1-m[sel].mean():.3f} '
              f'AUC(w_out)={auc_r:.3f} AUC(draft_maxp)={auc_d:.3f}')
    print('\n--- usable-at-draft-time signal (draft confidence) ---')
    for thr in [0.5, 0.7, 0.8, 0.9, 0.95]:
        sel = out['d_maxp'] >= thr
        print(f'  draft maxp >= {thr}: coverage {sel.mean():.1%}, mismatch {1-m[sel].mean():.4f}')


if __name__ == '__main__':
    main()
