"""Branch-point accounting: what share of rejected tokens are branch points, and can a branch point
be anticipated from the tokens *before* it?

Saves per-token arrays (ids, teacher/draft confidence, match) for offline analysis.
"""
import sys, json, argparse
import numpy as np
import torch
sys.path.insert(0, 'code')
from draft import load_dense
from transformers import AutoTokenizer, AutoModelForCausalLM

T = '/path/to/models/Qwen3.6-35B-A3B'
D = '/path/to/spec-research/data'

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='ckpt/dense_rl')
ap.add_argument('--n', type=int, default=120)
ap.add_argument('--out', default='out/branch_tokens_full.npz')
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(T)
m, _ = load_dense(a.ckpt, dtype=torch.bfloat16, device='cuda:1')
m.eval()
teacher = AutoModelForCausalLM.from_pretrained(T, dtype=torch.bfloat16, device_map={'': 0},
                                               low_cpu_mem_usage=True).eval()
recs = [json.loads(l) for l in open(f'{D}/teacher_math500_nothink.jsonl')][:a.n]
rows = []
with torch.no_grad():
    for ri, r in enumerate(recs):
        inp, out = r['input_ids'], r['output_ids']
        while out and out[-1] == 248044:
            out = out[:-1]
        ids = torch.tensor(inp + out, dtype=torch.long)[None]
        L = ids.shape[1]
        s = len(inp)
        lg = m(input_ids=ids[:, :].to('cuda:1'), use_cache=False).logits[0, s - 1: L - 1].float()
        tg = teacher(input_ids=ids.cuda(), use_cache=False).logits[0, s - 1: L - 1].float()
        tgt = ids[0, s:].to(lg.device)
        lp = torch.log_softmax(lg, -1)
        p = lp.exp()
        top2 = p.topk(2, dim=-1).values
        tlp = torch.log_softmax(tg, -1)
        tp = tlp.exp()
        ttop2 = tp.topk(2, dim=-1).values
        match = (lg.argmax(-1) == tgt)
        nt = tgt.numel()
        prev = ids[0, s - 1: L - 2]
        rows.append(dict(t=tgt.cpu().numpy().astype(np.int32), prev=prev.cpu().numpy().astype(np.int32),
                         tmax=ttop2[:, 0].cpu().numpy().astype(np.float32),
                         margin=(ttop2[:, 0] - ttop2[:, 1]).cpu().numpy().astype(np.float32),
                         tent=-(tp * tlp).sum(-1).cpu().numpy().astype(np.float32),
                         dmax=top2[:, 0].cpu().numpy().astype(np.float32),
                         dent=-(p * lp).sum(-1).cpu().numpy().astype(np.float32),
                         match=match.cpu().numpy().astype(np.uint8),
                         seq=np.full(nt, ri, dtype=np.int32)))
        if (ri + 1) % 20 == 0:
            print('seq', ri + 1, flush=True)
out = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
np.savez(a.out, **out)
print('tokens', len(out['match']), 'match', out['match'].mean())
