"""Top-k recall of the teacher's token under the draft distribution -> ceiling for tree/budgeted drafting."""
import sys, json, argparse
import numpy as np
import torch
sys.path.insert(0, 'code')
from draft import load_dense
from transformers import AutoTokenizer

T = '/path/to/models/Qwen3.6-35B-A3B'
D = '/path/to/spec-research/data'

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='ckpt/dense_rl')
ap.add_argument('--n', type=int, default=100)
ap.add_argument('--ks', default='1,2,4,8,16,32')
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(T)
m, _ = load_dense(a.ckpt, dtype=torch.bfloat16, device='cuda:0')
m.eval()
recs = [json.loads(l) for l in open(f'{D}/teacher_math500_nothink.jsonl')][:a.n]
ks = [int(x) for x in a.ks.split(',')]
hits = {k: [] for k in ks}
with torch.no_grad():
    for r in recs:
        inp, out = r['input_ids'], r['output_ids']
        while out and out[-1] == 248044:
            out = out[:-1]
        ids = torch.tensor(inp + out, dtype=torch.long).cuda()[None]
        lg = m(input_ids=ids, use_cache=False).logits[0]
        s = len(inp)
        L = ids.shape[1]
        pred = lg[s - 1: L - 1].float()
        tgt = ids[0, s:]
        for k in ks:
            topk = pred.topk(k, dim=-1).indices
            hits[k].append((topk == tgt[:, None]).any(-1).float().cpu().numpy())
out = {}
for k in ks:
    h = np.concatenate(hits[k])
    out[k] = {'recall': float(h.mean())}
    # expected acceptance length if the draft could pick any of the top-k at each step
    print(f'k={k:2d} recall={h.mean():.4f}  ' +
          '  '.join(f'tau_ideal(g={g})={sum(np.prod([h[i] for i in range(min(g, 8))]) for _ in [0]) if False else 0:.2f}' for g in [5]))
# expected tau with a perfect top-k tree (independent approximation)
for k in ks:
    h = np.concatenate(hits[k])
    r = h.mean()
    for g in [3, 5, 8]:
        tau = 1 + sum(r ** j for j in range(1, g + 1))
        print(f'k={k:2d} g={g}: ideal tau={tau:.2f}')
json.dump(out, open('out/topk_recall.json', 'w'), indent=1)
