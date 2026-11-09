"""Composite drafting: dense draft + native MTP head per token, with a learned gate.

Collects per-token predictions/confidences of both drafters on cached teacher sequences, measures the
oracle union, then trains a gate (using only features available at draft time) to pick between them.
"""
import sys, json, copy, argparse
import numpy as np
import torch
sys.path.insert(0, 'code')
from draft import load_dense
from mtp_eval import build_mtp
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

T = '/path/to/models/Qwen3.6-35B-A3B'
D = '/path/to/spec-research/data'

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=100)
ap.add_argument('--out', default='out/composite.npz')
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(T)
cfg = AutoConfig.from_pretrained(T); tc = cfg.text_config
draft, _ = load_dense('ckpt/dense_rl', dtype=torch.bfloat16, device='cuda:1')
draft.eval()
teacher = AutoModelForCausalLM.from_pretrained(T, dtype=torch.bfloat16, device_map={'': 0},
                                               low_cpu_mem_usage=True).eval()
mcfg = copy.deepcopy(tc); mcfg.layer_types = ['full_attention']; mcfg.num_hidden_layers = 1
mcfg._attn_implementation = 'sdpa'; mcfg._experts_implementation = 'eager'
mtp = build_mtp(tc, mcfg).to('cuda').eval()
cap = {}
hk = teacher.model.layers[-1].register_forward_hook(
    lambda mod, i, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))

recs = [json.loads(l) for l in open(f'{D}/teacher_math500_nothink.jsonl')][:a.n]
rows = []
with torch.no_grad():
    for ri, r in enumerate(recs):
        inp, out = r['input_ids'], r['output_ids']
        while out and out[-1] == 248044:
            out = out[:-1]
        ids = torch.tensor(inp + out, dtype=torch.long)[None]
        L = ids.shape[1]; s = len(inp)
        ld = draft(input_ids=ids.to('cuda:1'), use_cache=False).logits[0, s - 1: L - 1].float()
        lt = teacher(input_ids=ids.cuda(), use_cache=False).logits[0, s - 1: L - 1].float()
        Hn = teacher.model.norm(cap['h'])[0]                       # (L,H)
        # MTP: token-by-token with a persistent KV cache (causal by construction, training-consistent)
        from transformers import DynamicCache
        cache = DynamicCache(config=mcfg)
        m_preds = []
        m_confs = []
        for p in range(s, L):
            e1 = teacher.model.embed_tokens(ids[0, p - 1].to('cuda')[None, None])
            x1 = mtp.fc(torch.cat([mtp.ne(e1), mtp.nh(Hn[p - 2][None, None])], -1))
            p4 = torch.full((4, 1, 1), p - 1, device='cuda', dtype=torch.long)
            pe1 = teacher.model.rotary_emb(x1, p4[1:])
            h1 = mtp.layer(x1, position_embeddings=pe1, attention_mask=None, position_ids=p4[0],
                           past_key_values=cache, use_cache=True)
            l1 = teacher.lm_head(mtp.norm(h1))[0, -1].float()
            pr = l1.softmax(-1)
            m_preds.append(int(pr.argmax().item()))
            m_confs.append(float(pr.max().item()))
        mtp_pred = torch.tensor(m_preds, dtype=torch.int32)
        mtp_conf = torch.tensor(m_confs, dtype=torch.float32)
        tgt = ids[0, s:].cuda()
        pd = ld.softmax(-1).max(-1)
        tl = lt.log_softmax(-1)
        tent = -(tl.exp() * tl).sum(-1)
        rows.append(dict(
            tgt=tgt.cpu().numpy().astype(np.int32),
            d_pred=ld.argmax(-1).cpu().numpy().astype(np.int32), d_conf=pd.values.cpu().numpy().astype(np.float32),
            m_pred=mtp_pred.numpy(), m_conf=mtp_conf.numpy(),
            t_ent=tent.cpu().numpy().astype(np.float32),
            t_max=tl.exp().max(-1).values.cpu().numpy().astype(np.float32),
            prev=ids[0, s - 1: L - 1].cpu().numpy().astype(np.int32),
            seq=np.full(len(tgt), ri, dtype=np.int32)))
        if (ri + 1) % 20 == 0:
            print('seq', ri + 1, flush=True)
hk.remove()
out = {}
for k in rows[0]:
    try:
        out[k] = np.concatenate([r[k] for r in rows])
    except Exception as e:
        print('skip', k, e)
np.savez(a.out, **out)
print('tokens', len(out['t_ent']))
