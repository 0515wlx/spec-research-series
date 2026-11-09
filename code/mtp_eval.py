"""Acceptance of the model's native MTP head (EAGLE-style 1-layer draft) on cached teacher sequences."""
import sys, json, copy, time, torch, torch.nn as nn
sys.path.insert(0, 'code')
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer, Qwen3_5MoeRMSNorm)
from transformers.masking_utils import create_causal_mask
from safetensors.torch import load_file

T = '/path/to/models/Qwen3.6-35B-A3B'


class MTP(nn.Module):
    def __init__(s, tc, mcfg):
        super().__init__()
        H = tc.hidden_size
        s.fc = nn.Linear(2 * H, H, bias=False)
        s.ne = Qwen3_5MoeRMSNorm(H, eps=tc.rms_norm_eps)
        s.nh = Qwen3_5MoeRMSNorm(H, eps=tc.rms_norm_eps)
        s.layer = Qwen3_5MoeDecoderLayer(mcfg, 0)
        s.norm = Qwen3_5MoeRMSNorm(H, eps=tc.rms_norm_eps)


def build_mtp(tc, mcfg):
    mtp = MTP(tc, mcfg).to(torch.bfloat16)
    idx = json.load(open(f'{T}/model.safetensors.index.json'))['weight_map']
    shards = {}
    sd = {}
    for k, f in idx.items():
        if k.startswith('mtp.'):
            if f not in shards:
                shards[f] = load_file(f'{T}/{f}')
            sd[k] = shards[f][k]
    remap = {k[4:].replace('layers.0.', 'layer.'): v for k, v in sd.items()}
    remap['ne.weight'] = remap.pop('pre_fc_norm_embedding.weight')
    remap['nh.weight'] = remap.pop('pre_fc_norm_hidden.weight')
    mtp.load_state_dict(remap, strict=False)
    return mtp


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--gammas', default='1,3,5,8')
    ap.add_argument('--out', default='out/mtp_accept.json')
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(T)
    cfg = AutoConfig.from_pretrained(T)
    tc = cfg.text_config
    m = AutoModelForCausalLM.from_pretrained(T, dtype=torch.bfloat16, device_map={'': 0},
                                             low_cpu_mem_usage=True).eval()
    mcfg = copy.deepcopy(tc)
    mcfg.layer_types = ['full_attention']
    mcfg.num_hidden_layers = 1
    mcfg._attn_implementation = 'sdpa'
    mcfg._experts_implementation = 'eager'
    mtp = build_mtp(tc, mcfg).to('cuda').eval()
    cap = {}
    hk = m.model.layers[-1].register_forward_hook(
        lambda mod, i, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))

    recs = [json.loads(l) for l in open('data/teacher_math500_nothink.jsonl')][:a.n]
    gammas = [int(x) for x in a.gammas.split(',')]
    from transformers import DynamicCache
    ccfg = copy.deepcopy(mcfg)

    stats = {g: {'acc': 0, 'prop': 0, 'lens': []} for g in gammas}
    t0 = time.time()
    with torch.no_grad():
        for ri, r in enumerate(recs):
            inp, out = r['input_ids'], r['output_ids']
            while out and out[-1] == 248044:
                out = out[:-1]
            ids = torch.tensor(inp + out, device='cuda')[None]
            _ = m(input_ids=ids, use_cache=False)
            Hn = m.model.norm(cap['h'])[0]     # post-norm hidden (L,H)
            seq = ids[0]
            n_prompt = len(inp)
            N = len(out)
            for g in gammas:
                t = n_prompt - 1               # position of the last prompt token = first hidden used
                while t < len(seq) - 1:
                    cache = DynamicCache(config=ccfg)
                    hcur = Hn[t:t + 1].view(1, 1, -1)
                    xnext = seq[t + 1:t + 2].view(1, 1)
                    L = 0
                    for k in range(g):
                        pos = t + k + 1
                        e = m.model.embed_tokens(xnext)
                        inpM = mtp.fc(torch.cat([mtp.ne(e), mtp.nh(hcur)], -1))
                        p4 = torch.full((4, 1, 1), pos, device='cuda', dtype=torch.long)
                        pe = m.model.rotary_emb(inpM, p4[1:])
                        hout = mtp.layer(inpM, position_embeddings=pe, attention_mask=None,
                                         position_ids=p4[0], past_key_values=cache, use_cache=True)
                        pred = m.lm_head(mtp.norm(hout))[0, -1].argmax()
                        tgt = seq[t + k + 2] if t + k + 2 < len(seq) else None
                        if tgt is not None and pred.item() == tgt.item():
                            L += 1
                            hcur = hout
                            xnext = pred.view(1, 1)
                        else:
                            break
                    stats[g]['acc'] += L
                    stats[g]['prop'] += g
                    stats[g]['lens'].append(L + 1)
                    t += L + 1
            if (ri + 1) % 10 == 0:
                print('seq', ri + 1, f'{time.time()-t0:.0f}s', flush=True)
    hk.remove()
    out = {'n_seq': len(recs), 'per_gamma': {}}
    for g in gammas:
        d = stats[g]
        import numpy as np
        out['per_gamma'][g] = {'alpha': d['acc'] / max(d['prop'], 1),
                               'tau': float(np.mean(d['lens']))}
    print(json.dumps(out, indent=1))
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
