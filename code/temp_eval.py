"""Does temperature fix 'confident but wrong'? Greedy: argmax invariant. Sampling: acceptance A(T)."""
import sys,json,torch,numpy as np; sys.path.insert(0,'code')
from draft import load_dense
from transformers import AutoTokenizer, AutoModelForCausalLM
T='/path/to/models/Qwen3.6-35B-A3B'
tok=AutoTokenizer.from_pretrained(T)
draft,_=load_dense('ckpt/dense_rl',dtype=torch.bfloat16,device='cuda:1'); draft.eval()
th=AutoModelForCausalLM.from_pretrained(T,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True).eval()
recs=[json.loads(l) for l in open('data/teacher_math500_nothink.jsonl')][:8]
Tds=[0.05,0.1,0.2,0.3,0.5,0.75,1.0]
res={t:[] for t in Tds}; same_t={t:[] for t in [1.0,1.25,1.5,2.0]}
greedy_ok=[]; conf_d=[]; conf_t=[]
with torch.no_grad():
    for r in recs:
        inp,out=r['input_ids'],r['output_ids']
        while out and out[-1]==248044: out=out[:-1]
        ids=torch.tensor(inp+out)[None]; L=ids.shape[1]; s=len(inp)
        ld=draft(input_ids=ids.to('cuda:1'),use_cache=False).logits[0,s-1:L-1].float().to('cuda:0')
        lt=th(input_ids=ids.cuda(),use_cache=False).logits[0,s-1:L-1].float()
        tgt=ids[0,s:].cuda()
        pt=lt.softmax(-1)
        greedy_ok.append((ld.argmax(-1)==tgt).float().mean().item())
        conf_d.append(ld.softmax(-1).max(-1).values.mean().item())
        conf_t.append(pt.max(-1).values.mean().item())
        for Td in Tds:
            q=(ld/Td).softmax(-1)
            res[Td].append(torch.minimum(pt,q).sum(-1).mean().item())
            if Td in same_t:
                q2=(ld/Td).softmax(-1); p2=(lt/Td).softmax(-1)
                same_t[Td].append(torch.minimum(p2,q2).sum(-1).mean().item())
print('greedy top-1 acc (T-invariant)', np.mean(greedy_ok))
print('mean draft maxp', np.mean(conf_d), 'mean teacher maxp', np.mean(conf_t))
print('\nT_draft sweep (teacher at T=1): acceptance prob per token')
for Td in Tds:
    print(f'  T={Td:4}: A={np.mean(res[Td]):.4f}')
print('\nboth at same T:')
for Td in same_t:
    print(f'  T={Td:4}: A={np.mean(same_t[Td]):.4f}')
# argmax invariance check
with torch.no_grad():
    for Td in [0.1,0.5,2.0,10.0]:
        am=(ld/Td).argmax(-1)==ld.argmax(-1)
        print('argmax invariant at T',Td,':',bool(am.all()))
