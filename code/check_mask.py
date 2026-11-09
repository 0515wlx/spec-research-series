import sys,json,copy,torch,numpy as np; sys.path.insert(0,'code')
from draft import load_dense
from mtp_eval import build_mtp
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask
T='/path/to/models/Qwen3.6-35B-A3B'
tok=AutoTokenizer.from_pretrained(T); cfg=AutoConfig.from_pretrained(T); tc=cfg.text_config
th=AutoModelForCausalLM.from_pretrained(T,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True).eval()
mcfg=copy.deepcopy(tc); mcfg.layer_types=['full_attention']; mcfg.num_hidden_layers=1
mcfg._attn_implementation='sdpa'; mcfg._experts_implementation='eager'
mtp=build_mtp(tc,mcfg).to('cuda').eval()
cap={}
hk=th.model.layers[-1].register_forward_hook(lambda m,i,o: cap.__setitem__('h', o[0] if isinstance(o,tuple) else o))
recs=[json.loads(l) for l in open('data/teacher_math500_nothink.jsonl')][:2]
for r in recs:
    inp,out=r['input_ids'],r['output_ids']
    while out and out[-1]==248044: out=out[:-1]
    ids=torch.tensor(inp+out)[None]; L=ids.shape[1]; s=len(inp)
    with torch.no_grad():
        th(input_ids=ids.cuda())
        Hn=th.model.norm(cap['h'])[0]
        e=th.model.embed_tokens(ids[0,1:].cuda())
        inpM=mtp.fc(torch.cat([mtp.ne(e),mtp.nh(Hn[:-1])],-1))
        pos=torch.arange(1,L,device='cuda').view(1,1,-1).expand(4,1,-1)
        pe=th.model.rotary_emb(inpM[None],pos[1:])
        am=create_causal_mask(config=mcfg,inputs_embeds=inpM[None],attention_mask=None,past_key_values=None,position_ids=pos[0])
        print('mask',None if am is None else (am.shape, am.dtype, float(am[0,0,0,1]), float(am[0,0,1,0])))
        h1=mtp.layer(inpM[None],position_embeddings=pe,attention_mask=am,position_ids=pos[0],past_key_values=None,use_cache=False)
        lg1=th.lm_head(mtp.norm(h1[0])).float()
        # explicit additive mask
        Lm=inpM.shape[0]
        add=torch.full((1,1,Lm,Lm),-1e4,device='cuda',dtype=torch.bfloat16)
        add=torch.triu(add,diagonal=1)
        h2=mtp.layer(inpM[None],position_embeddings=pe,attention_mask=add,position_ids=pos[0],past_key_values=None,use_cache=False)
        lg2=th.lm_head(mtp.norm(h2[0])).float()
        tgt=ids[0,2:].cuda()
        a1=(lg1[s-2:L-2].argmax(-1)==tgt[s-2:]).float().mean().item()
        a2=(lg2[s-2:L-2].argmax(-1)==tgt[s-2:]).float().mean().item()
        # no mask at all
        h3=mtp.layer(inpM[None],position_embeddings=pe,attention_mask=None,position_ids=pos[0],past_key_values=None,use_cache=False)
        lg3=th.lm_head(mtp.norm(h3[0])).float()
        a3=(lg3[s-2:L-2].argmax(-1)==tgt[s-2:]).float().mean().item()
        # single-position loop (mtp_eval style) on 60 generated positions
        ok=0;n=0
        for p in range(s, min(s+60, L-1)):
            inp1=mtp.fc(torch.cat([mtp.ne(th.model.embed_tokens(ids[0,p].cuda()[None,None])),mtp.nh(Hn[p-2][None,None])],-1))
            p4=torch.full((4,1,1),p-1,device='cuda',dtype=torch.long)
            pe1=th.model.rotary_emb(inp1,p4[1:])
            hh=mtp.layer(inp1,position_embeddings=pe1,attention_mask=None,position_ids=p4[0],past_key_values=None,use_cache=False)
            pr=th.lm_head(mtp.norm(hh))[0,-1].argmax().item()
            ok+=int(pr==ids[0,p+1].item()); n+=1
        print(f'batched(create_causal) {a1:.4f}  batched(explicit triu) {a2:.4f}  batched(no mask) {a3:.4f}  single-pos {ok/n:.4f}')
hk.remove()
