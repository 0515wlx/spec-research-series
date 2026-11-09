import sys,json,copy,torch; sys.path.insert(0,'code')
from mtp_eval import build_mtp
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
T='/path/to/models/Qwen3.6-35B-A3B'
tok=AutoTokenizer.from_pretrained(T); cfg=AutoConfig.from_pretrained(T); tc=cfg.text_config
th=AutoModelForCausalLM.from_pretrained(T,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True).eval()
mcfg=copy.deepcopy(tc); mcfg.layer_types=['full_attention']; mcfg.num_hidden_layers=1
mcfg._attn_implementation='sdpa'; mcfg._experts_implementation='eager'
mtp=build_mtp(tc,mcfg).to('cuda').eval()
cap={}
hk=th.model.layers[-1].register_forward_hook(lambda m,i,o: cap.__setitem__('h', o[0] if isinstance(o,tuple) else o))
r=json.loads(open('data/teacher_math500_nothink.jsonl').readline())
inp,out=r['input_ids'],r['output_ids']
while out and out[-1]==248044: out=out[:-1]
ids=torch.tensor(inp+out)[None]; L=ids.shape[1]; s=len(inp)

def mtp_logits(ids_):
    th(input_ids=ids_.cuda())
    Hn=th.model.norm(cap['h'])[0]
    e=th.model.embed_tokens(ids_[0,1:].cuda())
    inpM=mtp.fc(torch.cat([mtp.ne(e),mtp.nh(Hn[:-1])],-1))
    pos=torch.arange(1,ids_.shape[1],device='cuda').view(1,1,-1).expand(4,1,-1)
    pe=th.model.rotary_emb(inpM[None],pos[1:])
    h=mtp.layer(inpM[None],position_embeddings=pe,attention_mask=None,position_ids=pos[0],past_key_values=None,use_cache=False)
    return th.lm_head(mtp.norm(h[0])).float()

with torch.no_grad():
    lg1=mtp_logits(ids)
    # append garbage to the END: if no leakage, logits for earlier positions must be unchanged
    ids2=torch.cat([ids, torch.randint(100,2000,(1,50))],1)
    lg2=mtp_logits(ids2)
    d=(lg1[:L-1]-lg2[:L-1]).abs().max().item()
    print('max |logit change| when appending future tokens:', d)
    # correct single-position check: predict ids[p] from h[p-2], emb(ids[p-1])
    Hn=th.model.norm(cap['h'])[0]
    ok=n=0
    for p in range(s, min(s+80, L)):
        e1=th.model.embed_tokens(ids[0,p-1].cuda()[None,None])
        x1=mtp.fc(torch.cat([mtp.ne(e1),mtp.nh(Hn[p-2][None,None])],-1))
        p4=torch.full((4,1,1),p-1,device='cuda',dtype=torch.long)
        pe1=th.model.rotary_emb(x1,p4[1:])
        h1=mtp.layer(x1,position_embeddings=pe1,attention_mask=None,position_ids=p4[0],past_key_values=None,use_cache=False)
        pr=th.lm_head(mtp.norm(h1))[0,-1].argmax().item()
        ok+=int(pr==ids[0,p].item()); n+=1
    print(f'single-position (no context, correct align) acc {ok/n:.4f} on {n}')
    k=s+80
    tb=(lg1[s-2:k-2].argmax(-1)==ids[0,s:k].cuda()).float().mean().item()
    print(f'batched acc over same span {tb:.4f}')
