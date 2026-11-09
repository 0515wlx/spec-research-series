import sys,json,copy,torch; sys.path.insert(0,'code')
from mtp_eval import build_mtp
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
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
with torch.no_grad():
    th(input_ids=ids.cuda()); Hn=th.model.norm(cap['h'])[0]
    e=th.model.embed_tokens(ids[0,1:].cuda())
    inpM=mtp.fc(torch.cat([mtp.ne(e),mtp.nh(Hn[:-1])],-1))
    # (b) full recompute each step with explicit mask, for p in s..s+120
    cache=DynamicCache(config=mcfg)
    ca=[];sa=[]
    for p in range(s, s+120):
        i=p-2
        # cached step
        e1=th.model.embed_tokens(ids[0,p-1].cuda()[None,None])
        x1=mtp.fc(torch.cat([mtp.ne(e1),mtp.nh(Hn[i][None,None])],-1))
        p4=torch.full((4,1,1),p-1,device='cuda',dtype=torch.long)
        pe1=th.model.rotary_emb(x1,p4[1:])
        h1=mtp.layer(x1,position_embeddings=pe1,attention_mask=None,position_ids=p4[0],past_key_values=cache,use_cache=True)
        ca.append(int(th.lm_head(mtp.norm(h1))[0,-1].argmax()))
        # full recompute of MTP inputs 0..i with explicit causal mask
        sub=inpM[:i+1][None]
        pos=torch.arange(1,i+2,device='cuda').view(1,1,-1).expand(4,1,-1)
        pe=th.model.rotary_emb(sub,pos[1:])
        m=torch.full((1,1,i+1,i+1),-1e4,device='cuda',dtype=torch.bfloat16)
        m=torch.triu(m,1)
        hh=mtp.layer(sub,position_embeddings=pe,attention_mask=m,position_ids=pos[0],past_key_values=None,use_cache=False)
        sa.append(int(th.lm_head(mtp.norm(hh))[0,-1].argmax()))
    tgt=ids[0,s:s+120]
    print('cached acc', sum(a==int(b) for a,b in zip(ca,tgt))/120)
    print('full-recompute acc', sum(a==int(b) for a,b in zip(sa,tgt))/120)
    print('agreement cached vs full', sum(a==b for a,b in zip(ca,sa))/120)
    # block-reset variant: cache reset at start of each 6-token block
    reset=[]
    blk=0
    while blk<120:
        c=DynamicCache(config=mcfg)
        for k in range(6):
            p=s+blk+k
            if p>=s+120: break
            i=p-2
            e1=th.model.embed_tokens(ids[0,p-1].cuda()[None,None])
            x1=mtp.fc(torch.cat([mtp.ne(e1),mtp.nh(Hn[i][None,None])],-1))
            p4=torch.full((4,1,1),p-1,device='cuda',dtype=torch.long)
            pe1=th.model.rotary_emb(x1,p4[1:])
            h1=mtp.layer(x1,position_embeddings=pe1,attention_mask=None,position_ids=p4[0],past_key_values=c,use_cache=True)
            reset.append(int(th.lm_head(mtp.norm(h1))[0,-1].argmax()))
        blk+=6
    print('block-reset acc', sum(a==int(b) for a,b in zip(reset,tgt))/120)
hk.remove()
