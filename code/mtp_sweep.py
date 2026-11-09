import sys, json, copy, torch, torch.nn as nn
sys.path.insert(0,'code')
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer, Qwen3_5MoeRMSNorm)
from transformers.masking_utils import create_causal_mask
from safetensors.torch import load_file
T='/path/to/models/Qwen3.6-35B-A3B'
tok=AutoTokenizer.from_pretrained(T)
cfg=AutoConfig.from_pretrained(T); tc=cfg.text_config
m=AutoModelForCausalLM.from_pretrained(T,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True).eval()
mcfg=copy.deepcopy(tc); mcfg.layer_types=['full_attention']; mcfg.num_hidden_layers=1
mcfg._attn_implementation='sdpa'; mcfg._experts_implementation='eager'
class MTP(nn.Module):
    def __init__(s,tc,mcfg):
        super().__init__(); H=tc.hidden_size
        s.fc=nn.Linear(2*H,H,bias=False)
        s.ne=Qwen3_5MoeRMSNorm(H,eps=tc.rms_norm_eps)
        s.nh=Qwen3_5MoeRMSNorm(H,eps=tc.rms_norm_eps)
        s.layer=Qwen3_5MoeDecoderLayer(mcfg,0)
        s.norm=Qwen3_5MoeRMSNorm(H,eps=tc.rms_norm_eps)
mtp=MTP(tc,mcfg).to(torch.bfloat16)
idx=json.load(open(f'{T}/model.safetensors.index.json'))['weight_map']
shards={}; sd={}
for k,f in need.items() if False else [(k,f) for k,f in idx.items() if k.startswith('mtp.')]:
    if f not in shards: shards[f]=load_file(f'{T}/{f}')
    sd[k]=shards[f][k]
remap={k[4:].replace('layers.0.','layer.'):v for k,v in sd.items()}
# pre_fc_norm_embedding -> ne, pre_fc_norm_hidden -> nh
remap['ne.weight']=remap.pop('pre_fc_norm_embedding.weight')
remap['nh.weight']=remap.pop('pre_fc_norm_hidden.weight')
print(mtp.load_state_dict(remap,strict=False))
mtp=mtp.to('cuda').eval()
cap={}
h=m.model.layers[-1].register_forward_hook(lambda mod,i,o: cap.__setitem__('h', o[0] if isinstance(o,tuple) else o))
recs=[json.loads(l) for l in open('data/teacher_math500_nothink.jsonl')][:6]
def prep(r):
    inp=r['input_ids']; out=r['output_ids']
    while out and out[-1]==248044: out=out[:-1]
    return torch.tensor(inp+out,device='cuda')[None]
seqs=[prep(r) for r in recs]
with torch.no_grad():
    hs=[]
    for ids in seqs:
        o=m(input_ids=ids,use_cache=False)
        hs.append((cap['h'].clone(), m.model.norm(cap['h']), o.logits))
def test(order, hsrc, shift, pos_off):
    # shift: target offset relative to h position: pred token index = i + shift
    acc=0;n=0
    with torch.no_grad():
        for (hp,hn,lg),ids in zip(hs,seqs):
            Hsrc = hp if hsrc=='pre' else hn
            L=ids.shape[1]
            # input token for each position i: ids[i+1] (emb of next token)
            e=m.model.embed_tokens(ids[:,1:])
            a=mtp.nh(Hsrc[:,:-1]); b=mtp.ne(e)
            x=torch.cat([b,a],-1) if order=='eh' else torch.cat([a,b],-1)
            inpM=mtp.fc(x)
            pos=torch.arange(1,L,device='cuda').view(1,1,-1).expand(4,1,-1)+pos_off
            pe=m.model.rotary_emb(inpM,pos[1:])
            am=create_causal_mask(config=mcfg,inputs_embeds=inpM,attention_mask=None,past_key_values=None,position_ids=pos[0])
            ho=mtp.layer(inpM,position_embeddings=pe,attention_mask=am,position_ids=pos[0],past_key_values=None,use_cache=False)
            lgs=m.lm_head(mtp.norm(ho)).float()
            for i in range(lgs.shape[1]):
                t=i+shift
                if t>=L: continue
                acc += int(lgs[0,i].argmax().item()==ids[0,t].item()); n+=1
    return acc/n
for order in ['eh','he']:
    for hsrc in ['pre','post']:
        for shift in [1,2]:
            for po in [0,-1]:
                print(order,hsrc,shift,po,round(test(order,hsrc,shift,po),4),flush=True)
