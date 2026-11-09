import sys, json, copy, time, torch
import torch.nn as nn
sys.path.insert(0,'code')
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer, Qwen3_5MoeRMSNorm, Qwen3_5MoeTextRotaryEmbedding)
from safetensors.torch import load_file

T='/path/to/models/Qwen3.6-35B-A3B'
tok=AutoTokenizer.from_pretrained(T)
cfg=AutoConfig.from_pretrained(T); tc=cfg.text_config
m=AutoModelForCausalLM.from_pretrained(T,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True).eval()

# ---- build MTP head ----
mcfg=copy.deepcopy(tc); mcfg.layer_types=['full_attention']; mcfg.num_hidden_layers=1
mcfg._attn_implementation='sdpa'; mcfg._experts_implementation='eager'
from transformers.masking_utils import create_causal_mask
class MTP(nn.Module):
    def __init__(self, tc, mcfg):
        super().__init__()
        H=tc.hidden_size
        self.fc=nn.Linear(2*H,H,bias=False)
        self.pre_fc_norm_embedding=Qwen3_5MoeRMSNorm(H,eps=tc.rms_norm_eps)
        self.pre_fc_norm_hidden=Qwen3_5MoeRMSNorm(H,eps=tc.rms_norm_eps)
        self.layer=Qwen3_5MoeDecoderLayer(mcfg,0)
        self.norm=Qwen3_5MoeRMSNorm(H,eps=tc.rms_norm_eps)
mtp=MTP(tc,mcfg).to(torch.bfloat16)
idx=json.load(open(f'{T}/model.safetensors.index.json'))['weight_map']
need={k:v for k,v in idx.items() if k.startswith('mtp.')}
shards={}
sd={}
for k,f in need.items():
    if f not in shards: shards[f]=load_file(f'{T}/{f}')
    sd[k]=shards[f][k]
remap={}
for k,v in sd.items():
    nk=k[len('mtp.'):]
    nk=nk.replace('layers.0.','layer.')
    remap[nk]=v
missing,unexpected=mtp.load_state_dict(remap,strict=False)
print('missing',[x for x in missing][:6],'unexpected',[x for x in unexpected][:6])
mtp=mtp.to('cuda').eval()

# capture pre-norm hidden of last layer
cap={}
h=m.model.layers[-1].register_forward_hook(lambda mod,i,o: cap.__setitem__('h', o[0] if isinstance(o,tuple) else o))
recs=[json.loads(l) for l in open('data/teacher_math500_nothink.jsonl')][:20]
def run(r, use_norm):
    inp=r['input_ids']; out=r['output_ids']
    while out and out[-1]==248044: out=out[:-1]
    ids=torch.tensor(inp+out,device='cuda')[None]
    with torch.no_grad():
        o=m(input_ids=ids,use_cache=False)
        hlast=o.logits  # no
    hmid=cap['h']  # (1,L,H) pre-norm
    hn=o.logits.new_zeros(0)
    with torch.no_grad():
        # post-norm hidden: apply model norm
        hpost=m.model.norm(hmid)
        hsrc = hpost if use_norm else hmid
        L=ids.shape[1]
        # single-step MTP for all positions t: predict token t+2 from (h_t, x_{t+1})
        e=m.model.embed_tokens(ids[:,1:])
        inpM=mtp.fc(torch.cat([mtp.pre_fc_norm_hidden(hsrc[:,:-1]), mtp.pre_fc_norm_embedding(e)],-1))
        pos=torch.arange(1,L,device='cuda').view(1,1,-1).expand(4,1,-1)
        pe=m.model.rotary_emb(inpM,pos[1:])
        amask=create_causal_mask(config=mcfg,inputs_embeds=inpM,attention_mask=None,past_key_values=None,position_ids=pos[0])
        hout=mtp.layer(inpM,position_embeddings=pe,attention_mask=amask,position_ids=pos[0],past_key_values=None,use_cache=False)
        lg=torch.log_softmax(m.lm_head(mtp.norm(hout)).float(),-1)
        # position i predicts token i+2 ... compare to ids[i+2]
        tgt=ids[0,2:]
        lp=lg[0,:len(tgt)]
    return (lp.argmax(-1)==tgt).float().mean().item(), len(tgt)
for un in [True,False]:
    a,n=zip(*[run(r,un) for r in recs[:8]])
    print('use_norm',un,'acc',sum(a)/len(a))
h.remove()
