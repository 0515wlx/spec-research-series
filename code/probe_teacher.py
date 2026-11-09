import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
M='/path/to/models/Qwen3.6-35B-A3B'
t0=time.time()
tok=AutoTokenizer.from_pretrained(M)
print('tok',time.time()-t0, tok.pad_token_id, tok.eos_token_id)
model=AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map={'':0}, low_cpu_mem_usage=True)
print('load',time.time()-t0, torch.cuda.memory_allocated()/1e9)
print('params', sum(p.numel() for p in model.parameters())/1e9)
print('cfg', model.config.model_type)
msgs=[{'role':'user','content':'What is 2+2? Answer briefly.'}]
ids=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', return_dict=True)
print(ids.keys(), ids['input_ids'].shape)
ids={k:v.to('cuda') for k,v in ids.items()}
with torch.no_grad():
    t=time.time(); out=model.generate(**ids, max_new_tokens=64, do_sample=False); print('gen1', time.time()-t)
    t=time.time(); out=model.generate(**ids, max_new_tokens=64, do_sample=False); print('gen2', time.time()-t)
print(tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=False))
