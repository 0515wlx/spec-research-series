import sys,time,json,torch; sys.path.insert(0,'code')
from draft import load_dense
from transformers import AutoTokenizer, AutoModelForCausalLM
T='/path/to/models/Qwen3.6-35B-A3B'
tok=AutoTokenizer.from_pretrained(T)
r=json.loads(open('data/teacher_math500_nothink.jsonl').readline())
ids=torch.tensor(r['input_ids']).cuda()[None]
d,_=load_dense('ckpt/dense_distill_s1',dtype=torch.bfloat16,device='cuda:1'); d.eval()
with torch.no_grad():
    o=d.generate(input_ids=ids.to('cuda:1'),max_new_tokens=16,do_sample=False,pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(); t=time.time()
    o=d.generate(input_ids=ids.to('cuda:1'),max_new_tokens=128,do_sample=False,pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(); dt=time.time()-t
print('draft tok/s',128/dt, flush=True)
del d; torch.cuda.empty_cache()
th=AutoModelForCausalLM.from_pretrained(T,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True).eval()
with torch.no_grad():
    o=th.generate(input_ids=ids,max_new_tokens=8,do_sample=False,pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(); t=time.time()
    o=th.generate(input_ids=ids,max_new_tokens=64,do_sample=False,pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(); dt=time.time()-t
print('teacher tok/s',64/dt)
