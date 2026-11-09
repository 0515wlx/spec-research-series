import torch,time
from transformers import AutoTokenizer, AutoModelForCausalLM
M='/path/to/models/Qwen3.6-35B-A3B'
tok=AutoTokenizer.from_pretrained(M)
model=AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map={'':0}, low_cpu_mem_usage=True).eval()
p='Solve: Let x be a real number such that x^2-5x+6=0. Find the sum of all possible values of x. Please reason step by step, and put your final answer within \\boxed{}.'
msgs=[{'role':'user','content':p}]
text=tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
print(repr(text[-200:]))
B=32
enc=tok([text]*B, return_tensors='pt', padding=True).to('cuda')
for B2,N in [(8,256),(32,256),(64,256)]:
    enc=tok([text]*B2, return_tensors='pt', padding=True).to('cuda')
    torch.cuda.synchronize(); t=time.time()
    with torch.no_grad(): out=model.generate(**enc, max_new_tokens=N, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(); dt=time.time()-t
    print(f'B={B2} N={N} time={dt:.1f}s tok/s={B2*N/dt:.0f} mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB')
print('outlen', out.shape)
print(tok.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)[:500])
