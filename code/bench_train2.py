import sys,torch,time; sys.path.insert(0,'code')
from draft import load_dense
m,_=load_dense('ckpt/dense_merge_usage',dtype=torch.bfloat16,device='cuda:0')
m.train(); m.config.use_cache=False
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
opt=torch.optim.AdamW(m.parameters(),lr=1e-5)
for B,T in [(4,2048),(8,2048),(12,1536)]:
    ids=torch.randint(0,1000,(B,T),device='cuda'); lab=ids.clone()
    for it in range(2):
        torch.cuda.synchronize(); t=time.time()
        out=m(input_ids=ids,use_cache=False).logits
        loss=torch.nn.functional.cross_entropy(out.view(-1,out.shape[-1]).float(),lab.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize(); dt=time.time()-t
    print(f'B{B} T{T} {dt:.2f}s {B*T/dt:.0f} tok/s mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB',flush=True)
