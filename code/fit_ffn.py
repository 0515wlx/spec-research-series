"""Layer-wise functional distillation of the MoE FFN into the dense FFN (fast warm start)."""
import argparse, json, os, sys, time, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from draft import DenseMLP, build_dense_model, load_dense, save_dense, TEACHER, DENSE_INTER

D = "/path/to/spec-research/data"


def collect(docs, tok, model, max_tokens=100000):
    layers = model.model.layers
    nL = len(layers)
    xs = [[] for _ in range(nL)]
    ys = [[] for _ in range(nL)]
    hs = []

    def mk(i):
        def hook(mod, inp, out):
            x = inp[0]
            xs[i].append(x[0].detach().to(torch.bfloat16).cpu())
            ys[i].append(out[0].detach().to(torch.bfloat16).cpu())
        return hook

    for i in range(nL):
        hs.append(layers[i].mlp.register_forward_hook(mk(i)))
    n = 0
    t0 = time.time()
    with torch.no_grad():
        for r in docs:
            text = f"Problem:\n{r['problem']}\n\nSolution:\n{r['solution']}"
            ids = tok(text, return_tensors="pt", truncation=True, max_length=2048)["input_ids"].to("cuda")
            if ids.shape[1] < 32:
                continue
            model(input_ids=ids, use_cache=False)
            n += ids.shape[1]
            if n >= max_tokens:
                break
            if n % 20000 < 2048:
                print("collected", n, f"{time.time()-t0:.0f}s", flush=True)
    for h in hs:
        h.remove()
    X = [torch.cat(x, 0) for x in xs]   # list of (N,H)
    Y = [torch.cat(y, 0) for y in ys]
    print("collected tokens", n, "per-layer", X[0].shape, flush=True)
    return X, Y


def fit_layer(mlp, X, Y, steps=300, bs=4096, lr=2e-4, device="cuda:0"):
    mlp = mlp.to(device)
    params = list(mlp.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    N = X.shape[0]
    ymean = Y.mean(0, keepdim=True).to(device)
    denom = ((Y.to(device) - ymean) ** 2).sum().item()

    def relerr():
        tot = 0.0
        with torch.no_grad():
            for s in range(0, N, 16384):
                xb = X[s:s + 16384].to(device, torch.bfloat16)
                yb = Y[s:s + 16384].to(device, torch.bfloat16)
                tot += ((mlp(xb).float() - yb.float()) ** 2).sum().item()
        return tot / max(denom, 1e-6)

    e0 = relerr()
    g = torch.Generator().manual_seed(0)
    for st in range(steps):
        idx = torch.randint(0, N, (bs,), generator=g)
        xb = X[idx].to(device, torch.bfloat16)
        yb = Y[idx].to(device, torch.bfloat16)
        out = mlp(xb)
        loss = ((out.float() - yb.float()) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if st % 100 == 0:
            print(f"  step {st} loss {loss.item():.5f}", flush=True)
    e1 = relerr()
    mlp.to("cpu")
    torch.cuda.empty_cache()
    return e0, e1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="ckpt/dense_merge_usage")
    ap.add_argument("--out", default="ckpt/dense_ffnfit")
    ap.add_argument("--tokens", type=int, default=100000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--layers", default="all")
    a = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(TEACHER)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True,
        attn_implementation="sdpa").eval()
    rows = [json.loads(l) for l in open(f"{D}/train_pool.jsonl")]
    random.Random(2).shuffle(rows)
    X, Y = collect(rows, tok, model, a.tokens)
    del model
    torch.cuda.empty_cache()

    dense, meta = load_dense(a.base, dtype=torch.bfloat16, device="cpu")
    meta["init"] = "merge+ffnfit"
    errs = []
    t0 = time.time()
    for i in range(len(dense.model.layers)):
        mlp = dense.model.layers[i].mlp
        e0, e1 = fit_layer(mlp, X[i], Y[i], steps=a.steps)
        errs.append((e0, e1))
        print(f"layer {i}: relerr {e0:.4f} -> {e1:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        X[i] = Y[i] = None
        if i == 0:
            meta["relerr_layer0"] = [e0, e1]
    meta["relerr"] = errs
    save_dense(dense, a.out, meta)
    print("mean relerr before", np.mean([e[0] for e in errs]), "after", np.mean([e[1] for e in errs]))


if __name__ == "__main__":
    main()
