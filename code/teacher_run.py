"""Teacher (Qwen3.6-35B-A3B) jobs: router stats, eval-set generation, distillation-data generation."""
import argparse, json, os, sys, time, random
import numpy as np
import torch

M = "/path/to/models/Qwen3.6-35B-A3B"
D = "/path/to/spec-research/data"
TORCH_DTYPE = torch.bfloat16


def get_model(device=0):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(
        M, dtype=TORCH_DTYPE, device_map={"": device}, low_cpu_mem_usage=True,
        attn_implementation="sdpa")
    model.eval()
    return tok, model


def chat(tok, problem, thinking=False):
    return tok.apply_chat_template([{"role": "user", "content": problem}],
                                   add_generation_prompt=True, tokenize=False,
                                   enable_thinking=thinking)


# ---------------- router statistics ----------------
def run_calib(tok, model, n_samples=160, maxlen=2048, out="calib_stats.npz"):
    layers = model.model.layers
    nL = len(layers)
    counts = np.zeros((nL, 256), dtype=np.float64)
    wsum = np.zeros((nL, 256), dtype=np.float64)
    ssum = np.zeros(nL, dtype=np.float64)
    scnt = np.zeros(nL, dtype=np.float64)
    caps = {}

    def mk(i):
        def hook(mod, inp, out):
            _, scores, idx = out  # scores [T,k] normalized, idx [T,k]
            s = scores.detach().float().cpu()
            ix = idx.detach().cpu()
            for j in range(ix.shape[1]):
                np.add.at(counts[i], ix[:, j].numpy(), 1.0)
                np.add.at(wsum[i], ix[:, j].numpy(), s[:, j].numpy())
            caps[i] = None
        return hook

    def mk_shared(i):
        def hook(mod, inp, out):
            g = torch.sigmoid(out.detach().float().cpu())
            ssum[i] += g.sum().item()
            scnt[i] += g.numel()
        return hook

    hs = [layers[i].mlp.gate.register_forward_hook(mk(i)) for i in range(nL)]
    hs += [layers[i].mlp.shared_expert_gate.register_forward_hook(mk_shared(i)) for i in range(nL)]

    rows = [json.loads(l) for l in open(f"{D}/train_pool.jsonl")]
    rng = random.Random(0)
    rng.shuffle(rows)
    t0 = time.time()
    with torch.no_grad():
        for n, r in enumerate(rows[:n_samples]):
            text = f"Problem:\n{r['problem']}\n\nSolution:\n{r['solution']}"
            ids = tok(text, return_tensors="pt", truncation=True, max_length=maxlen)["input_ids"].to(model.device)
            if ids.shape[1] < 64:
                continue
            model(input_ids=ids, use_cache=False)
            if (n + 1) % 20 == 0:
                print(f"calib {n+1}/{n_samples} {time.time()-t0:.0f}s", flush=True)
            if time.time() - t0 > 1200:
                print("calib time cap hit", flush=True)
                break
    for h in hs:
        h.remove()
    usage = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    mean_w = wsum / np.maximum(counts, 1)
    top8 = np.argsort(-usage, axis=1)[:, :8]
    shared_w = ssum / np.maximum(scnt, 1)
    np.savez(f"{D}/{out}", usage=usage, mean_w=mean_w, top8=top8, shared_w=shared_w, counts=counts)
    for i in [0, 3, 20]:
        print(f"layer{i} top8 {top8[i].tolist()} w {np.round(mean_w[i][top8[i]], 3).tolist()} shared {shared_w[i]:.3f}", flush=True)
    print("saved calib", flush=True)


# ---------------- generation ----------------
@torch.no_grad()
def gen_batch(model, tok, prompts, thinking, max_new, bs, on_batch=None):
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), bs):
        chunk = order[s:s + bs]
        texts = [chat(tok, prompts[i], thinking) for i in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, use_cache=True,
                             pad_token_id=tok.pad_token_id)
        plen = enc["input_ids"].shape[1]
        outs = []
        for j, i in enumerate(chunk):
            new = gen[j][plen:].tolist()
            outs.append((i, enc["input_ids"][j].tolist(), new))
        if on_batch is not None:
            on_batch(outs, s + len(chunk), len(prompts))
        print(f"gen {s+len(chunk)}/{len(prompts)}", flush=True)
    return


def run_gen(tok, model, infile, outfile, thinking, max_new, bs, limit=None, shard=0, nshard=1):
    rows = [json.loads(l) for l in open(infile)]
    rows = [r for i, r in enumerate(rows) if i % nshard == shard]
    done = set()
    if os.path.exists(outfile):
        for l in open(outfile):
            try:
                done.add(json.loads(l)["idx"])
            except Exception:
                pass
    todo = [(i, r) for i, r in enumerate(rows) if i not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"todo {len(todo)} / {len(rows)}", flush=True)
    prompts = [r["problem"] for _, r in todo]
    t0 = time.time()
    ntok = [0]

    def write(outs, n_done, total):
        with open(outfile, "a") as f:
            for (k, inp, new) in outs:
                i, r = todo[k]
                rec = {"idx": i, "prompt": r["problem"], "input_ids": inp, "output_ids": new,
                       "src": r.get("src")}
                f.write(json.dumps(rec) + "\n")
                ntok[0] += len(new)
        dt = time.time() - t0
        print(f"[write] {n_done}/{total} seqs {ntok[0]} tok {dt:.0f}s {ntok[0]/max(dt,1):.0f} tok/s", flush=True)

    gen_batch(model, tok, prompts, thinking, max_new, bs, on_batch=write)
    print(f"done {len(todo)} seqs, {ntok[0]} tokens, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["calib", "eval", "gen", "all"])
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max_new", type=int, default=768)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--n_calib", type=int, default=160)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(f"{D}", exist_ok=True)
    tok, model = get_model()
    if a.task in ("calib", "all"):
        run_calib(tok, model, n_samples=a.n_calib)
    if a.task in ("eval", "all"):
        tag = "think" if a.thinking else "nothink"
        run_gen(tok, model, f"{D}/math500.jsonl", f"{D}/teacher_math500_{tag}.jsonl",
                a.thinking, a.max_new, a.bs, shard=a.shard, nshard=a.nshard)
    if a.task in ("gen", "all"):
        tag = "think" if a.thinking else "nothink"
        run_gen(tok, model, f"{D}/train_pool.jsonl", f"{D}/distill_{tag}.jsonl",
                a.thinking, a.max_new, a.bs, limit=a.limit, shard=a.shard, nshard=a.nshard)
