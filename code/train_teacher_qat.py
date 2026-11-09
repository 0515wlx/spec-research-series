"""Short QAT of the ORIGINAL teacher model (Qwen3.6-35B-A3B) with W4 fake quant.

Full 35B QAT needs FSDP; for a quick extreme-case check we keep the routed experts
frozen at PTQ W4 and adapt the cheap always-active parts:
  - router (`mlp.gate.weight`)
  - shared expert + shared_expert_gate
  - all norms
  - optionally lm_head / attention (off by default: memory)

The experts are still fake-quantized in the forward (STE), so this is a real QAT
of the adapted parameters against the quantized expert features.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from quant import (apply_fake_quant, refresh_fake_quant, clear_fake_quant,
                   dequantize_inplace)
from quant_teacher import load_teacher, patch_moe_experts
from packed_moe import pack_experts_inplace, patch_packed_experts
from eval_accept import get_tok, token_matches
from train_draft import load_rows, collate

D = "/path/to/spec-research/data"


def is_trainable(name, parts):
    if "router" in parts and name.endswith(".mlp.gate.weight"):
        return True
    if "shared" in parts and ("shared_expert" in name or "shared_expert_gate" in name):
        return True
    if "norm" in parts and ("layernorm" in name or ".norm." in name or "q_norm" in name
                            or "k_norm" in name):
        return True
    if "lm_head" in parts and name == "lm_head.weight":
        return True
    if "attn" in parts and (".self_attn." in name or ".linear_attn." in name):
        return True
    return False


@torch.no_grad()
def quick_eval(model, val_rows, n=20, gamma=8):
    model.eval()
    tot = hit = 0
    for i in range(min(n, len(val_rows))):
        ids, labels = collate(val_rows, [i])
        ids, labels = ids.cuda(), labels.cuda()
        refresh_fake_quant(model)
        logits = model(input_ids=ids, use_cache=False).logits
        m = labels[0] != -100
        y = labels[0][m]
        pred = logits[0][m].argmax(-1)
        hit += int((pred == y).sum())
        tot += int(m.sum())
        clear_fake_quant(model)
    model.train()
    return hit / max(tot, 1)


def simulate(m, gammas):
    out = {}
    for g in gammas:
        acc = prop = 0
        lens = []
        for mm in m:
            t, N = 0, len(mm)
            while t < N:
                k = 0
                while k < g and t + k < N and mm[t + k]:
                    k += 1
                acc += k
                prop += min(g, N - t)
                lens.append(k + 1)
                t += k + 1
        out[str(g)] = {"alpha": acc / max(prop, 1), "tau": float(np.mean(lens))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{D}/distill_nothink.jsonl")
    ap.add_argument("--val", default=f"{D}/teacher_math500_nothink.jsonl")
    ap.add_argument("--out", default="ckpt/teacher_qat_router_shared.pt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--lm_head_bits", type=int, default=4)
    ap.add_argument("--parts", default="attn,router,shared,norm,lm_head")
    ap.add_argument("--pack_experts", action="store_true",
                    help="store routed experts as packed int4 (frees ~48GB; recommended)")
    ap.add_argument("--eval_limit", type=int, default=50)
    ap.add_argument("--gammas", default="1,2,3,5,8,12,16,24,32")
    ap.add_argument("--eval_out", default="out/eval_teacher_qat_router_shared.json")
    a = ap.parse_args()
    parts = set(a.parts.split(","))

    rows = load_rows(a.data, max_len=a.max_len)
    val_quick = load_rows(a.val, max_len=1024)   # only for cheap in-training val
    print("train seqs", len(rows), "val_quick", len(val_quick), flush=True)

    model = load_teacher(0)
    names = apply_fake_quant(model, bits=a.bits, group=a.group,
                             lm_head_bits=a.lm_head_bits)
    if a.pack_experts:
        nbytes = pack_experts_inplace(model, bits=a.bits, group=a.group)
        patch_packed_experts()
        print(f"packed experts: {nbytes/1e9:.2f} GB", flush=True)
    else:
        patch_moe_experts(a.bits, a.group, ste=True)
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    for n, p in model.named_parameters():
        p.requires_grad_(is_trainable(n, parts))
    params = [p for p in model.parameters() if p.requires_grad]
    print("quantized linears", len(names), "trainable",
          sum(p.numel() for p in params) / 1e9, "B", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.0, eps=1e-8)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(p, 1.0))))

    t0 = time.time()
    seen = 0
    bl = ba = 0.0
    rn = 0
    for step in range(a.steps):
        ids, labels = collate(rows, [step % len(rows)])
        ids, labels = ids.cuda(), labels.cuda()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        refresh_fake_quant(model)
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
        logits = model.lm_head(hidden).float()
        y = labels.clamp(min=0)
        mask = labels != -100
        ce = F.cross_entropy(logits[mask], y[mask])
        acc = (logits[mask].argmax(-1) == y[mask]).float().mean()
        ce.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        clear_fake_quant(model)
        seen += int(mask.sum())
        bl += float(ce); ba += float(acc); rn += 1
        if step % 10 == 0:
            dt = time.time() - t0
            print(f"step {step}/{a.steps} ce {bl/rn:.4f} acc {ba/rn:.4f} "
                  f"tok/s {seen/dt:.0f}", flush=True)
            bl = ba = rn = 0.0
        if (step + 1) % 50 == 0:
            va = quick_eval(model, val_quick, n=20, gamma=8)
            print(f"[val] step {step+1} token_acc {va:.4f}", flush=True)

    # save only the adapted parameters
    obj = {"parts": sorted(parts), "bits": a.bits, "group": a.group,
           "steps": a.steps, "train_tokens": seen,
           "state": {n: p.detach().cpu() for n, p in model.named_parameters()
                     if p.requires_grad}}
    torch.save(obj, a.out)
    print("saved", a.out, flush=True)

    # full acceptance eval in the same process
    tok = get_tok()
    recs = [json.loads(l) for l in open(a.val)][:a.eval_limit]
    seqs, gen_lens = [], []
    for r in recs:
        inp, out = r["input_ids"], r["output_ids"]
        while out and out[-1] == tok.pad_token_id:
            out = out[:-1]
        if len(out) < 8:
            continue
        seqs.append(torch.tensor(inp + out, dtype=torch.long))
        gen_lens.append(len(out))
    model.eval()
    refresh_fake_quant(model)
    t0 = time.time()
    matches = token_matches(model, tok, seqs, gen_lens, "cuda:0", bs=1, max_tok=4096)
    gammas = [int(x) for x in a.gammas.split(",")]
    info = {"tag": "teacher_qat", "parts": sorted(parts), "bits": a.bits,
            "group": a.group, "steps": a.steps, "train_tokens": seen,
            "n_seq": len(seqs), "eval_time": time.time() - t0,
            "pos_match_rate": float(sum(int(m.sum()) for m in matches) /
                                    max(sum(len(m) for m in matches), 1)),
            "per_gamma": simulate(matches, gammas)}
    surv = {}
    for k in [1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64]:
        tot = cnt = 0
        for m in matches:
            if len(m) < k:
                continue
            ok = np.convolve(m.astype(np.int8), np.ones(k, dtype=np.int8), "valid") == k
            tot += int(ok.sum()); cnt += len(ok)
        surv[str(k)] = tot / max(cnt, 1)
    info["survival"] = surv
    print(json.dumps(info, indent=1))
    json.dump(info, open(a.eval_out, "w"), indent=1)


if __name__ == "__main__":
    main()
