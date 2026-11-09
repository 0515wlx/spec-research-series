"""End-to-end distillation of the dense draft on teacher-generated sequences (hard-label CE)."""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from draft import load_dense, save_dense

D = "/path/to/spec-research/data"
PAD = 248044


def load_rows(path, max_len=2048, limit=None):
    rows = []
    for l in open(path):
        r = json.loads(l)
        inp, out = r["input_ids"], r["output_ids"]
        while out and out[-1] == PAD:
            out = out[:-1]
        if len(out) < 2:
            continue
        ids = inp + out
        if len(ids) > max_len:
            # keep prompt, truncate completion
            keep = max_len - len(inp)
            if keep < 8:
                ids = ids[-max_len:]
                rows.append({"ids": ids, "n_prompt": 0})
                continue
            out = out[:keep]
            ids = inp + out
        rows.append({"ids": ids, "n_prompt": len(inp), "n_out": len(out)})
        if limit and len(rows) >= limit:
            break
    return rows


def bucket_batches(rows, max_tok, seed=0):
    """Group rows of similar length, yield batches with total padded tokens <= max_tok."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(rows[i]["ids"])
        nb = max(cur_max, L) * (len(cur) + 1)
        if cur and nb > max_tok:
            batches.append(cur)
            cur, cur_max = [i], L
        else:
            cur.append(i)
            cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    rng = random.Random(seed)
    rng.shuffle(batches)
    return batches


def collate(rows, idxs, pad=PAD):
    L = max(len(rows[i]["ids"]) for i in idxs)
    B = len(idxs)
    ids = torch.full((B, L), pad, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    for b, i in enumerate(idxs):
        r = rows[i]
        s = torch.tensor(r["ids"], dtype=torch.long)
        ids[b, :len(s)] = s
        labels[b, r["n_prompt"]:len(s)] = s[r["n_prompt"]:]
    return ids, labels


@torch.no_grad()
def quick_val(model, val_rows, device, n=24):
    model.eval()
    hit = tot = 0
    for i in range(0, min(n, len(val_rows)), 6):
        idxs = list(range(i, min(i + 6, len(val_rows))))
        ids, labels = collate(val_rows, idxs)
        ids, labels = ids.to(device), labels.to(device)
        logits = model(input_ids=ids, attention_mask=(ids != PAD).long(), use_cache=False).logits
        m = labels != -100
        pred = logits[m].argmax(-1)
        hit += (pred == labels[m]).sum().item(); tot += m.sum().item()
    model.train()
    return hit / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/dense_ffnfit")
    ap.add_argument("--out", default="ckpt/dense_distill")
    ap.add_argument("--data", default=f"{D}/distill_nothink.jsonl")
    ap.add_argument("--val", default=f"{D}/teacher_math500_nothink.jsonl")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_tok", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_len", type=int, default=1536)
    a = ap.parse_args()

    torch.manual_seed(0)
    rows = load_rows(a.data, max_len=a.max_len)
    val = load_rows(a.val, max_len=a.max_len)
    print("train seqs", len(rows), "tokens", sum(r["n_out"] for r in rows), "val", len(val), flush=True)
    batches = bucket_batches(rows, a.max_tok)
    steps_per_epoch = len(batches)
    total = a.steps or int(steps_per_epoch * a.epochs)
    print("steps/epoch", steps_per_epoch, "total", total, flush=True)

    model, meta = load_dense(a.ckpt, dtype=torch.bfloat16, device=a.device)
    model.train()
    model.config.use_cache = False
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0,
                            eps=1e-8)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, total - a.warmup)
        return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    t0 = time.time()
    seen = 0
    run_loss = 0.0
    run_acc = 0.0
    run_n = 0
    bi = 0
    for step in range(total):
        if bi >= len(batches):
            bi = 0
            random.shuffle(batches)
        idxs = batches[bi]; bi += 1
        ids, labels = collate(rows, idxs)
        ids, labels = ids.to(a.device), labels.to(a.device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        logits = model(input_ids=ids, attention_mask=(ids != PAD).long(), use_cache=False).logits
        m = labels != -100
        loss = F.cross_entropy(logits[m].float(), labels[m])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            acc = (logits[m].argmax(-1) == labels[m]).float().mean().item()
        ntok = int(m.sum())
        seen += ntok
        run_loss += loss.item(); run_acc += acc; run_n += 1
        if step % 10 == 0:
            dt = time.time() - t0
            print(f"step {step}/{total} loss {np.mean(run_loss/max(run_n,1)):.4f} acc {run_acc/run_n:.4f} "
                  f"tok/s {seen/dt:.0f} lr {lr_at(step):.2e}", flush=True)
            run_loss = run_acc = run_n = 0
        if (step + 1) % a.save_every == 0 or step + 1 == total:
            v = quick_val(model, val, a.device)
            print(f"[val] step {step+1} token_acc {v:.4f}", flush=True)
            meta2 = dict(meta); meta2.update({"step": step + 1, "train_tokens": seen,
                                              "loss": float(loss.item()), "val_token_acc": v})
            save_dense(model, a.out, meta2)
    print("done", seen, "tokens", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
