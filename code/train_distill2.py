"""Dense-draft distillation (hard-label CE, chunked lm_head, optional DDP)."""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from draft import load_dense, save_dense

D = "/path/to/spec-research/data"
PAD = 248044


def log0(*a, **k):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*a, **k, flush=True)


def load_rows(path, max_len=1536, limit=None):
    rows = []
    for l in open(path):
        r = json.loads(l)
        inp, out = r["input_ids"], r["output_ids"]
        while out and out[-1] == PAD:
            out = out[:-1]
        if len(out) < 2:
            continue
        if len(inp) + len(out) > max_len:
            keep = max_len - len(inp)
            if keep < 8:
                continue
            out = out[:keep]
        rows.append({"ids": inp + out, "n_prompt": len(inp), "n_out": len(out)})
        if limit and len(rows) >= limit:
            break
    return rows


def bucket_batches(rows, max_tok, seed=0):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(rows[i]["ids"])
        if cur and max(cur_max, L) * (len(cur) + 1) > max_tok:
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
    ids = torch.full((len(idxs), L), pad, dtype=torch.long)
    labels = torch.full((len(idxs), L), -100, dtype=torch.long)
    for b, i in enumerate(idxs):
        s = torch.tensor(rows[i]["ids"], dtype=torch.long)
        ids[b, :len(s)] = s
        labels[b, rows[i]["n_prompt"]:len(s)] = s[rows[i]["n_prompt"]:]
    return ids, labels


def chunked_ce(model, ids, labels, chunk=2048, scale=1.0, backward=True):
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
    sel = labels != -100
    h = hidden[sel]
    y = labels[sel]
    N = max(int(y.numel()), 1)
    tot = 0.0
    hit = 0
    for s in range(0, int(y.numel()), chunk):
        hc = h[s:s + chunk]
        yc = y[s:s + chunk]
        logits = model.lm_head(hc).float()
        l = F.cross_entropy(logits, yc, reduction="sum")
        if backward:
            (l * scale / N).backward(retain_graph=(s + chunk < int(y.numel())))
        tot += l.item()
        with torch.no_grad():
            hit += (logits.argmax(-1) == yc).sum().item()
    return tot / N, hit / N, int(y.numel())


@torch.no_grad()
def quick_val(model, val_rows, n=30, chunk=2048):
    model.eval()
    tot_loss = tot_hit = tot_n = 0
    for s in range(0, min(n, len(val_rows)), 6):
        idxs = list(range(s, min(s + 6, len(val_rows))))
        ids, labels = collate(val_rows, idxs)
        ids, labels = ids.cuda(), labels.cuda()
        l, h, m = chunked_ce(model, ids, labels, chunk=chunk, backward=False)
        tot_loss += l * m; tot_hit += h * m; tot_n += m
    model.train()
    return tot_loss / max(tot_n, 1), tot_hit / max(tot_n, 1)


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
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--init", default="")
    a = ap.parse_args()

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank(); local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dev = f"cuda:{local}"
    else:
        rank, dev = 0, "cuda:0"
    torch.manual_seed(0 + rank)

    rows = load_rows(a.data, max_len=a.max_len)
    val = load_rows(a.val, max_len=a.max_len)
    log0("train seqs", len(rows), "out tokens", sum(r["n_out"] for r in rows), "val", len(val))
    batches = bucket_batches(rows, a.max_tok, seed=0)
    batches = [b for i, b in enumerate(batches) if (not ddp) or (i % dist.get_world_size() == rank)]
    steps_per_epoch = max(len(batches), 1)
    total = a.steps or max(1, int(steps_per_epoch * a.epochs))
    log0("steps/epoch(rank)", steps_per_epoch, "total", total)

    model, meta = load_dense(a.ckpt, dtype=torch.bfloat16, device=dev)
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if a.freeze_backbone:
        for n, p in model.named_parameters():
            if ".mlp." not in n:
                p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    log0("trainable", sum(p.numel() for p in params) / 1e9, "B")
    def sync_grads():
        if not ddp:
            return
        ws = dist.get_world_size()
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= ws
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, total - a.warmup)
        return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    t0 = time.time()
    seen = 0
    bl = 0
    rl = ra = rn = 0.0
    bi = 0
    for step in range(total):
        if bi >= len(batches):
            bi = 0
            random.shuffle(batches)
        idxs = batches[bi]; bi += 1
        ids, labels = collate(rows, idxs)
        ids, labels = ids.to(dev), labels.to(dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss, acc, ntok = chunked_ce(model, ids, labels)
        sync_grads()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        seen += ntok * (dist.get_world_size() if ddp else 1)
        bl += loss; ra += acc; rn += 1
        if step % 10 == 0:
            dt = time.time() - t0
            log0(f"step {step}/{total} loss {bl/rn:.4f} acc {ra/rn:.4f} tok/s {seen/dt:.0f} "
                 f"lr {lr_at(step):.2e}")
            bl = ra = rn = 0.0
        if (step + 1) % a.save_every == 0 or step + 1 == total:
            m = model
            vl, va = quick_val(m, val)
            log0(f"[val] step {step+1} loss {vl:.4f} token_acc {va:.4f}")
            if rank == 0:
                meta2 = dict(meta)
                meta2.update({"step": step + 1, "train_tokens": seen, "val_token_acc": va})
                save_dense(m, a.out, meta2)
    log0("done", seen, "tokens", time.time() - t0)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
