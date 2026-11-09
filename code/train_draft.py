"""Dense-draft training: hard-label CE distillation and/or expected-acceptance-length objective.

RL objective (differentiable surrogate, no sampling): within blocks of `gamma` draft positions,
maximize  E_t[ prod_{i<=k} p_draft(teacher_token_i | prefix) ], i.e. the expected number of
consecutively accepted draft tokens -- exactly the quantity greedy speculative decoding rewards.
"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model_io import load_model, save_model

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
    # shift: logits at position t predict token t+1 (HF causal-LM convention)
    shifted = torch.full_like(labels, -100)
    shifted[:, :-1] = labels[:, 1:]
    return ids, shifted


def ce_loss_batched(model, ids, labels, chunk=1024, scale=1.0):
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
    B, T, H = hidden.shape
    hflat = hidden.reshape(B * T, H)
    yflat = labels.reshape(-1)
    ntok = int((yflat != -100).sum())
    ce_sum = 0.0
    hit = 0
    last = (B * T)
    for s0 in range(0, last, chunk):
        hc = hflat[s0:s0 + chunk]
        yc = yflat[s0:s0 + chunk]
        mc = yc != -100
        if not mc.any():
            continue
        logits = model.lm_head(hc).float()
        l = F.cross_entropy(logits, yc.clamp(min=0), reduction="none") * mc
        ce_sum += l.sum().item()
        hit += ((logits.argmax(-1) == yc) & mc).sum().item()
        if scale > 0:
            (l.sum() * scale / max(ntok, 1)).backward(retain_graph=(s0 + chunk < last))
    return ce_sum / max(ntok, 1), hit / max(ntok, 1), ntok


def rl_loss_per_seq(model, ids, labels, gamma=5, ce_w=1.0, rl_w=1.0, scale=1.0):
    """Expected (soft) acceptance length objective; returns per-seq grads already backpropagated."""
    B, T = labels.shape
    dev = ids.device
    tot_ce = tot_rl = tot_hit = 0.0
    ntok = 0
    for b in range(B):
        m = labels[b] != -100
        n = int(m.sum())
        if n < 2:
            continue
        hidden = model.model(input_ids=ids[b:b + 1], use_cache=False).last_hidden_state
        gs = []
        ces = []
        for s0 in range(0, T, 1024):
            yc = labels[b, s0:s0 + 1024]
            if (yc == -100).all():
                gs.append(torch.zeros(yc.shape[0], device=dev))
                continue
            logits = model.lm_head(hidden[:, s0:s0 + 1024].reshape(-1, hidden.shape[-1])).float()
            l = F.cross_entropy(logits, yc.clamp(min=0), reduction="none") * (yc != -100)
            ces.append(l.sum())
            lsm = logits.log_softmax(-1)
            gs.append((lsm.gather(1, yc.clamp(min=0).unsqueeze(1)).squeeze(1) * (yc != -100)))
            tot_hit += ((logits.argmax(-1) == yc) & (yc != -100)).sum().item()
        lp = torch.cat(gs)[:T]
        ce = torch.stack(ces).sum() / n if ces else torch.zeros((), device=dev)
        # block prefix products
        pos = torch.arange(T, device=dev)
        gen_start = int(m.float().argmax().item())
        rel = pos - gen_start
        blk = torch.where(rel >= 0, torch.div(rel, gamma, rounding_mode="floor"),
                          torch.zeros_like(rel))
        cs = torch.cumsum(lp, 0)
        start_idx = (blk * gamma + gen_start).clamp(0, T - 1)
        cum = cs - cs.gather(0, start_idx)
        r = torch.exp(cum) * m.float()
        rl_val = r.sum() / n
        loss = ce_w * ce + rl_w * (-rl_val)
        (loss * scale).backward()
        tot_ce += ce.item() * n
        tot_rl += rl_val.item() * n
        ntok += n
    return tot_ce / max(ntok, 1), tot_hit / max(ntok, 1), ntok, tot_rl / max(ntok, 1)




def rl_loss_batched(model, ids, labels, gamma=5, ce_w=1.0, rl_w=1.0, mb=2, scale=1.0):
    """Batched CE + expected-acceptance-length (soft) objective, single backward per micro-batch."""
    B, T = labels.shape
    nmicro = (B + mb - 1) // mb
    sc = scale / nmicro
    tot_ce = tot_rl = tot_hit = 0.0
    ntok = 0
    for mstart in range(0, B, mb):
        idm = ids[mstart:mstart + mb]
        lab = labels[mstart:mstart + mb]
        b = lab.shape[0]
        hidden = model.model(input_ids=idm, use_cache=False).last_hidden_state
        logits = model.lm_head(hidden).float()          # (b,T,V)
        y = lab.clamp(min=0)
        mask = lab != -100
        useg = torch.where(mask, y, torch.zeros_like(y))
        lsm = logits.log_softmax(-1)
        lp = lsm.gather(2, useg.unsqueeze(-1)).squeeze(-1) * mask
        ce_seq = -(lp * mask).sum(1) / mask.sum(1).clamp(min=1)     # (b,)
        pos = torch.arange(T, device=ids.device).view(1, T)
        gs = mask.float().argmax(1, keepdim=True)
        rel = pos - gs
        blk = torch.where(rel >= 0, torch.div(rel, gamma, rounding_mode="floor"),
                          torch.zeros_like(rel))
        cs = torch.cumsum(lp, 1)
        base = cs.gather(1, (blk * gamma + gs).clamp(0, T - 1))
        cum = cs - base
        r = torch.exp(cum) * mask.float()
        rl_seq = r.sum(1) / mask.sum(1).clamp(min=1)
        nn = int(mask.sum())
        loss = ce_w * ce_seq.mean() + rl_w * (-rl_seq.mean())
        (loss * sc).backward()
        tot_ce += float(ce_seq.mean()) * nn
        tot_rl += float(rl_seq.mean()) * nn
        with torch.no_grad():
            tot_hit += ((logits.argmax(-1) == y) & mask).sum().item()
        ntok += nn
    return tot_ce / max(ntok, 1), tot_hit / max(ntok, 1), ntok, tot_rl / max(ntok, 1)



@torch.no_grad()
def quick_val(model, val_rows, n=30, gamma=5):
    model.eval()
    tot_ce = tot_acc = tot_n = 0.0
    for s in range(0, min(n, len(val_rows)), 6):
        idxs = list(range(s, min(s + 6, len(val_rows))))
        ids, labels = collate(val_rows, idxs)
        ids, labels = ids.cuda(), labels.cuda()
        ce, acc, ntok = ce_loss_batched(model, ids, labels, scale=0.0)
        tot_ce += ce * ntok; tot_acc += acc * ntok; tot_n += ntok
    # soft RL metric with grad disabled
    tot_rl = tot_rn = 0.0
    for i in range(min(n, len(val_rows))):
        ids, labels = collate(val_rows, [i])
        ids, labels = ids.cuda(), labels.cuda()
        B, T = labels.shape
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
        m = labels[0] != -100
        n = int(m.sum())
        lp = torch.zeros(T, device=ids.device)
        for s0 in range(0, T, 1024):
            yc = labels[0, s0:s0 + 1024]
            if (yc == -100).all():
                continue
            logits = model.lm_head(hidden[:, s0:s0 + 1024].reshape(-1, hidden.shape[-1])).float()
            lp[s0:s0 + 1024] = logits.log_softmax(-1).gather(
                1, yc.clamp(min=0).unsqueeze(1)).squeeze(1) * (yc != -100)
        pos = torch.arange(T, device=ids.device)
        gs = int(m.float().argmax().item())
        rel = pos - gs
        blk = torch.where(rel >= 0, torch.div(rel, gamma, rounding_mode="floor"), torch.zeros_like(rel))
        cs = torch.cumsum(lp, 0)
        cum = cs - cs.gather(0, (blk * gamma + gs).clamp(0, T - 1))
        tot_rl += float((torch.exp(cum) * m.float()).sum() / n)
        tot_rn += 1
    model.train()
    return tot_ce / max(tot_n, 1), tot_acc / max(tot_n, 1), tot_rl / max(tot_rn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/dense_ffnfit")
    ap.add_argument("--out", default="ckpt/dense_distill")
    ap.add_argument("--data", default=f"{D}/distill_nothink.jsonl")
    ap.add_argument("--val", default=f"{D}/teacher_math500_nothink.jsonl")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_tok", type=int, default=6144)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--ce_w", type=float, default=1.0)
    ap.add_argument("--rl_w", type=float, default=0.0)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--mb", type=int, default=2)
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

    model, meta = load_model(a.ckpt, dtype=torch.bfloat16, device=dev)
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
    bl = ba = br = 0.0
    rn = 0
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
        ws = dist.get_world_size() if ddp else 1
        if a.rl_w > 0:
            ce, acc, ntok, rl = rl_loss_batched(model, ids, labels, gamma=a.gamma,
                                                ce_w=a.ce_w, rl_w=a.rl_w, mb=a.mb)
        else:
            ce, acc, ntok = ce_loss_batched(model, ids, labels)
            rl = 0.0
        sync_grads()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        seen += ntok * (dist.get_world_size() if ddp else 1)
        bl += ce; ba += acc; br += rl; rn += 1
        if step % 10 == 0:
            dt = time.time() - t0
            log0(f"step {step}/{total} ce {bl/rn:.4f} acc {ba/rn:.4f} rl {br/rn:.4f} "
                 f"tok/s {seen/dt:.0f} lr {lr_at(step):.2e}")
            bl = ba = br = rn = 0.0
        if (step + 1) % a.save_every == 0 or step + 1 == total:
            vl, va, vr = quick_val(model, val, gamma=a.gamma)
            log0(f"[val] step {step+1} ce {vl:.4f} token_acc {va:.4f} soft_tau_frac {vr:.4f}")
            if rank == 0:
                meta2 = dict(meta)
                meta2.update({"step": step + 1, "train_tokens": seen, "val_token_acc": va,
                              "val_soft_rl": vr})
                save_model(model, a.out, meta2)
    log0("done", seen, "tokens", time.time() - t0)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
