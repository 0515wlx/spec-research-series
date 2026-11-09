"""Generic distillation trainer for Qwen3.5 text drafts.

Supports single-GPU and torchrun DDP, token-budget training, CE or expected-acceptance
objective, periodic validation, checkpoint/resume, and JSONL logs.

Examples:
  # single GPU, 2B
  python -m infra.train_distill --student qwen3.5-2b --data data/processed/distill.jsonl \
    --max-tokens 50000000 --out ckpt/distill/qwen3.5-2b-50m

  # 8-GPU DDP, 4B
  torchrun --nproc_per_node 8 -m infra.train_distill --student qwen3.5-4b ...

Notes:
  * Full fine-tuning 9B/27B with AdamW does not fit on one 80G GPU.  Use 2B/4B for the
    training arm, or port this loop to FSDP (see code/train_teacher_qat_fsdp.py).
  * `--objective accept` is the differentiable expected-acceptance-length surrogate from
    the old `train_draft.py`; `ce` is hard-label distillation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from infra.model_loader import load_text_model, load_tokenizer


def log0(*a, **k):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*a, **k, flush=True)


def load_rows(path: str, max_len: int, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            inp, out = r["input_ids"], r["output_ids"]
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


def bucket_batches(rows: list[dict], max_tok: int, seed: int = 0) -> list[list[int]]:
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
    random.Random(seed).shuffle(batches)
    return batches


def collate(rows: list[dict], idxs: list[int], pad: int, max_len: int | None = None):
    L = max(len(rows[i]["ids"]) for i in idxs)
    if max_len is not None:
        L = max(L, int(max_len))
    ids = torch.full((len(idxs), L), pad, dtype=torch.long)
    labels = torch.full((len(idxs), L), -100, dtype=torch.long)
    att = torch.zeros((len(idxs), L), dtype=torch.long)
    for b, i in enumerate(idxs):
        s = torch.tensor(rows[i]["ids"], dtype=torch.long)
        ids[b, :len(s)] = s
        att[b, :len(s)] = 1
        labels[b, rows[i]["n_prompt"]:len(s)] = s[rows[i]["n_prompt"]:]
    # HF CausalLM shifts internally: logits[:, :-1] predict labels[:, 1:].
    return ids, att, labels


def ce_step(model, ids, att, labels):
    out = model(input_ids=ids, attention_mask=att, labels=labels, use_cache=False)
    logits = out.logits.float()[:, :-1]
    y = labels[:, 1:]
    with torch.no_grad():
        mask = y != -100
        hit = ((logits.argmax(-1) == y.clamp(min=0)) & mask).sum().item()
        n = int(mask.sum())
    return out.loss, hit, n


def accept_step(model, ids, att, labels, gamma: int, ce_w: float, accept_w: float):
    out = model(input_ids=ids, attention_mask=att, use_cache=False)
    logits = out.logits.float()[:, :-1]
    y = labels[:, 1:]
    mask = y != -100
    logp = logits.log_softmax(-1).gather(2, y.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
    ce = -(logp.sum(1) / mask.sum(1).clamp(min=1)).mean()
    B, T = y.shape
    pos = torch.arange(T, device=ids.device).view(1, T)
    gs = mask.float().argmax(1, keepdim=True)
    rel = pos - gs
    blk = torch.where(rel >= 0, torch.div(rel, gamma, rounding_mode="floor"), torch.zeros_like(rel))
    cs = torch.cumsum(logp, 1)
    base = cs.gather(1, (blk * gamma + gs).clamp(0, T - 1))
    r = torch.exp(cs - base) * mask.float()
    acc = (r.sum(1) / mask.sum(1).clamp(min=1)).mean()
    loss = ce_w * ce - accept_w * acc
    with torch.no_grad():
        hit = ((logits.argmax(-1) == y.clamp(min=0)) & mask).sum().item()
        n = int(mask.sum())
    return loss, hit, n, float(ce), float(acc)


@torch.no_grad()
def quick_val(model, val_rows: list[dict], pad: int, gamma: int, max_batches: int = 20,
              seq_len: int | None = None):
    model.eval()
    ce_sum = hit = ntok = acc_sum = 0.0
    for bi, idxs in enumerate(bucket_batches(val_rows, 4096, seed=0)[:max_batches]):
        ids, att, labels = collate(val_rows, idxs, pad, max_len=seq_len)
        ids, att, labels = ids.cuda(), att.cuda(), labels.cuda()
        loss, h, n = ce_step(model, ids, att, labels)
        ce_sum += float(loss) * n
        hit += h; ntok += n
        # soft acceptance on the same batch
        out = model(input_ids=ids, attention_mask=att, use_cache=False)
        logits = out.logits.float()[:, :-1]
        y = labels[:, 1:]
        mask = y != -100
        logp = logits.log_softmax(-1).gather(2, y.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
        T = y.shape[1]
        pos = torch.arange(T, device=ids.device).view(1, T)
        gs = mask.float().argmax(1, keepdim=True)
        rel = pos - gs
        blk = torch.where(rel >= 0, torch.div(rel, gamma, rounding_mode="floor"), torch.zeros_like(rel))
        cs = torch.cumsum(logp, 1)
        base = cs.gather(1, (blk * gamma + gs).clamp(0, T - 1))
        r = torch.exp(cs - base) * mask.float()
        acc_sum += float((r.sum(1) / mask.sum(1).clamp(min=1)).mean()) * n
    model.train()
    return ce_sum / max(ntok, 1), hit / max(ntok, 1), acc_sum / max(ntok, 1)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def save_ckpt(model, tok, out_dir: str, meta: dict):
    base = unwrap(model)
    os.makedirs(out_dir, exist_ok=True)
    base.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=50_000_000)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--objective", default="ce", choices=["ce", "accept"])
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--ce-w", type=float, default=1.0)
    ap.add_argument("--accept-w", type=float, default=1.0)
    ap.add_argument("--save-every-tokens", type=int, default=10_000_000)
    ap.add_argument("--save-checkpoint-dirs", action="store_true",
                    help="save each interval as OUT/tokens-<N> instead of overwriting OUT")
    ap.add_argument("--val-every-tokens", type=int, default=5_000_000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dev = f"cuda:{local}"
        world = dist.get_world_size()
    else:
        rank, dev, world = 0, "cuda:0", 1
    torch.manual_seed(a.seed + rank)

    rows = load_rows(a.data, a.seq_len)
    if a.val:
        val = load_rows(a.val, a.seq_len)
    else:
        rng = random.Random(a.seed)
        rng.shuffle(rows)
        nval = max(1, min(512, len(rows) // 20))
        val, rows = rows[:nval], rows[nval:]
    log0(f"train rows={len(rows)} val={len(val)} out_tokens={sum(r['n_out'] for r in rows):,}")

    batches = bucket_batches(rows, a.batch_tokens, seed=a.seed)
    batches = [b for i, b in enumerate(batches) if i % world == rank]
    log0(f"batches/rank={len(batches)}")

    model = load_text_model(a.student, dtype=a.dtype, device_map=dev)
    tok = load_tokenizer(a.student)
    if a.grad_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    model.config.use_cache = False
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=False
        )

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=a.weight_decay, eps=1e-8)
    tokens_per_rank = a.max_tokens // world
    total_steps = max(1, math.ceil(tokens_per_rank / a.batch_tokens) * 2)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / max(a.warmup, 1)
        p = (s - a.warmup) / max(1, total_steps - a.warmup)
        return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join("out", "train", os.path.basename(a.out.rstrip("/")) + ".jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logf = open(log_path, "a") if rank == 0 else None

    seen = 0
    step = 0
    t0 = time.time()
    bi = 0
    last_val = 0
    last_save = 0
    bl = ba = br = 0.0
    rn = 0
    while seen < tokens_per_rank and bi < len(batches) * 100:
        if bi >= len(batches):
            bi = 0
            random.shuffle(batches)
        idxs = batches[bi]; bi += 1
        ids, att, labels = collate(rows, idxs, tok.pad_token_id, max_len=a.seq_len)
        ids, att, labels = ids.to(dev), att.to(dev), labels.to(dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        if a.objective == "ce":
            loss, hit, n = ce_step(model, ids, att, labels)
            acc_val = 0.0
        else:
            loss, hit, n, ce_v, acc_val = accept_step(
                model, ids, att, labels, a.gamma, a.ce_w, a.accept_w
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += n
        step += 1
        bl += float(loss); ba += hit / max(n, 1); br += acc_val; rn += 1

        if step % a.log_every == 0:
            msg = {"tokens": seen, "step": step, "loss": bl / rn, "token_acc": ba / rn,
                   "soft_accept": br / rn, "lr": lr_at(step), "elapsed": time.time() - t0}
            log0(json.dumps(msg))
            if logf:
                logf.write(json.dumps(msg) + "\n"); logf.flush()
            bl = ba = br = rn = 0

        if seen - last_val >= a.val_every_tokens:
            vl, va, vr = quick_val(unwrap(model), val, tok.pad_token_id, a.gamma,
                                   seq_len=a.seq_len)
            msg = {"tokens": seen, "val_ce": vl, "val_token_acc": va, "val_soft_accept": vr}
            log0("[val]", json.dumps(msg))
            if logf:
                logf.write(json.dumps({"event": "val", **msg}) + "\n"); logf.flush()
            last_val = seen

        if seen - last_save >= a.save_every_tokens:
            if rank == 0:
                dest = (os.path.join(a.out, f"tokens-{seen}") if a.save_checkpoint_dirs
                        else a.out)
                save_ckpt(model, tok, dest, {"tokens": seen, "step": step, "student": a.student,
                                             "objective": a.objective, "max_tokens": a.max_tokens})
            if ddp:
                dist.barrier()
            last_save = seen

    if rank == 0:
        dest = (os.path.join(a.out, f"tokens-{seen}") if a.save_checkpoint_dirs else a.out)
        save_ckpt(model, tok, dest, {"tokens": seen, "step": step, "student": a.student,
                                     "objective": a.objective, "max_tokens": a.max_tokens})
    log0(f"done tokens={seen:,} steps={step} elapsed={time.time()-t0:.0f}s")
    if logf:
        logf.close()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
