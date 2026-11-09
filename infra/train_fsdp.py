"""FSDP full-fine-tune trainer for Qwen3.5-27B (and smaller models).

Key choices for 27B on 8xA100-80G:
  * FULL_SHARD FSDP, bf16 params/reduce, activation checkpointing;
  * per-GPU batch-tokens 8192 (4 x 2048) to amortize all-gather/reduce-scatter;
  * forward_prefetch + BACKWARD_PRE prefetch + limit_all_gathers;
  * sync_module_states=False (the old prototype OOM'd because it broadcast the
    full 70GB model to every rank before sharding);
  * each rank loads the checkpoint on CPU, then FSDP shards it.

This is intentionally a training-loop, not a general framework.  It expects the
same JSONL format as ``infra.train_distill`` (input_ids / output_ids).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

from infra.model_loader import load_text_model, load_tokenizer
from infra.train_distill import load_rows, bucket_batches


def log0(*a, **k):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*a, **k, flush=True)


def collate_fixed(rows, idxs, pad, seq_len):
    """Pad every batch to the same sequence length on every rank.

    Qwen3.5's hybrid linear-attention path is sensitive to sequence length;
    different per-rank max lengths can make FSDP ranks execute different
    parameter paths and deadlock.  Fixed padding keeps the collective order
    identical across ranks.
    """
    ids = torch.full((len(idxs), seq_len), pad, dtype=torch.long)
    labels = torch.full((len(idxs), seq_len), -100, dtype=torch.long)
    att = torch.zeros((len(idxs), seq_len), dtype=torch.long)
    for b, i in enumerate(idxs):
        s = torch.tensor(rows[i]["ids"][:seq_len], dtype=torch.long)
        ids[b, :len(s)] = s
        att[b, :len(s)] = 1
        labels[b, rows[i]["n_prompt"]:len(s)] = s[rows[i]["n_prompt"]:]
    return ids, att, labels


def lr_at(step, warmup, total_steps, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def save_ckpt(model, tok, out_dir, meta):
    with FSDP.summon_full_params(model, writeback=False, recurse=True,
                                 offload_to_cpu=True, rank0_only=True):
        if int(os.environ.get("RANK", 0)) == 0:
            os.makedirs(out_dir, exist_ok=True)
            model.module.save_pretrained(out_dir, safe_serialization=True,
                                         max_shard_size="5GB")
            tok.save_pretrained(out_dir)
            with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
                json.dump(meta, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="qwen3.5-27b")
    ap.add_argument("--data", default="data/processed/distill_mixed.jsonl")
    ap.add_argument("--out", default="ckpt/fsdp/qwen3.5-27b-mixed20m")
    ap.add_argument("--max-tokens", type=int, default=20_000_000)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--save-every-tokens", type=int, default=5_000_000)
    ap.add_argument("--no-save", action="store_true", help="skip checkpoint writing (smoke test)")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpoint", action="store_false", dest="grad_checkpoint")
    ap.add_argument("--optim", default="adamw", choices=["adamw", "adafactor"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    torch.manual_seed(a.seed + rank)
    os.makedirs(a.out, exist_ok=True)

    rows = load_rows(a.data, a.seq_len)
    rows = [r for r in rows if r["n_prompt"] > 0]
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    nval = max(1, min(128, len(rows) // 50))
    val, rows = rows[:nval], rows[nval:]
    batches = bucket_batches(rows, a.batch_tokens, seed=a.seed)
    batches = [b for i, b in enumerate(batches) if i % world == rank]
    log0(f"rank={rank} rows={len(rows)} batches/rank={len(batches)}")

    # Each rank loads on CPU; FSDP then shards.  sync_module_states is disabled.
    model = load_text_model(a.student, dtype=a.dtype, device_map="cpu")
    tok = load_tokenizer(a.student)
    if a.grad_checkpoint:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    mp = MixedPrecision(param_dtype=torch.bfloat16,
                        reduce_dtype=torch.bfloat16,
                        buffer_dtype=torch.bfloat16)
    layer_cls = type(model.model.layers[0])
    import torch.nn as nn

    def auto_wrap(module, recurse, nonwrapped_numel):
        if isinstance(module, layer_cls):
            return True
        if isinstance(module, nn.Embedding) and module.weight.numel() >= 1_000_000_000:
            return True
        if isinstance(module, nn.Linear) and module.weight.numel() >= 1_000_000_000:
            return True
        return False
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local,
        sync_module_states=False,
        use_orig_params=True,
        limit_all_gathers=True,
        forward_prefetch=False,
        backward_prefetch=None,
        mixed_precision=mp,
        auto_wrap_policy=auto_wrap,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    if a.optim == "adafactor":
        from transformers.optimization import Adafactor
        opt = Adafactor(params, lr=a.lr, scale_parameter=False,
                        relative_step=False, warmup_init=False,
                        weight_decay=a.weight_decay)
    else:
        try:
            opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95),
                                    weight_decay=a.weight_decay, eps=1e-8, fused=True)
        except TypeError:
            opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95),
                                    weight_decay=a.weight_decay, eps=1e-8)

    tokens_per_rank = a.max_tokens // world
    total_steps = max(1, math.ceil(tokens_per_rank / a.batch_tokens))
    log0(f"tokens/rank={tokens_per_rank:,} total_steps={total_steps} "
         f"batch_tokens/rank={a.batch_tokens}")

    seen = 0
    step = 0
    bi = 0
    t0 = time.time()
    last_save = 0
    bl = rn = 0.0
    while seen < tokens_per_rank:
        if bi >= len(batches):
            bi = 0
            random.shuffle(batches)
        idxs = batches[bi]; bi += 1
        ids, att, labels = collate_fixed(rows, idxs, tok.pad_token_id, a.seq_len)
        ids, att, labels = ids.to(dev), att.to(dev), labels.to(dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.warmup, total_steps, a.lr)
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=ids, attention_mask=att, labels=labels, use_cache=False)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        n = int((labels[:, 1:] != -100).sum())
        seen += n * world
        step += 1
        bl += float(loss); rn += 1
        if step % a.log_every == 0:
            dt = time.time() - t0
            msg = {"tokens": seen, "step": step, "loss": bl / rn,
                   "tokens_per_s": seen / max(dt, 1e-9),
                   "step_time_s": dt / step, "lr": lr_at(step, a.warmup, total_steps, a.lr)}
            log0(json.dumps(msg))
            bl = rn = 0.0
        if not a.no_save and seen - last_save >= a.save_every_tokens:
            save_ckpt(model, tok, a.out,
                      {"tokens": seen, "step": step, "student": a.student,
                       "max_tokens": a.max_tokens})
            dist.barrier()
            last_save = seen

    if not a.no_save:
        save_ckpt(model, tok, a.out,
                  {"tokens": seen, "step": step, "student": a.student,
                   "max_tokens": a.max_tokens})
    log0(f"done tokens={seen:,} steps={step} elapsed={time.time()-t0:.0f}s")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
