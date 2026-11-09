"""LoRA + DDP fallback for 27B when FSDP is unstable.

The full-finetune FSDP path on this PyTorch/NCCL stack hits a deterministic
all-gather deadlock for Qwen3.5-27B.  This script keeps the base model frozen
(one replica per GPU) and trains LoRA adapters with DDP; only the small adapter
gradients are all-reduced, so there is no FSDP all-gather.
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

from infra.model_loader import load_text_model, load_tokenizer
from infra.train_distill import load_rows, bucket_batches
from infra.train_fsdp import collate_fixed


def log0(*a, **k):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*a, **k, flush=True)


def lr_at(step, warmup, total_steps, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="qwen3.5-27b")
    ap.add_argument("--data", default="data/processed/distill_mixed.jsonl")
    ap.add_argument("--out", default="ckpt/lora/qwen3.5-27b-mixed20m")
    ap.add_argument("--max-tokens", type=int, default=20_000_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-tokens", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--save-every-tokens", type=int, default=5_000_000)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpoint", action="store_false", dest="grad_checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    torch.manual_seed(a.seed + rank)
    os.makedirs(a.out, exist_ok=True)

    from peft import LoraConfig, get_peft_model

    rows = load_rows(a.data, a.seq_len)
    rows = [r for r in rows if r["n_prompt"] > 0]
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    nval = max(1, min(128, len(rows) // 50))
    val, rows = rows[:nval], rows[nval:]
    batches = bucket_batches(rows, a.batch_tokens, seed=a.seed)
    batches = [b for i, b in enumerate(batches) if i % world == rank]
    log0(f"rank={rank} rows={len(rows)} batches/rank={len(batches)}")

    model = load_text_model(a.student, dtype="bfloat16", device_map=dev)
    tok = load_tokenizer(a.student)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.0,
                     target_modules="all-linear", bias="none",
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    if a.grad_checkpoint:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local], find_unused_parameters=False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.0, eps=1e-8)
    log0("trainable", sum(p.numel() for p in params) / 1e6, "M")

    tokens_per_rank = a.max_tokens // world
    total_steps = max(1, math.ceil(tokens_per_rank / a.batch_tokens))
    seen = step = bi = 0
    last_save = 0
    t0 = time.time()
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
        bl += float(loss.detach()); rn += 1
        if step % a.log_every == 0:
            dt = time.time() - t0
            log0(json.dumps({"tokens": seen, "step": step, "loss": bl / rn,
                             "tokens_per_s": seen / max(dt, 1e-9),
                             "step_time_s": dt / step}))
            bl = rn = 0.0
        if not a.no_save and seen - last_save >= a.save_every_tokens:
            if rank == 0:
                model.module.save_pretrained(a.out)
                tok.save_pretrained(a.out)
                with open(os.path.join(a.out, "train_meta.json"), "w") as f:
                    json.dump({"tokens": seen, "step": step}, f, indent=1)
            dist.barrier()
            last_save = seen
    if not a.no_save:
        if rank == 0:
            model.module.save_pretrained(a.out)
            tok.save_pretrained(a.out)
            with open(os.path.join(a.out, "train_meta.json"), "w") as f:
                json.dump({"tokens": seen, "step": step}, f, indent=1)
    log0(f"done tokens={seen:,} steps={step} elapsed={time.time()-t0:.0f}s")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
