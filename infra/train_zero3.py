"""DeepSpeed ZeRO-3 full-fine-tune trainer for Qwen3.5 (0.8B ... 27B).

Why this exists
---------------
The FSDP trainer (``infra/train_fsdp.py``) deadlocked in a small NCCL collective
after ~35-40 steps on the Qwen3.5-27B hybrid linear-attention model.  This
trainer uses DeepSpeed ZeRO-3 instead, and fixes the biggest throughput bug in
the FSDP loop: every sample was padded to 2048 tokens even though the corpus
averages ~850 tokens (max 1355).  Here batches are length-grouped and padded
only to the longest member, which nearly doubles useful tokens/s.

Throughput accounting
---------------------
Every log line reports the nominal model FLOPs (``6N`` for matmul weights plus
the full-attention quadratic term, see ``infra.flops``), so

    mfu = achieved_model_TFLOP/s / 312

is a standard A100-bf16 MFU number.  With activation checkpointing the hardware
does one extra forward, so the ``mfu_hw`` field multiplies the nominal FLOPs by
4/3; that is the true distance from the 312 TFLOP/s peak.

Usage
-----
  # smoke test (no checkpoint)
  torchrun --nproc_per_node 8 -m infra.train_zero3 \
      --student qwen3.5-0.8b --max-tokens 200000 --no-save

  # 27B, 20M tokens, checkpoints at 5/10/20M
  torchrun --nproc_per_node 8 -m infra.train_zero3 \
      --student qwen3.5-27b --data data/processed/distill_mixed.jsonl \
      --out ckpt/zero3/qwen3.5-27b-mixed20m --max-tokens 20000000

The data format is the same as ``infra.train_distill`` (input_ids/output_ids).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import deepspeed

from infra.flops import (A100_BF16_PEAK_TFLOPS, flops_per_token, param_stats,
                         _config)
from infra.model_loader import load_text_model, load_tokenizer
from infra.train_distill import load_rows


def log0(*a, **k):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*a, **k, flush=True)


def log_json(obj):
    if int(os.environ.get("RANK", 0)) == 0:
        print("@@ " + json.dumps(obj), flush=True)


def lr_at(step: int, warmup: int, total_steps: int, base_lr: float, min_ratio: float = 0.05) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def make_chunks(rows: list[dict], micro_seqs: int) -> list[list[int]]:
    """Length-grouped, fixed-size batches (no token-budget slack).

    Sorting by length first and then slicing into fixed-size chunks means most
    chunks contain eight ~1024-token samples: almost no padding, and a stable
    micro-batch size for DeepSpeed bookkeeping.
    """
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]))
    return [order[i:i + micro_seqs] for i in range(0, len(order) - micro_seqs + 1, micro_seqs)]


def collate_dynamic(rows: list[dict], idxs: list[int], pad: int, max_len: int):
    L = min(max(len(rows[i]["ids"]) for i in idxs), max_len)
    ids = torch.full((len(idxs), L), pad, dtype=torch.long)
    labels = torch.full((len(idxs), L), -100, dtype=torch.long)
    att = torch.zeros((len(idxs), L), dtype=torch.long)
    for b, i in enumerate(idxs):
        s = torch.tensor(rows[i]["ids"][:L], dtype=torch.long)
        ids[b, :len(s)] = s
        att[b, :len(s)] = 1
        labels[b, rows[i]["n_prompt"]:len(s)] = s[rows[i]["n_prompt"]:]
    return ids, att, labels


def build_ds_config(a, world: int, total_steps: int) -> dict:
    zero = {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": a.reduce_bucket,
        "stage3_prefetch_bucket_size": a.prefetch_bucket,
        "stage3_param_persistence_threshold": a.param_persistence,
        "stage3_max_live_parameters": a.max_live_params,
        "stage3_max_reuse_distance": a.max_reuse_distance,
        "stage3_gather_16bit_weights_on_model_save": True,
        "sub_group_size": 1e9,
        "offload_optimizer": {
            "device": "cpu" if a.offload_optimizer else "none",
            "pin_memory": True,
        },
        "offload_param": {"device": "none"},
    }
    opt_params = {
        "lr": a.lr,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": a.weight_decay,
    }
    if not a.fused_adam:
        # torch's fused/foreach AdamW avoids a DeepSpeed JIT build and is fast
        # enough; the optimizer is <10% of step time at 27B.
        opt_params["torch_adam"] = True
        opt_params["foreach"] = True
    return {
        "train_batch_size": a.micro_seqs * world * a.grad_accum,
        "train_micro_batch_size_per_gpu": a.micro_seqs,
        "gradient_accumulation_steps": a.grad_accum,
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "optimizer": {"type": "AdamW", "params": opt_params},
        "zero_optimization": zero,
        "gradient_clipping": 1.0,
        "steps_per_print": 10**9,
        "wall_clock_breakdown": False,
    }


class FusedLMHeadCE(nn.Module):
    """LM head that computes CE in token chunks instead of materialising logits.

    HF's ``ForCausalLMLoss`` does ``logits.float()`` on the full
    ``[B, T, 248320]`` tensor (~8 GB at 8x1024) and cross-entropy allocates
    another copy.  Here each chunk is upcast separately, so peak loss memory is
    one chunk (~1 GB).  Kept as a module so ZeRO-3 still sees the weight and
    gathers / reduce-scatters it through its normal hooks.
    """

    def __init__(self, weight: nn.Parameter, chunk: int = 1024):
        super().__init__()
        self.weight = weight
        self.chunk = chunk

    def forward(self, hidden, labels=None):
        if labels is None:
            return F.linear(hidden, self.weight)
        h = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])
        y = labels[:, 1:].reshape(-1)
        n_valid = (y != -100).sum().clamp(min=1)
        total = h.new_zeros((), dtype=torch.float32)
        for s in range(0, h.shape[0], self.chunk):
            hc = h[s:s + self.chunk]
            yc = y[s:s + self.chunk]
            if not bool((yc != -100).any()):
                continue
            logits = F.linear(hc, self.weight).float()
            total = total + F.cross_entropy(logits, yc, ignore_index=-100,
                                            reduction="sum")
        return total / n_valid


def install_fused_head(model, chunk: int = 1024):
    """Replace lm_head + loss with the chunked-CE path (in place)."""

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         use_cache=False)
        hidden = out.last_hidden_state
        if labels is None:
            return SimpleNamespace(logits=self.lm_head(hidden))
        return SimpleNamespace(loss=self.lm_head(hidden, labels))

    model.lm_head = FusedLMHeadCE(model.lm_head.weight, chunk=chunk)
    model.forward = MethodType(forward, model)
    return model


def save_hf(engine, model, tok, out_dir: str, meta: dict):
    """Consolidate ZeRO-3 shards to a regular HF checkpoint on rank 0.

    Each milestone gets its own ``tokens-<N>`` directory; saving all of them to
    the run directory would overwrite the 5M/10M checkpoints with the final one.
    """
    target = os.path.join(out_dir, f"tokens-{meta['tokens']}")
    sd = engine._zero3_consolidated_16bit_state_dict()
    dist.barrier()
    if dist.get_rank() == 0:
        os.makedirs(target, exist_ok=True)
        model.save_pretrained(target, state_dict=sd, safe_serialization=True,
                              max_shard_size="5GB")
        tok.save_pretrained(target)
        with open(os.path.join(target, "train_meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        log0(f"saved {target} tokens={meta['tokens']:,}")
    dist.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="qwen3.5-27b")
    ap.add_argument("--data", default="data/processed/distill_mixed.jsonl")
    ap.add_argument("--out", default="ckpt/zero3/qwen3.5-27b-mixed20m")
    ap.add_argument("--max-tokens", type=int, default=20_000_000,
                    help="useful (non-padding) target tokens to train on")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-seqs", type=int, default=8,
                    help="sequences per GPU per micro-batch")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--save-every-tokens", type=int, default=5_000_000)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--offload-optimizer", action="store_true")
    ap.add_argument("--fused-adam", action="store_true")
    ap.add_argument("--zero-init", action="store_true", default=True)
    ap.add_argument("--no-zero-init", action="store_false", dest="zero_init")
    ap.add_argument("--fused-ce", action="store_true", default=True,
                    help="chunked LM head + cross-entropy (no full fp32 logits)")
    ap.add_argument("--no-fused-ce", action="store_false", dest="fused_ce")
    ap.add_argument("--ce-chunk", type=int, default=1024)
    # ZeRO-3 buffers: the 27B run peaks at ~75/80 GB with DeepSpeed defaults,
    # so keep these below the defaults to leave headroom for NCCL workspaces.
    ap.add_argument("--reduce-bucket", type=float, default=2e8)
    ap.add_argument("--prefetch-bucket", type=float, default=2e8)
    ap.add_argument("--max-live-params", type=float, default=5e8)
    ap.add_argument("--max-reuse-distance", type=float, default=1e8)
    ap.add_argument("--param-persistence", type=float, default=1e5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # DeepSpeed keeps its own comm wrapper; ``zero.Init`` only sees the world
    # size after ``deepspeed.init_distributed`` (a bare torch init leaves
    # deepspeed.comm uninitialised and it silently falls back to world_size=1).
    deepspeed.init_distributed(dist_backend="nccl")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    torch.manual_seed(a.seed + rank)
    os.makedirs(a.out, exist_ok=True)

    # Keep self-supervised rows too: distill_mixed.jsonl contains 2,000 raw
    # FineWeb sequences with n_prompt=0 (2.05M of 5.18M target tokens).  The
    # old FSDP loop filtered them out, which silently removed the raw-domain
    # coverage arm from the training mix.
    rows = load_rows(a.data, a.seq_len)
    chunks = make_chunks(rows, a.micro_seqs)
    # Drop the tail so every rank wraps epochs on the same global step.
    usable = len(chunks) - (len(chunks) % (world * a.grad_accum))
    chunks = chunks[:usable]
    useful_per_chunk = [sum(rows[i]["n_out"] for i in c) for c in chunks]
    padded_per_chunk = [max(len(rows[i]["ids"]) for i in c) * len(c) for c in chunks]
    avg_useful = sum(useful_per_chunk) / len(chunks)
    tokens_per_global_step = avg_useful * world * a.grad_accum
    total_steps = max(1, math.ceil(a.max_tokens / tokens_per_global_step))
    log0(f"rank={rank} rows={len(rows)} chunks={len(chunks)} "
         f"avg_useful/chunk={avg_useful:.0f} p50_len={sorted(len(r['ids']) for r in rows)[len(rows)//2]} "
         f"global_tokens/step={tokens_per_global_step:.0f} total_steps={total_steps}")

    stats = param_stats(_resolve_model_path(a.student))
    cfg = _config(_resolve_model_path(a.student))
    log0(f"params total={stats.total/1e9:.3f}B compute={stats.compute/1e9:.3f}B "
         f"embed={stats.embed/1e9:.3f}B lm_head={stats.lm_head/1e9:.3f}B")

    ds_config = build_ds_config(a, world, total_steps)
    t_load = time.time()
    if a.zero_init:
        with deepspeed.zero.Init(config_dict_or_path=ds_config,
                                 mem_efficient_linear=False):
            model = load_text_model(a.student, dtype="bfloat16", device_map=None)
    else:
        model = load_text_model(a.student, dtype="bfloat16", device_map=None)
    tok = load_tokenizer(a.student)
    if not a.no_grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if a.fused_ce:
        install_fused_head(model, chunk=a.ce_chunk)
        log0(f"fused chunked CE enabled (chunk={a.ce_chunk})")
    model.train()
    log0(f"model loaded in {time.time()-t_load:.0f}s")

    ds_args = SimpleNamespace(
        train_batch_size=a.micro_seqs * world * a.grad_accum,
        train_micro_batch_size_per_gpu=a.micro_seqs,
        gradient_accumulation_steps=a.grad_accum,
        local_rank=local,
    )
    engine, optimizer, _, _ = deepspeed.initialize(
        args=ds_args, model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=ds_config,
    )
    engine.train()

    fpt_cache: dict[int, float] = {}

    def flops_for_seq(s: int) -> float:
        if s not in fpt_cache:
            fpt_cache[s] = flops_per_token(stats, cfg, s)
        return fpt_cache[s]

    pad_id = tok.pad_token_id
    # DeepSpeed's step counts are global, but the loop only sees this rank's
    # tokens.  Track local counts for the FLOPs window and multiply by world
    # for token budgets / checkpoint milestones (chunks are round-robin, so the
    # ranks are balanced to within one chunk).
    seen_useful = 0
    seen_padded = 0
    flops_nominal = 0.0
    flops_hw_factor = 1.0 if a.no_grad_checkpoint else (8.0 / 6.0)
    step = 0
    micro_global = 0
    next_save = a.save_every_tokens
    t0 = time.time()
    t_last = t0
    loss_sum, loss_n = 0.0, 0
    useful_at_last = 0
    padded_at_last = 0
    flops_at_last = 0.0
    steps_at_last = 0

    def chunk_at(k: int):
        epoch, idx = divmod(k, len(chunks))
        order = list(range(len(chunks)))
        random.Random(a.seed * 100003 + epoch).shuffle(order)
        return chunks[order[idx]]

    while step < total_steps:
        lr = lr_at(step, a.warmup, total_steps, a.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        for _ in range(a.grad_accum):
            idxs = chunk_at(micro_global)
            micro_global += 1
            ids, att, labels = collate_dynamic(rows, idxs, pad_id, a.seq_len)
            ids, att, labels = ids.to(dev), att.to(dev), labels.to(dev)
            out = engine(input_ids=ids, attention_mask=att, labels=labels, use_cache=False)
            loss = out.loss / a.grad_accum
            engine.backward(loss)
            n_useful = int((labels[:, 1:] != -100).sum())
            n_padded = ids.numel()
            seen_useful += n_useful
            seen_padded += n_padded
            flops_nominal += flops_for_seq(min(a.seq_len, ids.shape[1])) * n_padded
            loss_sum += float(out.loss.detach()); loss_n += 1
        engine.step()
        step += 1

        if step % a.log_every == 0 or step == 1:
            torch.cuda.synchronize()
            now = time.time()
            dt = now - t_last
            d_steps = step - steps_at_last
            d_useful = seen_useful - useful_at_last
            d_pad = seen_padded - padded_at_last
            d_flops = flops_nominal - flops_at_last
            tflops = d_flops / dt / 1e12
            seen_global = seen_useful * world
            msg = {
                "step": step,
                "total_steps": total_steps,
                "tokens": seen_global,
                "pct": round(100 * seen_global / a.max_tokens, 1),
                "loss": round(loss_sum / max(loss_n, 1), 4),
                "lr": lr,
                "step_time_s": round(dt / d_steps, 3),
                "interval_tokens_per_s": round(d_useful * world / dt, 1),
                "padded_tokens_per_s": round(d_pad * world / dt, 1),
                "tokens_per_s": round(seen_global / max(now - t0, 1e-9), 1),
                "tflops_per_gpu": round(tflops, 1),
                "mfu": round(tflops / A100_BF16_PEAK_TFLOPS, 4),
                "tflops_hw": round(tflops * flops_hw_factor, 1),
                "mfu_hw": round(tflops * flops_hw_factor / A100_BF16_PEAK_TFLOPS, 4),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
            }
            log_json(msg)
            t_last = now
            steps_at_last = step
            useful_at_last = seen_useful
            padded_at_last = seen_padded
            flops_at_last = flops_nominal
            loss_sum, loss_n = 0.0, 0

        seen_global = seen_useful * world
        if not a.no_save and seen_global >= next_save:
            meta = {"tokens": seen_global, "tokens_local": seen_useful,
                    "padded_tokens": seen_padded * world, "step": step,
                    "student": a.student, "max_tokens": a.max_tokens,
                    "world": world, "micro_seqs": a.micro_seqs,
                    "grad_accum": a.grad_accum, "seq_len": a.seq_len,
                    "global_batch_tokens": round(tokens_per_global_step)}
            save_hf(engine, model, tok, a.out, meta)
            next_save += a.save_every_tokens

    if not a.no_save:
        seen_global = seen_useful * world
        meta = {"tokens": seen_global, "tokens_local": seen_useful,
                "padded_tokens": seen_padded * world, "step": step,
                "student": a.student, "max_tokens": a.max_tokens,
                "world": world, "micro_seqs": a.micro_seqs,
                "grad_accum": a.grad_accum, "seq_len": a.seq_len,
                "global_batch_tokens": round(tokens_per_global_step)}
        save_hf(engine, model, tok, a.out, meta)

    elapsed = time.time() - t0
    seen_global = seen_useful * world
    log0(f"done useful_tokens={seen_global:,} (global) padded={seen_padded*world:,} "
         f"steps={step} elapsed={elapsed:.0f}s tokens/s={seen_global/elapsed:.0f}")
    dist.destroy_process_group()


def _resolve_model_path(key: str) -> str:
    from infra import model_registry
    return model_registry.resolve(key)


if __name__ == "__main__":
    main()
