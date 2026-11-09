"""FSDP QAT prototype for the ORIGINAL teacher (Qwen3.6-35B-A3B).

NOTE: this prototype OOMs during FSDP init on 8xA100-80G: `sync_module_states`
broadcasts the full 70GB model to every rank before sharding.  It is kept for
reference only.  The working quick-QAT path is:

  python code/train_teacher_qat.py --pack_experts --steps 100 ...

which stores the routed experts as packed int4 and trains the always-active
parameters on one GPU.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

sys.path.insert(0, os.path.dirname(__file__))
from quant import apply_fake_quant
from quant_teacher import patch_moe_experts, TEACHER
from train_draft import load_rows, collate
from train_teacher_qat import is_trainable

D = "/path/to/spec-research/data"


def build_model(rank):
    from transformers import AutoConfig, AutoModelForCausalLM
    if rank == 0:
        return AutoModelForCausalLM.from_pretrained(
            TEACHER, dtype=torch.bfloat16, device_map="cpu",
            low_cpu_mem_usage=True, attn_implementation="sdpa")
    cfg = AutoConfig.from_pretrained(TEACHER)
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(tc)
    return m.to_empty(device="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{D}/distill_nothink.jsonl")
    ap.add_argument("--out", default="ckpt/teacher_qat_fsdp.pt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--lm_head_bits", type=int, default=4)
    ap.add_argument("--parts", default="router,shared,norm")
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    torch.manual_seed(0)

    rows = load_rows(a.data, max_len=a.max_len)
    my_idx = list(range(len(rows)))[rank::world]
    if rank == 0:
        print("train seqs", len(rows), "per-rank", len(my_idx), flush=True)

    t0 = time.time()
    model = build_model(rank)
    if rank == 0:
        print(f"model loaded {time.time()-t0:.0f}s", flush=True)
    apply_fake_quant(model, bits=a.bits, group=a.group, lm_head_bits=a.lm_head_bits)
    patch_moe_experts(a.bits, a.group, ste=True)
    parts = set(a.parts.split(","))
    for n, p in model.named_parameters():
        p.requires_grad_(is_trainable(n, parts))
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print("trainable", ntrain / 1e9, "B", flush=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()

    model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
                 device_id=local, sync_module_states=True,
                 use_orig_params=True, limit_all_gathers=True)
    params = [p for p in model.parameters() if p.requires_grad]
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
        idx = my_idx[step % len(my_idx)]
        ids, labels = collate(rows, [idx])
        ids, labels = ids.to(dev), labels.to(dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits.float()
        y = labels.clamp(min=0)
        mask = labels != -100
        ce = F.cross_entropy(logits[mask], y[mask])
        acc = (logits[mask].argmax(-1) == y[mask]).float().mean()
        ce.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        seen += int(mask.sum()) * world
        bl += float(ce); ba += float(acc); rn += 1
        if step % 10 == 0 and rank == 0:
            dt = time.time() - t0
            print(f"step {step}/{a.steps} ce {bl/rn:.4f} acc {ba/rn:.4f} "
                  f"tok/s {seen/dt:.0f}", flush=True)
            bl = ba = rn = 0.0

    # collect only the adapted params on rank 0 (all ranks must enter the context)
    state = {}
    with FSDP.summon_full_params(model, writeback=False, recurse=True,
                                 offload_to_cpu=True, rank0_only=True):
        if rank == 0:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    state[n] = p.detach().to(torch.bfloat16).cpu()
    if rank == 0:
        torch.save({"parts": sorted(parts), "bits": a.bits, "group": a.group,
                    "steps": a.steps, "train_tokens": seen, "state": state}, a.out)
        print("saved", a.out, len(state), "tensors", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
