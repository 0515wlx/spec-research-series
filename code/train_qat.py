"""Short QAT fine-tune of the dense draft with weight-only fake quantization.

Start from a distilled dense draft, replace large Linear layers with QLinear
(group-wise symmetric W4/W8 + STE), fine-tune with the same hard-label CE (or
CE+acceptance) objective, then save a plain dequantized checkpoint for eval.
"""
import argparse, json, math, os, random, sys, time
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
from draft import load_dense, save_dense
from quant import (apply_fake_quant, dequantize_inplace, quant_stats,
                   refresh_fake_quant, clear_fake_quant)
from train_draft import (D, load_rows, bucket_batches, collate, ce_loss_batched,
                         rl_loss_batched, quick_val, log0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/dense_distill_s1")
    ap.add_argument("--out", default="ckpt/dense_qat_w4")
    ap.add_argument("--data", default=f"{D}/distill_nothink.jsonl")
    ap.add_argument("--val", default=f"{D}/teacher_math500_nothink.jsonl")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_tok", type=int, default=6144)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--ce_w", type=float, default=1.0)
    ap.add_argument("--rl_w", type=float, default=0.0)
    ap.add_argument("--mb", type=int, default=2)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--lm_head_bits", type=int, default=4)
    ap.add_argument("--skip_lm_head", action="store_true")
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
    names = apply_fake_quant(model, bits=a.bits, group=a.group,
                             skip_lm_head=a.skip_lm_head, lm_head_bits=a.lm_head_bits)
    log0("quantized modules", len(names), "sizes", quant_stats(model))
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    log0("trainable", sum(p.numel() for p in params) / 1e9, "B")

    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, total - a.warmup)
        return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    def sync_grads():
        if not ddp:
            return
        ws = dist.get_world_size()
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= ws

    t0 = time.time(); seen = 0
    bl = ba = br = 0.0; rn = 0; bi = 0
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
        refresh_fake_quant(model)          # one quantize pass per optimizer step
        if a.rl_w > 0:
            ce, acc, ntok, rl = rl_loss_batched(model, ids, labels, gamma=a.gamma,
                                                ce_w=a.ce_w, rl_w=a.rl_w, mb=a.mb)
        else:
            ce, acc, ntok = ce_loss_batched(model, ids, labels)
            rl = 0.0
        sync_grads()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        clear_fake_quant(model)
        seen += ntok * (dist.get_world_size() if ddp else 1)
        bl += ce; ba += acc; br += rl; rn += 1
        if step % 10 == 0:
            dt = time.time() - t0
            log0(f"step {step}/{total} ce {bl/rn:.4f} acc {ba/rn:.4f} rl {br/rn:.4f} "
                 f"tok/s {seen/dt:.0f} lr {lr_at(step):.2e}")
            bl = ba = br = rn = 0.0
        if (step + 1) % 50 == 0:
            refresh_fake_quant(model)
            vl, va, vr = quick_val(model, val, gamma=a.gamma)
            clear_fake_quant(model)
            log0(f"[val] step {step+1} ce {vl:.4f} token_acc {va:.4f} soft_tau_frac {vr:.4f}")

    # save dequantized bf16 weights so the normal dense loader / evaluator works
    if ddp:
        dist.barrier()
    refresh_fake_quant(model)
    vl, va, vr = quick_val(model, val, gamma=a.gamma)
    clear_fake_quant(model)
    log0(f"[final val] ce {vl:.4f} token_acc {va:.4f} soft_tau_frac {vr:.4f}")
    if rank == 0:
        n = dequantize_inplace(model)
        meta2 = dict(meta)
        meta2.update({"init": meta.get("init", "") + "+qat", "qat_bits": a.bits,
                      "qat_group": a.group, "qat_lm_head_bits": a.lm_head_bits,
                      "qat_steps": total, "qat_tokens": seen,
                      "qat_val_token_acc": va, "dequantized_modules": n})
        save_dense(model, a.out, meta2)
    log0("done", seen, "tokens", time.time() - t0)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
