# 27B training: FLOPs, ZeRO-3 stack, and measured MFU

Last updated: 2026-11-04.  This supersedes the earlier FSDP-only notes.
The completed 20M run and its evaluation are in `docs/TRAIN_27B_RESULT.md`
(every held-out corpus gets worse; the OOD floor does not drop).

## 1. Correct parameter / FLOPs accounting

Read directly from the safetensors headers, excluding the vision tower and the
unused MTP head (`infra.flops`):

| component | params | share |
|---|---:|---:|
| MLP (64 layers) | 17.113B | 63.6% |
| linear attention (48 layers) | 5.562B | 20.7% |
| full attention (16 layers) | 1.678B | 6.2% |
| lm_head | 1.271B | 4.7% |
| embedding | 1.271B | 4.7% |
| norms | 0.0007B | 0.0% |
| **total text backbone** | **26.896B** | |
| of which matmul weights (`compute`) | 25.625B | |

The old estimate of 22.6B was wrong: it missed the untied lm_head and
undercounted the linear-attention projections (the config estimate used the
full-attention head counts for all layers).

Training FLOPs (fwd + bwd, causal):

- `6N` with `N = 25.625B` → **153.75 GFLOP/token**
- full-attention quadratic term: +0.6 GFLOP/token at seq 1024 (0.4%)
- **154.4 GFLOP/token at seq 1024**, 155.0 at seq 2048.

So MLP GEMMs dominate; attention-kernel work is secondary.  The old doc said
136 GFLOP/token; the corrected number is ~13% higher.

## 2. Training stack (DeepSpeed ZeRO-3)

`infra/train_zero3.py`, launched by `scripts/17_train27b_zero3.sh`.

Why not FSDP: the FSDP loop deadlocked in a small NCCL collective after 35–40
steps on the hybrid linear-attention model.  No FSDP configuration fixed it.

ZeRO-3 settings:

- stage 3, bf16, `overlap_comm`, `contiguous_gradients`;
- no parameter/optimizer offload (GPU memory fits);
- buckets reduced from DeepSpeed defaults to leave NCCL headroom:
  `reduce_bucket_size=2e8`, `stage3_prefetch_bucket_size=2e8`,
  `stage3_max_live_parameters=5e8`, `stage3_max_reuse_distance=1e8`,
  `stage3_param_persistence_threshold=1e5`;
- **all rows**, including the 2,000 raw FineWeb self-supervised sequences with
  `n_prompt=0` (2.05M of the 5.18M target tokens).  The inherited FSDP loop
  filtered `n_prompt>0` and silently dropped the raw-domain coverage arm;
- micro-batch 6 sequences, length-grouped, padded only to the batch max;
- gradient accumulation 1 (effective ~37k useful tokens per global step);
- activation checkpointing with `use_reentrant=False`;
- torch AdamW (`foreach=True`), cosine LR 5e-5, warmup 50 steps.

Two loading details matter:

1. `deepspeed.zero.Init` is **not** usable with transformers 5.x here: it
   creates `[0]`-shaped parameter placeholders and the strict loader raises
   `ignore_mismatched_sizes`.  Load on CPU first, then
   `deepspeed.initialize` partitions (`--no-zero-init`).
2. Call `deepspeed.init_distributed()` *before* building the torch process
   group, otherwise `zero.Init`/config asserts see `world_size=1`.

## 3. Where the time was going

### 3.1 Linear-attention fallback (largest single win)

`fla` / `causal-conv1d` were not installed, so transformers used
`torch_chunk_gated_delta_rule`: a Python loop over 64-token chunks with a
float32 upcast.  On 0.8B, one GPU, seq 1024:

| batch | torch fallback | fla Triton | speedup |
|---:|---:|---:|---:|
| 1 | 1,143 tok/s | 9,000 tok/s | 7.9x |
| 4 | 4,303 tok/s | 15,558 tok/s | 3.6x |

Profiling showed 4.5 s CPU vs 1.28 s CUDA over 3 iterations, 13,233 `bmm`
launches: the fallback is **launch-bound**, not FLOP-bound.  Installing
`flash-linear-attention==0.6.0` makes the model's
`use_kernel_func_from_hub_with_fallback` path pick up the Triton kernel
automatically; no model code changes.

### 3.2 Padding waste

The corpus averages 850 tokens/row (max 1355), but the FSDP loop padded every
sample to 2048 — more than 2x wasted GEMMs.  `train_zero3` sorts by length and
pads only to the batch max, so most batches are 6–8 x 1024 with almost no
padding.

### 3.3 Loss memory (27B OOM fix)

HF `ForCausalLMLoss` does `logits.float()` on `[B,T,248320]`: at 8x1024 that is
8 GB, plus cross-entropy buffers.  The 27B ZeRO-3 run OOM'd on step 2 at
~74 GB.  `FusedLMHeadCE` (`--fused-ce`, default on) computes the LM head and CE
in 512-token chunks, so only one `[512, 248320]` chunk is upcast at a time
(~0.5 GB).  Loss and gradients match the HF path (loss identical, grad rel.
diff 3e-5).  It is kept as an `nn.Module` so ZeRO-3 still gathers and
reduce-scatters `lm_head.weight` through its normal hooks.

## 4. Measured performance (8xA100-80G, 27B)

Per-GPU numbers from the training log; aggregate = 8x.

| micro-seqs | step time | useful tok/s/GPU | padded tok/s/GPU | nominal TFLOP/s | MFU | MFU incl. recompute | peak mem |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 (test) | 10.6 s | ~500 | ~610 | ~94 | 30% | 40% | 74.5 GB |
| 6 (test) | 7.5 s | ~520 | ~615 | ~95 | 30% | 40% | 71.3 GB |
| 6 (20M run) | 8.1 s | ~595 | ~655 | ~100 | 31–33% | 42–44% | 71.6 GB |

- Nominal MFU = model FLOPs / 312 TFLOP/s.  Activation checkpointing makes the
  hardware do one extra forward, so **hardware utilisation = 4/3 x nominal**;
  this is the real distance from the A100 peak.  Steady state: ~100 TFLOP/s
  per GPU nominal, ~133 TFLOP/s hardware, ~1.06 PFLOP/s aggregate hardware.
- Theoretical ceiling with full per-layer checkpointing is 75% nominal MFU
  (4 units of work per 3 units of model FLOPs), so 32% nominal is ~43% of the
  checkpointed ceiling.
- The 20M run finished 536 global steps in **4,383 s (73 min)** including the
  final 51 GB checkpoint export; each rank processed 2.51M useful tokens, i.e.
  **20.1M global**.  The old FSDP measurement was 19.5 s/step for 65,536 padded
  tokens (~1.9 h), so this is ~1.4x faster in padded-token terms and ~2x in
  useful-token terms after removing padding.
- Checkpoint exports take ~2 min each (51 GB, 12 shards).  Keep at most four
  of them on disk (3.2 TB free at the time of writing).

Remaining headroom, in rough order: ZeRO-3 all-gather traffic (~3x model size
per step), the 33% checkpoint recompute, and unfused elementwise ops
(RMSNorm/RoPE/SwiGLU).  `torch.compile` is the next thing to try, but it must
be validated against ZeRO-3 + variable sequence lengths before trusting it.

## 5. Known constraints

- Peak memory 71 GB (micro 6).  Micro 8 reaches 74.5 GB and occasionally fails
  20 MB `expandable_segments` mappings, though it completes.  Keep micro <= 6
  for long runs.
- Optimizer state is the floor: bf16 params 6.8 GB + grads 6.8 + fp32 master
  13.6 + Adam m/v 27.2 = 54.4 GB/GPU.  Bigger batches need offload or 8-bit
  Adam, not activation tricks.
- Checkpoints are exported to plain HF safetensors at 5M/10M/20M via
  `engine._zero3_consolidated_16bit_state_dict()`; export takes a few minutes
  and ~52 GB on rank 0.  Verified loadable with `AutoModelForCausalLM`.

## 6. Commands

```bash
# smoke test (no checkpoint)
torchrun --nproc_per_node 8 -m infra.train_zero3 \
  --student qwen3.5-27b --max-tokens 600000 --no-save \
  --no-zero-init --micro-seqs 6 --log-every 5

# full run
bash scripts/17_train27b_zero3.sh
# -> ckpt/zero3/qwen3.5-27b-mixed20m, log out/logs/zero3_27b_20m.log

# parameter / FLOPs table
python -m infra.flops /path/to/models/Qwen3.5-27B
```
