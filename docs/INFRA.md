# Infrastructure

> Everything needed to run the Route A experiments on this machine.
> Hardware: 8×A100-80G. No HuggingFace proxy; use ModelScope.

## 1. Environment

| item | version / note |
|---|---|
| Python | 3.12 |
| PyTorch | 2.13.0 (CUDA) |
| transformers | 5.15.0 (supports `Qwen3_5ForCausalLM`) |
| lm_eval | 0.4.13 |
| modelscope | 1.39.1 |
| accelerate | 1.15.0 |
| scipy / matplotlib / pandas / pyarrow | installed |

Check:

```bash
python -m infra.env_check --json out/env.json
```

The check verifies GPUs, disk, packages, model presence, tokenizer consistency, and ModelScope
reachability.

## 2. Models: Qwen3.5 series

The local checkpoints are **multimodal** (`Qwen3_5ForConditionalGeneration`, with a vision tower).
For text-only SD experiments we load the text backbone directly:

```python
from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained(path)
text_cfg = getattr(cfg, "text_config", cfg)
model = AutoModelForCausalLM.from_pretrained(
    path, config=text_cfg, dtype=torch.bfloat16, device_map="auto",
    low_cpu_mem_usage=True,
)
```

This skips the vision weights and gives `Qwen3_5ForCausalLM` (or `Qwen3_5MoeForCausalLM` for the MoE).
Verified on `Qwen3.5-4B` (loads 426 text tensors, forward logits shape `[1, T, 248320]`).

**Throughput tip:** for the 35B target use `device_map="balanced"` rather than `"auto"`. On this
machine `auto` put all 72GB on one A100 (slow, ~140s load); `balanced` shards across all 8 GPUs
(~14s load) and generated 4×128 tokens in ~13s. Drafts can use `auto`.

**Why this matters:** `AutoModelForCausalLM.from_pretrained(path)` without `config=text_config` tries
to instantiate the conditional-generation class and may fail or load the vision tower. Always use the
wrapper in `infra/model_loader.py`.

### Model registry

`infra/model_registry.py` lists the capacity series:

| key | path | params | kind |
|---|---|---|---|
| `qwen3.5-0.8b` | `/path/to/models/Qwen3.5-0.8B` | 0.8B | dense |
| `qwen3.5-2b` | `/path/to/models/Qwen3.5-2B` | 2B | dense |
| `qwen3.5-4b` | `/path/to/models/Qwen3.5-4B` | 4B | dense |
| `qwen3.5-9b` | `/path/to/models/Qwen3.5-9B` | 9B | dense |
| `qwen3.5-27b` | `/path/to/models/Qwen3.5-27B` | 27B | dense |
| `qwen3.5-35b-a3b` | `/path/to/models/Qwen3.5-35B-A3B` | 35B / 3B active | MoE |

### Download (ModelScope + HF mirror fallback)

```bash
python -m infra.download --models qwen3.5-0.8b qwen3.5-2b qwen3.5-27b qwen3.5-35b-a3b --retry 10
```

The script wraps `modelscope download --local-dir` and, with `--backend auto`, falls back to
the HF mirror. On 2026-11-02 the ModelScope `/repo/files` endpoint returned HTTP 500
(`unihub ... 401 ... user deletion progress`) for all models. The HF mirror works, but the
default Xet CAS path returns 401; disable Xet:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
hf download Qwen/Qwen3.5-2B --local-dir /path/to/models/Qwen3.5-2B --max-workers 8
```

`infra.download` sets both variables for the HF backend. The 4B/9B are already present.

### Text-only export (for lm-eval / vLLM / quantization)

Some tools do not understand the multimodal checkpoint. Export a text-only HF checkpoint:

```bash
python -m infra.export_text_model --model qwen3.5-9b --out ckpt/text/qwen3.5-9b
```

This writes `config.json`, `model.safetensors` (sharded if large), tokenizer files, and a
`README_TEXT_ONLY.md`. Point `lm-eval` / vLLM at this directory.

## 3. Data

### Sources (ModelScope; HF is blocked)

| domain | source | notes |
|---|---|---|
| `math500` | `HuggingFaceH4/MATH-500` | 500 test problems |
| `gsm8k` | `opencompass/gsm8k` | test split; use first 500 |
| `humaneval` | `opencompass/humaneval` | 164 problems |
| `wikitext` | `Salesforce/wikitext` | local cache: `~/ .cache/modelscope/datasets/Salesforce--wikitext/snapshots/master/wikitext-2-raw-v1/test-00000-of-00001.parquet` |
| `mtbench` | `lmsys/mt_bench_human_judgments` | use first-turn prompts from `gpt4_pair` |
| `nq` | `google-research-datasets/natural_questions` | optional |

`infra/prepare_data.py` normalizes each source to a JSONL with:

```json
{"id": "...", "domain": "...", "prompt": "...", "reference": "...", "meta": {...}}
```

For `modelscope/gsm8k` and `modelscope/humaneval`, `MsDataset.load` needs `trust_remote_code=True`.
`opencompass/gsm8k` and `opencompass/humaneval` work without it; prefer those.

### Target generation

The acceptance evaluator scores drafts on **target-generated** sequences, not on reference answers.
Generate them once per target/domain:

```bash
python -m infra.generate_target --target qwen3.5-35b-a3b --domain math500 --limit 500 \
  --max-new 512 --batch-size 8 --device-map auto --out data/processed/math500_target.jsonl
```

Record format: `input_ids`, `output_ids`, `prompt`, `text`, `meta` (target, max_new, thinking,
tokenizer hash). Keep `output_ids` unpadded (strip pad/eos as appropriate).

### Chat template / thinking

Qwen3.5 thinks by default. The chat template may support `enable_thinking=False`; check
`tokenizer.apply_chat_template(..., enable_thinking=False)`. If not, prepend a fixed empty-think block
and use the same convention for every draft. Record the template hash in `meta`. For WikiText use raw
continuation (no chat template).

## 4. Acceptance evaluation (teacher forcing)

`infra/accept_eval.py` is the core measurement:

- loads the target-generated sequences;
- runs the draft on the target's true prefixes;
- computes `match`, `TV`, `KL(P||Q)`, `KL(Q||P)`, `JS`, entropy, margin `delta`, `rho`, `p0/q0`,
  rank;
- simulates greedy block acceptance for gamma = 1..64 and saves `alpha`, `tau`, `survival`;
- saves per-token arrays (`.npz`) for later analysis and summary JSON.

```bash
python -m infra.accept_eval \
  --target qwen3.5-35b-a3b --draft qwen3.5-2b \
  --data data/processed/math500_target.jsonl \
  --gammas 1,2,4,8,16,32,64 --limit 500 \
  --device-map auto --out out/accept/math500_qwen3.5-2b.json
```

Notes:
- Full-vocab TV/KL on GPU, chunked over positions. If top-k is used, include the residual bucket.
- For 27B, `--device-map auto` across 2 GPUs; sort by length to reduce padding.
- `q = 1 - mean(match)`; `E[TV]` should be close to `q` (sampling-mode identity; greedy proxy).

## 5. Real speculative loop

`infra/spec_loop.py` implements greedy SD with KV caches:

- draft proposes `gamma` tokens autoregressively;
- target verifies `prefix + proposals` in one forward;
- accept the longest matching prefix, replace the first mismatch with the target token;
- record per-block accepted length, per-depth acceptance, and timings.

```bash
python -m infra.spec_loop --target qwen3.5-35b-a3b --draft qwen3.5-2b \
  --domain math500 --limit 30 --gamma 8 --max-new 256 \
  --out out/loop/math500_qwen3.5-2b_g8.json
```

Run on 30–50 prompts/domain for gamma 4/8/16/32. This validates teacher forcing and gives the
wall-clock cost ratio `c`, but it is not the main throughput benchmark.

## 6. LM evaluation (lm-eval)

`infra/eval_lm.py` wraps `lm_eval`:

```bash
python -m infra.eval_lm --model qwen3.5-9b \
  --tasks gsm8k,minerva_math,humaneval,wikitext \
  --batch-size 8 --apply-chat-template --out out/lm_eval/qwen3.5-9b.json
```

- Uses `--model hf` by default; `--model vllm` if vLLM is installed.
- For the multimodal checkpoint, export text-only first (recommended) or pass
  `--model_args pretrained=ckpt/text/...`.
- `--apply-chat-template` for instruct tasks; disable for WikiText perplexity.
- The wrapper writes the raw lm-eval results plus a compact summary.

Tasks: `gsm8k`, `minerva_math` (or `math`), `humaneval`, `wikitext`, optional `mmlu`,
`arc_challenge`, `hellaswag`.

## 7. Training infrastructure

`infra/train_distill.py` is a generic distillation trainer:

- student = any Qwen3.5 text model;
- data = target-generated JSONL (`input_ids` / `output_ids`);
- objectives: `ce` (hard token CE), `kd` (top-k KL to target, optional), `accept` (expected
  acceptance-length surrogate, as in the old `train_draft.py`);
- full fine-tune or LoRA (`--lora`), gradient checkpointing, DDP via `torchrun`; FSDP for 27B
  documented in the script header;
- token-budget based training (`--max-tokens`), cosine schedule, warmup, grad clip;
- periodic validation: token accuracy, CE, expected acceptance length at `gamma`;
- checkpointing and resume; JSONL logs every N tokens.

Example:

```bash
torchrun --nproc_per_node 8 -m infra.train_distill \
  --student qwen3.5-2b \
  --data data/processed/distill_target.jsonl \
  --max-tokens 100000000 --seq-len 2048 \
  --lr 1e-4 --warmup 500 --objective ce \
  --val-every-tokens 5000000 --save-every-tokens 25000000 \
  --out ckpt/distill/qwen3.5-2b-100m
```

**Token budget.** The old exploratory runs used ~3M tokens and did not converge. For a scaling claim,
use 50M–200M tokens (at least 2 points per size) and check that val accuracy / val acceptance has
plateaued. Cost estimate: 9B × 100M tokens ≈ 12–20 GPU-hours on 8×A100; data generation is comparable.

**2D scaling.** The trainer logs `tokens`, `loss`, `token_acc`, `soft_accept`, and validation metrics
to `out/train/*.jsonl`. Fit `val_ce(T)` and `q(T)` with a power law and check for a plateau. Combined
with the capacity scan, this gives `q(D,T) = q_inf + A D^-alpha T^-beta` (see `docs/ROUTE_A_PLAN.md`
H2b). Do not interpret the old 3M-token point as an asymptote.

**Data for training.** Generate target outputs on a larger prompt pool (`data/train_pool.jsonl`,
NuminaMath) and optionally code/chat prompts. Keep a held-out validation set that is disjoint from
the eval prompts (MATH-500 test).

## 8. Inference engine / serving

### Default research engine (HF)

`infra/spec_loop.py` is the reference implementation for metrics. It is not optimized for throughput.

### transformers serve (simple OpenAI-compatible)

```bash
transformers serve --force-model /path/to/models/Qwen3.5-9B --port 8000 --continuous-batching
```

### vLLM (recommended for throughput; install separately)

Qwen3.5 needs a recent vLLM (main branch per the model card). Example:

```bash
vllm serve /path/to/models/Qwen3.5-35B-A3B \
  --port 8000 --tensor-parallel-size 2 --max-model-len 32768 \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

For a separate draft model:

```bash
vllm serve <target> --speculative-config '{"model":"<draft>","num_speculative_tokens":5}'
```

### SGLang

```bash
python -m sglang.launch_server --model-path <target> --port 8000 --tp-size 2 \
  --speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

Scripts: `scripts/serve_vllm.sh`, `scripts/serve_sglang.sh` (edit model paths / TP size).
Note: vLLM/SGLang are **not installed** in the current env; install with a recent build if needed.
For the Route A measurements, the HF engine is sufficient; vLLM is for the end-to-end sanity check.

## 9. Metrics and formulas

For prefix `x_<t`, target `P`, draft `Q`:

```
TV(P,Q)      = 0.5 * sum_v |P(v) - Q(v)|
KL(P||Q)     = sum_v P(v) log(P(v)/Q(v))
JS           = 0.5 KL(P||M) + 0.5 KL(Q||M), M=(P+Q)/2
Hellinger    = sqrt(1 - sum_v sqrt(P(v) Q(v)))
delta        = P(x1) - P(x2)                 # probability margin
rho          = 2 TV / delta
A_t          = 1{argmax Q == argmax P}
q            = 1 - mean(A_t)
q_h          = mean(rho >= 1)
tau_inf      = 1/q                            # stationary renewal lemma
S(k)         = Pr[next k drafts all accepted]
alpha(g)     = (1/g) sum_{k=1..g} S(k)
tau(g)       = 1 + sum_{k=1..g} S(k)
```

Certificate (from TV + margin; cite arXiv 2606.30265 for the exact KL version):

```
rho < 1  =>  A_t = 1          (guaranteed acceptance)
A_t = 0  =>  rho >= 1         (rejection implies hard token)
q <= q_h
```

Caution: `delta` is a **probability** margin, not a logit margin. If using logits, convert through
softmax. With top-k truncation, `TV` is a lower bound; include the residual mass.

## 10. Troubleshooting

| symptom | fix |
|---|---|
| `AutoModelForCausalLM.from_pretrained` loads vision / OOM | pass `config=text_config` (see §2) |
| ModelScope 500 / `repo/files` error | retry with backoff; check `out/logs/download_*.log`; copy from another host |
| `modelscope/gsm8k` needs remote code | use `opencompass/gsm8k` or `trust_remote_code=True` |
| `mt_bench` split not found | available splits are `gpt4_pair`, `human`; use `gpt4_pair` and extract prompts |
| CUDA OOM on 27B/35B | `--device-map auto`, smaller batch, length bucketing, or 2–4 GPUs |
| CUDA OOM in `accept_eval` | use `DRAFT_DM=cuda:0`, batch 8 for 0.8B/2B, 4 for 4B/9B, 1 for 27B; set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the scan script does this) |
| loading 27B/35B seems stuck | first load reads 50–70GB from disk; allow 3–5 min and a 600s timeout; `device_map=auto` may split across GPUs |
| lm-eval cannot load Qwen3.5 | export text-only checkpoint (§2) |
| vLLM does not support `qwen3_5` | use a recent main build; otherwise HF engine |
| `tau` real loop much lower than teacher forcing | within-block drift; report both, check gamma <= 8 |
| top-k TV seems too small | include residual bucket `p0/q0`; increase k |

## 11. Reproduce a minimal pilot (2 hours)

```bash
python -m infra.env_check
python -m infra.generate_target --target qwen3.5-9b --domain math500 --limit 50 \
  --max-new 256 --batch-size 4 --device-map cuda:0 --out data/processed/pilot_math.jsonl
python -m infra.accept_eval --target qwen3.5-9b --draft qwen3.5-4b \
  --data data/processed/pilot_math.jsonl --gammas 1,2,4,8,16,32 \
  --device-map cuda:0 --out out/accept/pilot_math_4b.json
python -m infra.spec_loop --target qwen3.5-9b --draft qwen3.5-4b \
  --domain math500 --limit 5 --gamma 4 --max-new 128 --out out/loop/pilot.json
```

This validates the whole pipeline before the 35B target is available.

## 12. 27B ZeRO-3 training stack (2026-11-04)

`infra/train_zero3.py` + `scripts/17_train27b_zero3.sh`; details and measured MFU
in `docs/TRAIN_27B.md`.  Four non-obvious requirements:

1. **Install `flash-linear-attention`.**  Without it, transformers uses a
   pure-torch chunked gated-delta-rule loop that is launch-bound (~8x slower on
   0.8B: 1,143 vs 9,000 tok/s).  The model picks up the Triton kernel through
   its existing `use_kernel_func_from_hub_with_fallback` path.
2. **Do not use `deepspeed.zero.Init` with transformers 5.x.**  It creates
   `[0]`-shaped placeholders and the strict weight loader raises
   `ignore_mismatched_sizes`.  Load on CPU, then `deepspeed.initialize`
   partitions (`--no-zero-init`).
3. **Call `deepspeed.init_distributed()`** before building the torch process
   group; otherwise the DeepSpeed config sees `world_size=1` and the batch
   assertion fails.
4. **Use the fused chunked CE** (`--fused-ce`, default).  HF's loss upcasts the
   full `[B,T,248320]` logits to fp32 (~8 GB at 8x1024) and the 27B run OOMs on
   step 2.  The fused head also handles tied embeddings correctly.

Additional gotchas:

- `distill_mixed.jsonl` has 2,000 rows with `n_prompt=0` (raw FineWeb,
  2.05M tokens).  Keep them; the old FSDP loop filtered them out.
- Save each milestone to `tokens-<N>/`; writing to the run directory
  overwrites 5M with the final checkpoint.
- Memory floor with ZeRO-3 (no offload): 54.4 GB/GPU for params + grads +
  fp32 master + Adam m/v.  Micro-batch 6 (peaks ~71 GB) is safe; micro 8 hits
  74.5 GB and occasionally fails 20 MB `expandable_segments` mappings.
