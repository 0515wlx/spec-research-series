# Data

Large datasets and generated sequences are **not** committed. This directory contains only
manifests / small files.

## Sources (ModelScope; HuggingFace is blocked on this machine)

| domain | ModelScope id | split | notes |
|---|---|---|---|
| `math500` | `HuggingFaceH4/MATH-500` | `test` | 500 problems |
| `gsm8k` | `opencompass/gsm8k` | `test` | use first 500 |
| `humaneval` | `opencompass/humaneval` | `test` | 164 problems |
| `wikitext` | `Salesforce/wikitext` | `test` (`wikitext-2-raw-v1`) | local cache already present |
| `mtbench` | `lmsys/mt_bench_human_judgments` | `gpt4_pair` | extract first user turn |
| `nq` | `google-research-datasets/natural_questions` | `validation` | optional |

## Prepare

```bash
python -m infra.prepare_data --domains math500 gsm8k humaneval mtbench --limit 500 --out data/processed
python -m infra.prepare_data --domains wikitext --limit 100 --out data/processed
```

Prepared on 2026-11-02:

| domain | rows |
|---|---:|
| math500 | 500 |
| gsm8k | 500 |
| humaneval | 164 |
| mtbench | 80 |
| wikitext | 100 |

This writes `data/processed/{domain}.jsonl` with:

```json
{"id": "...", "domain": "...", "prompt": "...", "reference": "...", "meta": {...}}
```

## Target-generated sequences

The acceptance evaluator scores drafts on target outputs, not reference answers:

```bash
python -m infra.generate_target --target qwen3.5-35b-a3b --domain math500 --limit 500 \
  --max-new 512 --batch-size 8 --device-map auto \
  --out data/processed/math500_target.jsonl
```

Record format:

```json
{"id": "...", "domain": "math500", "prompt": "...", "input_ids": [...], "output_ids": [...],
 "text": "...", "meta": {"target": "qwen3.5-35b-a3b", "max_new": 512, "thinking": false}}
```

For WikiText, `--domain wikitext` uses raw continuation chunks (no generation):
`input_ids` = first half, `output_ids` = second half.

## Existing exploratory data (from the old report)

- `math500.jsonl` — MATH-500 problems (157 KB)
- `teacher_math500_nothink.jsonl` — old teacher (Qwen3.6-35B-A3B) outputs (2.3 MB)
- `distill_nothink.jsonl` — old distillation data (18 MB, gitignored)
- `train_pool.jsonl` — NuminaMath prompt pool (59 MB, gitignored)
- `calib_stats.npz` — old calibration statistics (gitignored)

These are kept for provenance and pilots; the Route A experiments should regenerate target
sequences with `Qwen3.5-35B-A3B`.

## Notes

- Keep tokenizer / chat template / thinking mode identical across drafts.
- Use greedy generation (`do_sample=False`) for the main experiments.
- For WikiText, no chat template.
- Store the target model revision and tokenizer hash in `meta`.
