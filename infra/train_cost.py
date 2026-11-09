"""Training FLOPs / communication cost for Qwen3.5-27B.

Planning tool, not a benchmark.  Parameter counts and FLOPs/token come from
``infra.flops`` (read from the checkpoint headers); this script adds the DP
all-gather / reduce-scatter estimate for different per-GPU batch sizes.
"""
from __future__ import annotations

import json

from infra.flops import _config, flops_per_token, param_stats

CONFIG = "/path/to/models/Qwen3.5-27B/config.json"
SEQ = 2048
GPU_PEAK_TFLOPS = 312.0       # A100 BF16 dense
NVLINK_GBPS = 600.0           # A100 NVLink
WORLD = 8


def main():
    path = "/path/to/models/Qwen3.5-27B"
    s = param_stats(path)
    cfg = _config(path)
    total = s.total
    fpt = flops_per_token(s, cfg, SEQ)

    print(f"hidden={cfg.hidden_size} layers={cfg.num_hidden_layers}")
    print(f"params: total={total/1e9:.3f}B compute={s.compute/1e9:.3f}B "
          f"mlp={s.mlp/1e9:.2f}B full_attn={s.full_attn/1e9:.2f}B "
          f"linear_attn={s.linear_attn/1e9:.2f}B lm_head={s.lm_head/1e9:.2f}B "
          f"embed={s.embed/1e9:.2f}B")
    print(f"MLP share of params: {s.mlp/total:.1%}")
    print(f"FLOPs/token: {fpt/1e9:.1f} GFLOP at seq={SEQ}")

    p_bytes = total * 2  # bf16
    # ZeRO-3 / FSDP: forward all-gather + backward all-gather + reduce-scatter
    comm_bytes_per_step = 3 * p_bytes
    print(f"DP comm/step: {comm_bytes_per_step/1e9:.1f} GB per rank")
    print(f"{'batch/GPU':>9} {'tokens/step':>12} {'comm/token':>12} "
          f"{'comm ms/token':>13} {'compute ms/token':>16} {'comm/compute':>12}")
    for b in [1, 2, 4, 8, 16]:
        tok = b * SEQ
        comm_tok = comm_bytes_per_step / tok
        comm_ms = comm_tok / (NVLINK_GBPS * 1e9) * 1e3
        comp_ms = fpt / (GPU_PEAK_TFLOPS * 1e12) * 1e3
        print(f"{b:9d} {tok:12d} {comm_tok/1e6:10.1f} MB {comm_ms:13.3f} "
              f"{comp_ms:16.3f} {comm_ms/comp_ms:12.1%}")
    print("\nNote: A100 peak is a ceiling.  Measured 2026-11-04 with DeepSpeed "
          "ZeRO-3: ~95 TFLOP/s/GPU nominal (~30% MFU), ~125 TFLOP/s including "
          "activation-checkpoint recompute (~40% of peak).")


if __name__ == "__main__":
    main()
