"""Post-hoc MoE compression scan: active-expert (top-k) reduction and expert deletion.

The target and the draft are the same MoE checkpoint.  Two post-hoc operators are
supported:

``topk``
    Reduce the number of active routed experts from ``k`` to ``k'`` at every layer.
    This is the compute-reducing MoE operator: the routed-expert FLOPs scale with
    ``k'`` while attention, the shared expert, and the router are unchanged.

``prune``
    Delete a fraction of experts (lowest calibration usage first) while keeping the
    original top-k.  This reduces parameters and expert memory but *not* per-token
    FLOPs, because each token still activates k experts.  It is reported as a
    memory-only control, analogous to weight-only quantization.

The target top-k cache is built from the uncompressed MoE target, so at k'=k with no
pruning the draft is the target and q=0.

Example
-------
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m infra.moe_scan \
  --model qwen3.5-35b-a3b \
  --sets math:data/processed/math500_target.jsonl:out/target_cache/math500_k256.npz \
         wikitext:data/processed/wikitext_target.jsonl:out/target_cache/wikitext_k256.npz \
         fineweb:data/processed/fineweb_target.jsonl:out/target_cache/fineweb_k256.npz \
  --topks 8,6,4,3,2,1 --prune-keeps 1.0,0.5,0.25 \
  --out out/rank_scan/qwen3.5-35b-a3b_moe
"""
from __future__ import annotations

import argparse
import json
import os
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

from infra import data
from infra.accept_eval import eval_batch
from infra.model_loader import load_text_model, model_device
from infra.rank_scan import _load_cache, _seq_q, bootstrap_q


def _patch_router():
    """Patch the Qwen3.5-MoE router to honour per-instance top_k / expert_mask."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(hidden_states, self.weight)
        mask = getattr(self, "expert_mask", None)
        if mask is not None:
            logits = logits + mask.to(device=logits.device, dtype=logits.dtype)
            kept = int((mask == 0).sum().item())
        else:
            kept = self.num_experts
        k = min(int(self.top_k), kept)
        probs = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
        topv, topi = torch.topk(probs, k, dim=-1)
        topv = topv / topv.sum(dim=-1, keepdim=True)
        return logits, topv.to(logits.dtype), topi

    Qwen3_5MoeTopKRouter.forward = forward


def _set_routers(model, top_k: int, keep_frac: float, usage: np.ndarray | None):
    """Install top-k / expert-mask state on every router.  Returns a restore list."""
    old = []
    for i, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "gate") or not hasattr(mlp.gate, "top_k"):
            continue
        r = mlp.gate
        old.append((r, r.top_k, getattr(r, "expert_mask", None)))
        r.top_k = int(top_k)
        if keep_frac >= 0.999 or usage is None:
            r.expert_mask = None
        else:
            n = r.num_experts
            keep = max(1, int(round(n * keep_frac)))
            order = np.argsort(-usage[i])
            kept = np.zeros(n, dtype=bool)
            kept[order[:keep]] = True
            mask = torch.full((n,), float("-inf"), dtype=torch.float32)
            mask[torch.from_numpy(kept)] = 0.0
            r.expert_mask = mask
    return old


def _restore_routers(old):
    for r, top_k, mask in old:
        r.top_k = top_k
        r.expert_mask = mask


def moe_cost_ratio(model_key: str, top_k: int, keep_frac: float = 1.0) -> dict:
    """Nominal FLOP and parameter ratios for the two MoE operators."""
    from infra.flops import param_stats
    from infra.model_registry import resolve
    from transformers import AutoConfig

    path = resolve(model_key)
    s = param_stats(path)
    cfg = AutoConfig.from_pretrained(path)
    tc = getattr(cfg, "text_config", cfg)
    L = int(tc.num_hidden_layers)
    h = int(tc.hidden_size)
    inter = int(tc.moe_intermediate_size)
    E = int(tc.num_experts)
    k0 = int(tc.num_experts_per_tok)
    routed_total = L * E * 3 * h * inter
    active_target = s.compute - routed_total + L * k0 * 3 * h * inter
    active_routed = L * int(top_k) * 3 * h * inter
    flop_ratio = (s.compute - routed_total + active_routed) / active_target
    keep = max(1, int(round(E * keep_frac)))
    mem_ratio = (s.total - (E - keep) * 3 * h * inter * L) / s.total
    return {
        "model": model_key,
        "num_experts": E,
        "top_k": int(top_k),
        "keep_experts": keep,
        "keep_frac": float(keep_frac),
        "routed_params_total_b": routed_total / 1e9,
        "active_routed_params_b": active_routed / 1e9,
        "rho_FLOP": float(flop_ratio),
        "rho_mem": float(mem_ratio),
    }


def _summarize(all_out, gammas, n_seq):
    keys = ["match", "tv", "margin_log", "d_margin_log", "margin", "ent_t"]
    cat = {k: np.concatenate([o[k] for o in all_out], axis=0) for k in keys}
    seq_len = np.array([len(o["match"]) for o in all_out], dtype=np.int32)
    seq_id = np.repeat(np.arange(len(all_out), dtype=np.int32), seq_len)
    match = cat["match"].astype(bool)
    q = 1.0 - float(match.mean())
    flip = (cat["margin_log"].astype(np.float64) + cat["d_margin_log"].astype(np.float64)) < 0
    q_lb = float(flip.mean())
    # cluster bootstrap for q and q_lb
    def boot(x):
        per = np.zeros(n_seq)
        cnt = np.zeros(n_seq)
        np.add.at(per, seq_id, x.astype(np.float64))
        np.add.at(cnt, seq_id, 1)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, n_seq, size=(2000, n_seq))
        return (per[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1))
    qb = 1.0 - boot(match)
    lbb = boot(flip)
    from infra.metrics import simulate_blocks
    return {
        "q": q,
        "q_ci": [float(np.percentile(qb, 2.5)), float(np.percentile(qb, 97.5))],
        "q_lb": q_lb,
        "q_lb_ci": [float(np.percentile(lbb, 2.5)), float(np.percentile(lbb, 97.5))],
        "E_TV": float(cat["tv"].mean()),
        "tokens": int(len(match)),
        "seqs": n_seq,
        "per_gamma": simulate_blocks(match, gammas),
    }, cat, seq_len, seq_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-35b-a3b")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--topks", default="8,6,4,3,2,1")
    ap.add_argument("--prune-keeps", default="1.0,0.5,0.25",
                    help="expert keep fractions; 1.0 means no deletion")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--gammas", default="1,2,4,8,16")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--usage", default="data/calib_stats.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    gammas = [int(x) for x in a.gammas.split(",")]
    datasets = []
    for spec in a.sets:
        name, dpath, cpath = spec.split(":")
        seqs = data.load_target_sequences(dpath, limit=a.limit)
        datasets.append((name, seqs, _load_cache(cpath)))
        print(f"set {name}: {len(seqs)} seqs", flush=True)

    usage = None
    if os.path.exists(a.usage):
        z = np.load(a.usage)
        if "usage" in z.files:
            usage = z["usage"].astype(np.float64)
            print(f"usage stats {usage.shape}", flush=True)

    _patch_router()
    model = load_text_model(a.model, dtype=a.dtype, device_map=a.device_map)
    dev = model_device(model)
    print(f"loaded {a.model} on {dev}", flush=True)

    topks = [int(x) for x in a.topks.split(",")]
    keeps = [float(x) for x in a.prune_keeps.split(",")]
    configs = []
    for k in topks:
        configs.append((k, 1.0, "topk"))
    for kp in keeps:
        if kp < 0.999:
            configs.append((max(topks), kp, "prune"))
    # combined configs: prune + reduce top-k, only if both axes are present
    for kp in keeps:
        if kp < 0.999:
            for k in topks:
                if k < max(topks):
                    configs.append((k, kp, "combined"))

    for k, kp, kind in configs:
        tag = f"{kind}_k{k}_e{kp:g}"
        out_json = os.path.join(a.out, f"{tag}.json")
        if os.path.exists(out_json):
            print(f"skip {tag}", flush=True)
            continue
        t0 = time.time()
        old = _set_routers(model, k, kp, usage)
        cost = moe_cost_ratio(a.model, k, kp)
        rows = []
        for name, seqs, cache in datasets:
            print(f"  {tag} set={name}", flush=True)
            order = sorted(range(len(seqs)),
                           key=lambda i: len(seqs[i]["input_ids"]) + len(seqs[i]["output_ids"]))
            all_out = [None] * len(seqs)
            for i in range(0, len(order), a.batch_size):
                idxs = order[i:i + a.batch_size]
                outs = eval_batch(model, seqs, cache, idxs, dev)
                for kk, o in zip(idxs, outs):
                    all_out[kk] = o
            summ, cat, seq_len, seq_id = _summarize(all_out, gammas, len(seqs))
            summ.update({"set": name, "model": a.model, "kind": kind, "top_k": k,
                         "keep_frac": kp, **cost})
            stem = os.path.join(a.out, f"{name}_{tag}")
            np.savez(stem + ".npz", **cat, gen_len=seq_len, seq_id=seq_id,
                     q_per_seq=_seq_q(cat["match"].astype(bool), seq_id, len(seqs))[0])
            with open(stem + ".json", "w") as f:
                json.dump(summ, f, indent=1)
            rows.append(summ)
            print(f"    q={summ['q']:.4f} q_lb={summ['q_lb']:.4f} "
                  f"TV={summ['E_TV']:.4f} rho={summ['rho_FLOP']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=1)
        _restore_routers(old)
        torch.cuda.empty_cache()
        print(f"  wrote {out_json} ({time.time()-t0:.0f}s)", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
