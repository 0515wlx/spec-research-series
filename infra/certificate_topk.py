"""Extended top-k rejection certificate.

The top-2 certificate in ``compression_certificate.py`` counts only the event
``z_Q(b) > z_Q(a)`` for the target's top-1 token ``a`` and top-2 token ``b``.
A rejection can also be caused by the target's third (or k-th) token overtaking
``a`` in the draft.  This script measures the nested events

    E_K = { max_{2 <= j <= K} z_Q(v_j) > z_Q(v_1) },   E_2 subset E_3 subset ...

so q_lb^(K) = Pr(E_K) is a tighter, still deterministic post-hoc lower bound on
greedy rejection.  The sequence-cluster bootstrap interval is computed for each
K.  The ratio q_lb^(K)/q_g and the remaining unexplained rejection are reported
so the paper can quantify certificate coverage rather than only its point value.

Only the target's cached top-K tokens and the draft's logits on them are needed;
the target top-256 cache already contains the top-K indices.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from infra import data
from infra.joint_scan import make_narrow
from infra.model_loader import load_text_model, model_device
from infra.rank_scan import _load_cache, _seq_q, bootstrap_q


@torch.inference_mode()
def eval_topk(model, seqs, cache, idxs, ks, device):
    fulls = [seqs[k]["input_ids"] + seqs[k]["output_ids"] for k in idxs]
    xs = [f[:-1] for f in fulls]
    maxlen = max(len(x) for x in xs)
    x = torch.zeros((len(xs), maxlen), dtype=torch.long)
    att = torch.zeros((len(xs), maxlen), dtype=torch.long)
    for i, xx in enumerate(xs):
        x[i, :len(xx)] = torch.tensor(xx, dtype=torch.long)
        att[i, :len(xx)] = 1
    x, att = x.to(device), att.to(device)
    logits = model(input_ids=x, attention_mask=att, use_cache=False).logits.float()
    outs = []
    for i, k in enumerate(idxs):
        s = seqs[k]
        start = len(s["input_ids"]) - 1
        L = len(s["output_ids"])
        lg = logits[i, start:start + L]
        logp = torch.log_softmax(lg, dim=-1)
        top_idx = torch.from_numpy(cache["top_idx"][k, :L]).long().to(device)
        top_logp = torch.from_numpy(cache["top_logp"][k, :L]).float().to(device)
        target_argmax = torch.from_numpy(cache["argmax"][k, :L]).long().to(device)
        pred = lg.argmax(-1)
        match = (pred == target_argmax).cpu().numpy()
        # draft log-probabilities on the target's cached top-K tokens
        d_lp = logp.gather(-1, top_idx)
        out = {"match": match, "q": 1.0 - float(match.mean())}
        for K in ks:
            K = min(K, top_idx.shape[-1])
            # the top-1 column is column 0; the event is any j in [1, K) > 0
            event = (d_lp[:, 1:K] > d_lp[:, :1]).any(-1)
            out[f"q_lb_top{K}"] = event.float().cpu().numpy()
        outs.append(out)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--importance", default="out/importance/qwen3.5-9b_act.pt")
    ap.add_argument("--width", type=float, default=0.98)
    ap.add_argument("--mode", default="width", choices=["width", "joint"])
    ap.add_argument("--sets", nargs="+", required=True,
                    help="name:data.jsonl:cache.npz entries")
    ap.add_argument("--ks", default="2,3,5,8,16,32,64,128,256")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ks = [int(x) for x in a.ks.split(",")]
    datasets = []
    for spec in a.sets:
        name, dpath, cpath = spec.split(":")
        seqs = data.load_target_sequences(dpath, limit=a.limit)
        datasets.append((name, seqs, _load_cache(cpath)))
        print(f"set {name}: {len(seqs)} seqs", flush=True)

    model = load_text_model(a.model, dtype=a.dtype, device_map=a.device_map)
    dev = model_device(model)
    imp = torch.load(a.importance, map_location="cpu", weights_only=True)
    for i, layer in enumerate(model.model.layers):
        layer.mlp, _ = make_narrow(layer, a.width, a.mode, next(layer.parameters()).device,
                                   imp.get(f"l{i:02d}") if imp else None)
    print(f"compressed to width={a.width} mode={a.mode}", flush=True)

    rows = []
    for name, seqs, cache in datasets:
        order = sorted(range(len(seqs)),
                       key=lambda i: len(seqs[i]["input_ids"]) + len(seqs[i]["output_ids"]))
        all_out = [None] * len(seqs)
        for i in range(0, len(order), a.batch_size):
            idxs = order[i:i + a.batch_size]
            outs = eval_topk(model, seqs, cache, idxs, ks, dev)
            for k, o in zip(idxs, outs):
                all_out[k] = o
            torch.cuda.empty_cache()
        seq_len = np.array([len(o["match"]) for o in all_out], dtype=np.int32)
        seq_id = np.repeat(np.arange(len(all_out), dtype=np.int32), seq_len)
        match = np.concatenate([o["match"] for o in all_out]).astype(bool)
        n_seq = len(all_out)
        q = 1.0 - float(match.mean())
        q_ci = bootstrap_q(match, seq_id, n_seq)
        row = {"set": name, "width": a.width, "q": q, "q_ci": list(q_ci),
               "tokens": int(len(match)), "seqs": n_seq}
        for K in ks:
            key = f"q_lb_top{K}"
            if key not in all_out[0]:
                continue
            arr = np.concatenate([o[key] for o in all_out]).astype(bool)
            per = np.zeros(n_seq); cnt = np.zeros(n_seq)
            np.add.at(per, seq_id, arr.astype(np.float64))
            np.add.at(cnt, seq_id, 1.0)
            rng = np.random.default_rng(0)
            idx = rng.integers(0, n_seq, size=(2000, n_seq))
            boot = per[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
            row[key] = float(arr.mean())
            row[key + "_ci"] = [float(np.percentile(boot, 2.5)),
                                float(np.percentile(boot, 97.5))]
            row[key + "_coverage"] = float(arr.mean() / max(q, 1e-12))
        rows.append(row)
        print(json.dumps({k: row[k] for k in row if not k.endswith("_ci")}, indent=1),
              flush=True)
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
