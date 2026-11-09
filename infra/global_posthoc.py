"""Global post-hoc FFN-width baselines.

The paper's main audit uses a fixed activation-aware score, a uniform per-layer
width, and a ridge readout.  This module adds the strongest post-hoc baselines
that a reviewer can reasonably ask for without training:

* ``global``: a global subset search over all FFN channels under a total width
  budget (top-R channels by a layer-normalised score).
* ``dp``: a per-layer width DP with a convex removal penalty, at the same total
  budget.
* ``obs``: an OBS-style second-moment score ``E[a_j^2] * ||W_down[:,j]||^2``
  instead of the activation-magnitude score ``E|a_j| * ||W_down[:,j]||``.

All three are evaluated on the same cached target distributions, with the same
``make_narrow`` operator and the same audit as the main sweep.  The point is to
test whether a global optimiser, not just a better ranking at fixed uniform
width, reopens the raw-text rejection floor.

Subcommands
-----------
collect
    Collect per-channel second moments from calibration text.
plan
    Build per-layer width plans for ``global`` and ``dp`` at a given average.
eval
    Apply a plan (or a uniform baseline) and evaluate q / q_lb on target caches.
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import time

import numpy as np
import torch

from infra import data
from infra.accept_eval import eval_batch
from infra.joint_scan import make_narrow
from infra.layer_scan import _summarize as summarize
from infra.model_loader import load_text_model, model_device
from infra.rank_scan import _load_cache, _seq_q


@torch.inference_mode()
def collect_second_moment(model, tok, rows, max_tokens, max_len, device):
    layers = model.model.layers
    sums = [None] * len(layers)
    counts = [0] * len(layers)
    handles = []

    def make_hook(i):
        def hook(mod, inp, out):
            x = inp[0]
            a = mod.act_fn(mod.gate_proj(x)) * mod.up_proj(x)
            v = (a.float() ** 2).sum(dim=(0, 1)).cpu()
            sums[i] = v if sums[i] is None else sums[i] + v
            counts[i] += a.shape[0] * a.shape[1]
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.mlp.register_forward_hook(make_hook(i)))
    n = 0
    for r in rows:
        text = r.get("text") or r.get("prompt") or ""
        if not text:
            continue
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"].to(device)
        if ids.shape[1] < 16:
            continue
        model(input_ids=ids, use_cache=False)
        n += ids.shape[1]
        if n >= max_tokens:
            break
    for h in handles:
        h.remove()
    return [s / max(c, 1) for s, c in zip(sums, counts)], n


def cmd_collect(a):
    from infra.model_loader import load_tokenizer
    rows = []
    for path in a.data:
        with open(path) as f:
            rows.extend(json.loads(line) for line in f)
    tok = load_tokenizer(a.model)
    model = load_text_model(a.model, dtype="bfloat16", device_map={"": a.device})
    dev = model_device(model)
    acts, n = collect_second_moment(model, tok, rows, a.max_tokens, a.max_len, dev)
    print(f"collected {n} tokens", flush=True)
    out = {}
    for i, layer in enumerate(model.model.layers):
        wd = layer.mlp.down_proj.weight.detach().float().cpu()
        # OBS-style marginal output error: E[a_j^2] * ||W_down[:, j]||^2.
        out[f"l{i:02d}"] = acts[i] * wd.pow(2).sum(0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(out, a.out)
    print(f"saved {len(out)} layers -> {a.out}", flush=True)


def _layer_scores(model, act_path, obs_path):
    act = torch.load(act_path, map_location="cpu", weights_only=True)
    obs = torch.load(obs_path, map_location="cpu", weights_only=True)
    layers = []
    for i, layer in enumerate(model.model.layers):
        key = f"l{i:02d}"
        wd = layer.mlp.down_proj.weight.detach().float().cpu()
        wnorm2 = wd.pow(2).sum(0)
        inter = int(layer.mlp.gate_proj.weight.shape[0])
        # Both scores are per-channel importances.  Normalise per layer so that
        # a global top-R is not dominated by a layer whose weights happen to be
        # on a different scale.
        a = act[key].float()
        o = obs[key].float() if obs is not None else a * wnorm2
        a = a / a.mean().clamp_min(1e-12)
        o = o / o.mean().clamp_min(1e-12)
        layers.append({"inter": inter, "act": a, "obs": o})
    return layers


def _top_global(layers, score_key, total_keep):
    scores = torch.cat([x[score_key] for x in layers])
    idx = torch.topk(scores, total_keep).indices
    counts = torch.zeros(len(layers), dtype=torch.long)
    start = 0
    for i, x in enumerate(layers):
        n = x["inter"]
        counts[i] = ((idx >= start) & (idx < start + n)).sum()
        start += n
    # every layer must keep at least one channel
    counts = counts.clamp_min(1)
    while int(counts.sum()) > total_keep:
        # remove from the layer with the most channels
        i = int(torch.argmax(counts))
        if counts[i] <= 1:
            break
        counts[i] -= 1
    while int(counts.sum()) < total_keep:
        i = int(torch.argmin(counts))
        if counts[i] >= layers[i]["inter"]:
            break
        counts[i] += 1
    return counts.tolist()


def _cum_removed(scores, counts_grid):
    """Cumulative removed mass for keeping r channels, assuming ascending removal."""
    asc = torch.sort(scores).values  # smallest first are removed first
    cum = torch.cumsum(asc, 0)
    total = cum[-1].clamp_min(1e-12)
    out = []
    for r in counts_grid:
        removed = int(len(scores) - r)
        m = float(cum[removed - 1] / total) if removed > 0 else 0.0
        out.append(m)
    return out


def _dp_counts(layers, score_key, total_keep, step=32, beta=1.0, layer_weight=None):
    """Per-layer width DP with a convex removal penalty.

    Minimise sum_l [m_l(r) + beta * m_l(r)^2], where m_l(r) is the fraction of
    layer l's score mass removed at width r.  The grid is a multiple of
    ``step`` channels; the remainder is assigned greedily at the end.
    """
    n = len(layers)
    target_steps = max(1, int(round(total_keep / step)))
    grids = []
    costs = []
    for x in layers:
        max_r = x["inter"]
        ks = list(range(0, max_r + 1, step))
        if ks[-1] != max_r:
            ks.append(max_r)
        # r counts are kept channels; convert to removed counts
        rem = [_cum_removed(x[score_key], [r])[0] for r in ks]
        w = 1.0 if layer_weight is None else float(layer_weight[len(costs)])
        c = [w * (m + beta * m * m) for m in rem]
        grids.append(ks)
        costs.append(c)
    # steps index: use floor(r/step), and allow the last partial step.
    # Simpler: DP over integer channel counts but on a step grid.
    # dp[s] = min cost with s kept channels (in units of step), plus a small
    # penalty for unused capacity to make the target exact.
    INF = 1e30
    dp = np.full(target_steps + 1, INF, dtype=np.float64)
    dp[0] = 0.0
    choice = []
    for i, x in enumerate(layers):
        max_s = min(target_steps, x["inter"] // step)
        new = np.full(target_steps + 1, INF, dtype=np.float64)
        arg = np.zeros(target_steps + 1, dtype=np.int16)
        c = np.asarray(costs[i][:max_s + 1], dtype=np.float64)
        # min-plus convolution, vectorised over s and k
        for s in range(target_steps + 1):
            lo = max(0, s - max_s)
            if lo > s:
                continue
            cand = dp[lo:s + 1] + c[s - lo::-1]
            j = int(np.argmin(cand))
            new[s] = cand[j]
            arg[s] = s - lo - j
        dp = new
        choice.append(arg)
    # backtrack
    counts = [0] * n
    s = target_steps
    for i in range(n - 1, -1, -1):
        k = int(choice[i][s])
        counts[i] = k * step
        s -= k
    # assign the remainder to the layers with the smallest marginal cost
    for _ in range(total_keep - sum(counts)):
        best = None
        for i, x in enumerate(layers):
            if counts[i] >= x["inter"]:
                continue
            m1 = _cum_removed(x[score_key], [counts[i]])[0]
            m2 = _cum_removed(x[score_key], [counts[i] + 1])[0]
            c = (m2 + beta * m2 * m2) - (m1 + beta * m1 * m1)
            if best is None or c < best[0]:
                best = (c, i)
        if best is None:
            break
        counts[best[1]] += 1
    return counts


def _dp_sensitivity(layers, total_keep, sensitivity, step=32, beta=1.0):
    """DP allocation with a measured per-layer sensitivity weight."""
    s = np.asarray(sensitivity, dtype=np.float64)
    # Normalise so the mean weight is one; only relative weights matter.
    s = s / max(float(s.mean()), 1e-12)
    return _dp_counts(layers, "act", total_keep, step=step, beta=beta,
                      layer_weight=s.tolist())


def _greedy_counts(layers, score_key, total_keep, beta=1.0):
    """Greedy global removal with a convex penalty; a cheap DP fallback."""
    counts = [x["inter"] for x in layers]
    remove = sum(counts) - total_keep
    # Precompute cumulative removed mass for each layer.
    cums = []
    for x in layers:
        asc = torch.sort(x[score_key]).values
        cums.append(torch.cumsum(asc, 0).numpy() / float(asc.sum().clamp_min(1e-12)))
    heap = []
    for i, x in enumerate(layers):
        def marg(i, r):
            n = len(cums[i])
            rem = n - r
            m0 = cums[i][rem] if rem < n else 1.0
            m1 = cums[i][rem + 1] if rem + 1 < n else 1.0
            return (m1 + beta * m1 * m1) - (m0 + beta * m0 * m0)
        heapq.heappush(heap, (marg(i, counts[i]), i))
    for _ in range(remove):
        val, i = heapq.heappop(heap)
        if counts[i] <= 1:
            heapq.heappush(heap, (1e30, i))
            continue
        counts[i] -= 1
        heapq.heappush(heap, (marg(i, counts[i]), i))
    return counts


def cmd_plan(a):
    model = load_text_model(a.model, dtype="bfloat16", device_map={"": "cpu"})
    layers = _layer_scores(model, a.act, a.obs)
    total_inter = sum(x["inter"] for x in layers)
    out = {}
    for avg in [float(x) for x in a.avgs.split(",")]:
        total_keep = int(round(avg * total_inter))
        for score_key in a.scores.split(","):
            g = _top_global(layers, score_key, total_keep)
            d = _dp_counts(layers, score_key, total_keep, step=a.step, beta=a.beta)
            for name, counts in [("global", g), ("dp", d)]:
                out[f"{name}_{score_key}_{avg:g}"] = {
                    "name": name, "score": score_key, "avg": avg,
                    "total_keep": total_keep, "total_inter": total_inter,
                    "counts": counts,
                    "fracs": [c / x["inter"] for c, x in zip(counts, layers)],
                    "min_frac": min(c / x["inter"] for c, x in zip(counts, layers)),
                    "max_frac": max(c / x["inter"] for c, x in zip(counts, layers)),
                }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        print(f"{k}: min={v['min_frac']:.4f} max={v['max_frac']:.4f} "
              f"mean={np.mean(v['fracs']):.4f}", flush=True)
    print(f"saved plans -> {a.out}", flush=True)


def _restore(model, orig):
    for layer, mlp in zip(model.model.layers, orig):
        layer.mlp = mlp


def _apply_config(model, orig, act, obs, score, counts, avg,
                  calib=None, ridge=False):
    from infra.joint_scan import NarrowMLP
    from infra.layerwise_reconstruct import reconstruct_layer
    _restore(model, orig)
    for i, layer in enumerate(model.model.layers):
        inter = int(orig[i].gate_proj.weight.shape[0])
        frac = counts[i] / inter if counts is not None else avg
        imp = act[f"l{i:02d}"] if score == "act" else obs[f"l{i:02d}"]
        layer_dev = next(layer.parameters()).device
        narrow, _ = make_narrow(layer, frac, "width", layer_dev, imp)
        if ridge and calib is not None:
            W, lam, mse0, mse1 = reconstruct_layer(narrow, calib[0][i], calib[1][i])
            layer.mlp = NarrowMLP(narrow.config, narrow.hidden_size,
                                  narrow.intermediate_size, narrow.gate_w,
                                  narrow.up_w, W, narrow.act_fn).to(layer_dev)
            del narrow
        else:
            layer.mlp = narrow
        torch.cuda.empty_cache()


@torch.inference_mode()
def _eval_set(model, seqs, cache, device, gammas, batch_size):
    order = sorted(range(len(seqs)),
                   key=lambda i: len(seqs[i]["input_ids"]) + len(seqs[i]["output_ids"]))
    outs = [None] * len(seqs)
    for i in range(0, len(order), batch_size):
        idxs = order[i:i + batch_size]
        for k, o in zip(idxs, eval_batch(model, seqs, cache, idxs, device)):
            outs[k] = o
    return summarize(outs, gammas, len(seqs))


def _counts_from_fracs(layers, fracs, total_keep):
    counts = [max(1, int(round(f * x["inter"]))) for f, x in zip(fracs, layers)]
    while sum(counts) > total_keep:
        i = int(np.argmax([c - 1 for c in counts]))
        if counts[i] <= 1:
            break
        counts[i] -= 1
    while sum(counts) < total_keep:
        i = int(np.argmin(counts))
        if counts[i] >= layers[i]["inter"]:
            break
        counts[i] += 1
    return counts


def cmd_random(a):
    """Best-of-budget random per-layer width search on a calibration split."""
    seqs = data.load_target_sequences(a.data, limit=a.limit)
    cache = _load_cache(a.cache)
    act = torch.load(a.act, map_location="cpu", weights_only=True)
    obs = torch.load(a.obs, map_location="cpu", weights_only=True)
    model = load_text_model(a.model, dtype="bfloat16", device_map={"": a.device})
    dev = model_device(model)
    orig = [layer.mlp for layer in model.model.layers]
    layers = []
    for i, layer in enumerate(model.model.layers):
        layers.append({"inter": int(layer.mlp.gate_proj.weight.shape[0])})
    total_inter = sum(x["inter"] for x in layers)
    gammas = [int(x) for x in a.gammas.split(",")]
    rng = np.random.default_rng(a.seed)
    out = {}
    for avg in [float(x) for x in a.avgs.split(",")]:
        total_keep = int(round(avg * total_inter))
        candidates = []
        for k in range(a.candidates):
            z = rng.normal(size=len(layers))
            w = avg + a.std * z
            w += avg - w.mean()
            w = np.clip(w, a.min_frac, 1.0)
            counts = _counts_from_fracs(layers, w.tolist(), total_keep)
            _apply_config(model, orig, act, obs, a.score, counts, avg)
            summ, _, _, _ = _eval_set(model, seqs, cache, dev, gammas, a.batch_size)
            candidates.append({"candidate": k, "q": summ["q"], "q_lb": summ["q_lb"],
                               "counts": counts,
                               "fracs": [c / x["inter"] for c, x in zip(counts, layers)]})
            print(f"  avg={avg:g} cand={k:02d} q={summ['q']:.4f} q_lb={summ['q_lb']:.4f}",
                  flush=True)
        best = min(candidates, key=lambda x: (x["q"], x["q_lb"]))
        out[f"random_{a.score}_{avg:g}"] = {
            "name": "random", "score": a.score, "avg": avg,
            "total_keep": total_keep, "total_inter": total_inter,
            "counts": best["counts"], "fracs": best["fracs"],
            "calib_q": best["q"], "calib_q_lb": best["q_lb"],
            "candidates": [{"q": c["q"], "q_lb": c["q_lb"]} for c in candidates],
        }
    _restore(model, orig)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved random-search plans -> {a.out}", flush=True)


def cmd_sensitivity(a):
    """Measure the q increase from removing ``delta`` of one layer at a time."""
    seqs = data.load_target_sequences(a.data, limit=a.limit)
    cache = _load_cache(a.cache)
    act = torch.load(a.act, map_location="cpu", weights_only=True)
    obs = torch.load(a.obs, map_location="cpu", weights_only=True)
    model = load_text_model(a.model, dtype="bfloat16", device_map={"": a.device})
    dev = model_device(model)
    orig = [layer.mlp for layer in model.model.layers]
    gammas = [int(x) for x in a.gammas.split(",")]

    _restore(model, orig)
    base, _, _, _ = _eval_set(model, seqs, cache, dev, gammas, a.batch_size)
    print(f"base q={base['q']:.4f}", flush=True)
    rows = []
    for i, layer in enumerate(model.model.layers):
        _restore(model, orig)
        inter = int(orig[i].gate_proj.weight.shape[0])
        frac = 1.0 - a.delta
        imp = act[f"l{i:02d}"] if a.score == "act" else obs[f"l{i:02d}"]
        layer_dev = next(layer.parameters()).device
        layer.mlp, _ = make_narrow(layer, frac, "width", layer_dev, imp)
        summ, _, _, _ = _eval_set(model, seqs, cache, dev, gammas, a.batch_size)
        rows.append({"layer": i, "frac": frac, "q": summ["q"],
                     "q_lb": summ["q_lb"], "q_ci": summ["q_ci"]})
        print(f"  layer {i:02d} q={summ['q']:.4f} q_lb={summ['q_lb']:.4f}", flush=True)
        torch.cuda.empty_cache()
    _restore(model, orig)
    out = {"model": a.model, "mode": "width", "score": a.score, "delta": a.delta,
           "base_q": base["q"], "base_q_lb": base["q_lb"], "layers": rows}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved sensitivity -> {a.out}", flush=True)


def cmd_plan_sens(a):
    model = load_text_model(a.model, dtype="bfloat16", device_map={"": "cpu"})
    layers = _layer_scores(model, a.act, a.obs)
    sens = json.load(open(a.sensitivity))
    base = sens["base_q"]
    w = [max(float(x["q"]) - base, a.floor) for x in sens["layers"]]
    total_inter = sum(x["inter"] for x in layers)
    out = {}
    for avg in [float(x) for x in a.avgs.split(",")]:
        total_keep = int(round(avg * total_inter))
        counts = _dp_sensitivity(layers, total_keep, w, step=a.step, beta=a.beta)
        out[f"sens_act_{avg:g}"] = {
            "name": "sens", "score": "act", "avg": avg,
            "total_keep": total_keep, "total_inter": total_inter,
            "counts": counts,
            "fracs": [c / x["inter"] for c, x in zip(counts, layers)],
            "min_frac": min(c / x["inter"] for c, x in zip(counts, layers)),
            "max_frac": max(c / x["inter"] for c, x in zip(counts, layers)),
            "sensitivity": w,
        }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        print(f"{k}: min={v['min_frac']:.4f} max={v['max_frac']:.4f} "
              f"mean={np.mean(v['fracs']):.4f}", flush=True)
    print(f"saved plans -> {a.out}", flush=True)


def cmd_eval(a):
    plan = json.load(open(a.plan)) if a.plan else None
    seqs_by_name = {}
    caches = {}
    for spec in a.sets:
        name, dpath, cpath = spec.split(":")
        seqs_by_name[name] = data.load_target_sequences(dpath, limit=a.limit)
        caches[name] = _load_cache(cpath)
        print(f"set {name}: {len(seqs_by_name[name])} seqs", flush=True)
    act = torch.load(a.act, map_location="cpu", weights_only=True)
    obs = torch.load(a.obs, map_location="cpu", weights_only=True)
    model = load_text_model(a.model, dtype="bfloat16", device_map={"": a.device})
    dev = model_device(model)
    orig = [layer.mlp for layer in model.model.layers]
    gammas = [int(x) for x in a.gammas.split(",")]
    os.makedirs(a.out, exist_ok=True)
    calib = None
    if a.ridge:
        from infra.layerwise_reconstruct import collect_calibration
        from infra.model_loader import load_tokenizer
        tok = load_tokenizer(a.model)
        rows = [json.loads(line) for line in open(a.calib)]
        X, Y = collect_calibration(model, tok, rows, a.calib_tokens, 1024, dev)
        calib = (X, Y)
        print(f"ridge calibration: {X[0].shape[0]} tokens", flush=True)

    configs = []
    for avg in [float(x) for x in a.avgs.split(",")]:
        configs.append({"tag": f"uniform_act_{avg:g}", "avg": avg, "score": "act",
                        "counts": None, "kind": "uniform"})
        configs.append({"tag": f"uniform_obs_{avg:g}", "avg": avg, "score": "obs",
                        "counts": None, "kind": "uniform"})
        if plan:
            for score in a.plan_scores.split(","):
                for kind in a.plan_kinds.split(","):
                    key = f"{kind}_{score}_{avg:g}"
                    if key in plan:
                        configs.append({"tag": key, "avg": avg, "score": score,
                                        "counts": plan[key]["counts"], "kind": kind})
    keep = set(a.tags.split(",")) if a.tags else None
    for cfg in configs:
        if keep is not None and cfg["tag"] not in keep:
            continue
        # fresh original MLPs for each config
        _apply_config(model, orig, act, obs, cfg["score"], cfg["counts"], cfg["avg"],
                      calib=calib, ridge=a.ridge)
        for name, seqs in seqs_by_name.items():
            print(f"  {cfg['tag']} set={name}", flush=True)
            order = sorted(range(len(seqs)),
                           key=lambda i: len(seqs[i]["input_ids"]) + len(seqs[i]["output_ids"]))
            outs = [None] * len(seqs)
            t0 = time.time()
            for i in range(0, len(order), a.batch_size):
                idxs = order[i:i + a.batch_size]
                for k, o in zip(idxs, eval_batch(model, seqs, caches[name], idxs, dev)):
                    outs[k] = o
            summ, cat, seq_len, seq_id = summarize(outs, gammas, len(seqs))
            if cfg["counts"] is None:
                fracs = [cfg["avg"]] * len(model.model.layers)
            else:
                fracs = [cfg["counts"][i] / int(orig[i].gate_proj.weight.shape[0])
                         for i in range(len(model.model.layers))]
            summ.update({"model": a.model, "set": name, "tag": cfg["tag"],
                         "kind": cfg["kind"], "score": cfg["score"], "avg": cfg["avg"],
                         "fracs": [float(x) for x in fracs],
                         "eval_seconds": time.time() - t0})
            stem = os.path.join(a.out, f"{name}_{cfg['tag']}")
            np.savez(stem + ".npz", **cat, gen_len=seq_len, seq_id=seq_id,
                     q_per_seq=_seq_q(cat["match"].astype(bool), seq_id, len(seqs))[0])
            with open(stem + ".json", "w") as f:
                json.dump(summ, f, indent=1)
            print(f"    q={summ['q']:.4f} q_lb={summ.get('q_lb', float('nan')):.4f} "
                  f"ci={summ['q_ci']} ({time.time()-t0:.0f}s)", flush=True)
            torch.cuda.empty_cache()
    _restore(model, orig)
    print("done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--model", default="qwen3.5-9b")
    c.add_argument("--data", nargs="+", default=["data/processed/distill_controlled.jsonl"])
    c.add_argument("--max-tokens", type=int, default=200000)
    c.add_argument("--max-len", type=int, default=1024)
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_collect)

    p = sub.add_parser("plan")
    p.add_argument("--model", default="qwen3.5-9b")
    p.add_argument("--act", required=True)
    p.add_argument("--obs", required=True)
    p.add_argument("--avgs", default="0.98,0.99")
    p.add_argument("--scores", default="act,obs")
    p.add_argument("--step", type=int, default=32)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_plan)

    r = sub.add_parser("random-search")
    r.add_argument("--model", default="qwen3.5-9b")
    r.add_argument("--data", required=True)
    r.add_argument("--cache", required=True)
    r.add_argument("--act", required=True)
    r.add_argument("--obs", required=True)
    r.add_argument("--score", default="obs", choices=["act", "obs"])
    r.add_argument("--avgs", default="0.5,0.98,0.99")
    r.add_argument("--candidates", type=int, default=12)
    r.add_argument("--std", type=float, default=0.03)
    r.add_argument("--min-frac", type=float, default=0.05)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--limit", type=int, default=30)
    r.add_argument("--batch-size", type=int, default=4)
    r.add_argument("--gammas", default="1,4")
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_random)

    s = sub.add_parser("sensitivity")
    s.add_argument("--model", default="qwen3.5-9b")
    s.add_argument("--data", required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--act", required=True)
    s.add_argument("--obs", required=True)
    s.add_argument("--score", default="act", choices=["act", "obs"])
    s.add_argument("--delta", type=float, default=0.02)
    s.add_argument("--limit", type=int, default=30)
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--gammas", default="1,4")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_sensitivity)

    ps = sub.add_parser("plan-sens")
    ps.add_argument("--model", default="qwen3.5-9b")
    ps.add_argument("--act", required=True)
    ps.add_argument("--obs", required=True)
    ps.add_argument("--sensitivity", required=True)
    ps.add_argument("--avgs", default="0.98,0.99")
    ps.add_argument("--step", type=int, default=32)
    ps.add_argument("--beta", type=float, default=1.0)
    ps.add_argument("--floor", type=float, default=1e-6)
    ps.add_argument("--out", required=True)
    ps.set_defaults(func=cmd_plan_sens)

    e = sub.add_parser("eval")
    e.add_argument("--model", default="qwen3.5-9b")
    e.add_argument("--sets", nargs="+", required=True)
    e.add_argument("--act", required=True)
    e.add_argument("--obs", required=True)
    e.add_argument("--plan", default=None)
    e.add_argument("--plan-kinds", default="global,dp")
    e.add_argument("--plan-scores", default="act,obs")
    e.add_argument("--avgs", default="0.98,0.99")
    e.add_argument("--tags", default=None,
                   help="comma-separated exact config tags to run")
    e.add_argument("--ridge", action="store_true",
                   help="apply the layer-wise down-projection ridge readout")
    e.add_argument("--calib", default="data/processed/distill_controlled.jsonl")
    e.add_argument("--calib-tokens", type=int, default=16000)
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--batch-size", type=int, default=4)
    e.add_argument("--gammas", default="1,4")
    e.add_argument("--device", default="cuda:0")
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_eval)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
