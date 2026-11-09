"""Clean batch-1 latency protocol without cache-copy overhead inside the timed region.

The earlier protocol (``verify_latency.py`` / ``latency_decomp.py``) clones the KV
cache with ``copy.deepcopy`` *inside* every timed call.  That copy is not part of a
serving pass; it adds a large constant to both the target and the draft, compressing
``rho`` and ``tau`` toward one.  Here we pre-clone the prefix cache outside the timed
region, then time only the model forward.

Two protocols are available:

``reload`` (default)
    One model is resident at a time: the target is measured first, then the model
    is reloaded and its FFN narrowed in place for each width.  This keeps the
    peak GPU memory at one model plus the pre-cloned caches, and is the only
    option for the 27B model.

``resident``
    The target stays resident while each narrow draft is loaded and measured.
    Target and draft timings are interleaved in every round, so GPU clock drift
    between the target phase and the draft phase cancels.  Use this for the
    boundary cells (9B / 8B / 4B) where a few percent of ``rho_zero`` decides
    whether the zero-MLP cost is above or below 5.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m infra.latency_clean --model qwen3-4b \
    --widths 0.5,0.9,0.98,0.99,0.999 --out out/latency/clean_qwen3_4b.json
  CUDA_VISIBLE_DEVICES=0 python -m infra.latency_clean --model qwen3.5-9b \
    --protocol resident --importance out/importance/qwen3.5-9b_act.pt \
    --widths 0.98,0.99 --out out/latency/clean/clean_qwen3.5-9b_resident.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics

import torch

from infra.joint_scan import make_narrow
from infra.model_loader import load_text_model, model_device


@torch.inference_mode()
def timed_forward(model, ids, base_cache, repeats=20, gamma=1):
    """Time a forward (or gamma sequential one-token forwards) without cache copies.

    Caches are pre-cloned outside the timed region; each repeat consumes one clone.
    """
    ts = []
    for _ in range(repeats):
        # Clone outside the timed region so the copy is not charged to the pass.
        c = copy.deepcopy(base_cache)
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        if gamma == 1:
            model(input_ids=ids, past_key_values=c, use_cache=True)
        else:
            for _ in range(gamma):
                out = model(input_ids=ids, past_key_values=c, use_cache=True)
                c = out.past_key_values
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) / 1000.0)
        del c
    torch.cuda.empty_cache()
    return ts


def bench(model, ids, base_cache, repeats, gamma=1):
    ts = timed_forward(model, ids, base_cache, repeats, gamma)
    return {"median_s": statistics.median(ts), "min_s": min(ts), "max_s": max(ts)}


def _one_pass(model, ids, c, gamma):
    """Run gamma sequential one-token decodes on the pre-cloned cache."""
    for _ in range(gamma - 1):
        c = model(input_ids=ids, past_key_values=c, use_cache=True).past_key_values
    model(input_ids=ids, past_key_values=c, use_cache=True)


@torch.inference_mode()
def timed_pair(target_model, draft_model, ids, base_t, base_d, repeats=20, gamma=1):
    """Interleave target and draft passes so clock drift cancels.

    Caches are pre-cloned outside the timed region; each repeat consumes one
    clone per model.  Returns (target_times, draft_times).
    """
    tt, td = [], []
    for _ in range(repeats):
        ct = copy.deepcopy(base_t)
        cd = copy.deepcopy(base_d)
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        _one_pass(target_model, ids, ct, gamma)
        en.record()
        torch.cuda.synchronize()
        tt.append(st.elapsed_time(en) / 1000.0)

        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        _one_pass(draft_model, ids, cd, gamma)
        en.record()
        torch.cuda.synchronize()
        td.append(st.elapsed_time(en) / 1000.0)
        del ct, cd
    torch.cuda.empty_cache()
    return tt, td


def pair_bench(target_model, draft_model, ids, base_t, base_d, repeats=20, gamma=1):
    tt, td = timed_pair(target_model, draft_model, ids, base_t, base_d, repeats, gamma)
    return ({"median_s": statistics.median(tt), "min_s": min(tt), "max_s": max(tt)},
            {"median_s": statistics.median(td), "min_s": min(td), "max_s": max(td)})


@torch.inference_mode()
def pipelined_window(model, ids, base_cache, repeats, gamma=1):
    """Time ``repeats`` back-to-back passes in one event window.

    All cache clones are prepared before the window, so the Python launch loop
    can queue ahead of the GPU and the measurement is GPU-bound rather than
    launch-bound.  Returns seconds per pass.
    """
    clones = [copy.deepcopy(base_cache) for _ in range(repeats)]
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for c in clones:
        if gamma == 1:
            model(input_ids=ids, past_key_values=c, use_cache=True)
        else:
            for _ in range(gamma):
                out = model(input_ids=ids, past_key_values=c, use_cache=True)
                c = out.past_key_values
    en.record()
    torch.cuda.synchronize()
    dt = st.elapsed_time(en) / 1000.0 / repeats
    del clones
    torch.cuda.empty_cache()
    return dt


@torch.inference_mode()
def gpu_window(model, ids, base_cache, repeats, gamma=1):
    """Return mean GPU kernel time per pass (excludes Python/launch bubbles)."""
    from torch.profiler import ProfilerActivity, profile

    clones = [copy.deepcopy(base_cache) for _ in range(repeats)]
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for c in clones:
            if gamma == 1:
                model(input_ids=ids, past_key_values=c, use_cache=True)
            else:
                for _ in range(gamma):
                    out = model(input_ids=ids, past_key_values=c, use_cache=True)
                    c = out.past_key_values
        torch.cuda.synchronize()
    total = 0.0
    for e in prof.key_averages():
        total += getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0.0)
    del clones
    torch.cuda.empty_cache()
    return total / 1e6 / repeats


def gpu_pair(target_model, draft_model, ids, base_t, base_d, repeats=20, gamma=1, rounds=3):
    tt, td = [], []
    for k in range(rounds):
        if k % 2 == 0:
            tt.append(gpu_window(target_model, ids, base_t, repeats, gamma))
            td.append(gpu_window(draft_model, ids, base_d, repeats, gamma))
        else:
            td.append(gpu_window(draft_model, ids, base_d, repeats, gamma))
            tt.append(gpu_window(target_model, ids, base_t, repeats, gamma))
    return ({"median_s": statistics.median(tt), "min_s": min(tt), "max_s": max(tt),
             "windows": tt},
            {"median_s": statistics.median(td), "min_s": min(td), "max_s": max(td),
             "windows": td})


def pipelined_pair(target_model, draft_model, ids, base_t, base_d, repeats=20, gamma=1,
                   rounds=3):
    """Alternate target/draft pipelined windows; return medians per pass."""
    tt, td = [], []
    for k in range(rounds):
        if k % 2 == 0:
            tt.append(pipelined_window(target_model, ids, base_t, repeats, gamma))
            td.append(pipelined_window(draft_model, ids, base_d, repeats, gamma))
        else:
            td.append(pipelined_window(draft_model, ids, base_d, repeats, gamma))
            tt.append(pipelined_window(target_model, ids, base_t, repeats, gamma))
    return ({"median_s": statistics.median(tt), "min_s": min(tt), "max_s": max(tt),
             "windows": tt},
            {"median_s": statistics.median(td), "min_s": min(td), "max_s": max(td),
             "windows": td})


def zero_mlp(model):
    saved = []
    for layer in model.model.layers:
        saved.append((layer.mlp, layer.mlp.forward))
        layer.mlp.forward = lambda x: torch.zeros_like(x)
    return saved


def restore_zero(saved):
    for mod, fn in saved:
        mod.forward = fn


def narrow_model(model, width, importance, pad_to):
    for i, layer in enumerate(model.model.layers):
        layer.mlp, _ = make_narrow(layer, width, "width", model_device(layer),
                                   importance.get(f"l{i:02d}") if importance else None,
                                   pad_to=pad_to)
    return model


def run_resident(a, pipelined=False):
    """Target-resident protocol: interleave target/draft timings per round.

    ``pipelined`` queues a whole window of passes between two CUDA events so the
    measurement is GPU-bound; otherwise each pass is timed individually and the
    result includes Python launch bubbles.
    """
    bench_pair = (pipelined_pair if pipelined == "pipelined"
                  else gpu_pair if pipelined == "gpu" else pair_bench)
    torch.manual_seed(0)
    dev = a.device
    target = load_text_model(a.model, dtype="bfloat16", device_map={"": dev})
    vocab = int(getattr(target.config, "vocab_size", 200000))
    prefix = torch.randint(100, vocab, (1, a.prefix), dtype=torch.long, device=dev)
    one = torch.randint(100, vocab, (1, 1), dtype=torch.long, device=dev)
    gammas = [int(x) for x in a.gammas.split(",")]
    verify_ids = {g: torch.randint(100, vocab, (1, g + 1), dtype=torch.long, device=dev)
                  for g in gammas}
    imp = torch.load(a.importance, map_location="cpu", weights_only=True) if a.importance else None

    with torch.inference_mode():
        base_t = target(input_ids=prefix, use_cache=True).past_key_values

    # Warm the target so clocks are at steady state before any timing.
    for _ in range(a.warmup):
        c = copy.deepcopy(base_t)
        with torch.inference_mode():
            target(input_ids=one, past_key_values=c, use_cache=True)

    rows = []
    for width in [float(x) for x in a.widths.split(",")]:
        draft = load_text_model(a.model, dtype="bfloat16", device_map={"": dev})
        narrow_model(draft, width, imp, a.pad_to)
        with torch.inference_mode():
            base_d = draft(input_ids=prefix, use_cache=True).past_key_values
            for _ in range(a.warmup):
                cd = copy.deepcopy(base_d)
                draft(input_ids=one, past_key_values=cd, use_cache=True)

        t1, d1 = bench_pair(target, draft, one, base_t, base_d, a.repeats, 1)
        tau = {}
        for g in gammas:
            tv, dv = bench_pair(target, draft, verify_ids[g], base_t, base_d, a.repeats, 1)
            tau[str(g)] = {"verify_s": tv["median_s"],
                           "tau": tv["median_s"] / t1["median_s"],
                           "draft_verify_s": dv["median_s"]}

        dg = {}
        for g in gammas:
            dg[str(g)] = {
                "draft_gamma_s": g * d1["median_s"],
                "rho_gamma": g * d1["median_s"] / t1["median_s"],
                "cost": g * d1["median_s"] / t1["median_s"] + tau[str(g)]["tau"],
            }

        saved = zero_mlp(draft)
        for _ in range(a.warmup):
            cd = copy.deepcopy(base_d)
            with torch.inference_mode():
                draft(input_ids=one, past_key_values=cd, use_cache=True)
        tz, dz = bench_pair(target, draft, one, base_t, base_d, a.repeats, 1)
        restore_zero(saved)

        rows.append({
            "width": width,
            "target_one_s": t1["median_s"],
            "draft_one_s": d1["median_s"],
            "rho_token": d1["median_s"] / t1["median_s"],
            "gamma": dg,
            "zero_mlp_one_s": dz["median_s"],
            "rho_zero_mlp": dz["median_s"] / tz["median_s"],
            "tau": {g: tau[g]["tau"] for g in tau},
            "spread": {
                "target": [t1["min_s"], t1["max_s"]],
                "draft": [d1["min_s"], d1["max_s"]],
                "target_zero": [tz["min_s"], tz["max_s"]],
                "draft_zero": [dz["min_s"], dz["max_s"]],
            },
        })
        print(json.dumps({"width": width, "rho": rows[-1]["rho_token"],
                          "rho_zero": rows[-1]["rho_zero_mlp"],
                          "target_s": t1["median_s"],
                          "zero_s": dz["median_s"]}), flush=True)
        del draft, base_d
        torch.cuda.empty_cache()

    out = {"model": a.model, "prefix": a.prefix,
           "protocol": pipelined if isinstance(pipelined, str) and pipelined else
           ("pipelined" if pipelined else "resident"),
           "target_one_s": rows[0]["target_one_s"] if rows else None,
           "target_tau": rows[-1]["tau"] if rows else {},
           "drafts": rows}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}")


def run_reload_gpu(a):
    """One model resident at a time; each phase uses GPU-kernel-time windows."""
    torch.manual_seed(0)
    dev = a.device
    model = load_text_model(a.model, dtype="bfloat16", device_map={"": dev})
    vocab = int(getattr(model.config, "vocab_size", 200000))
    prefix = torch.randint(100, vocab, (1, a.prefix), dtype=torch.long, device=dev)
    one = torch.randint(100, vocab, (1, 1), dtype=torch.long, device=dev)
    gammas = [int(x) for x in a.gammas.split(",")]
    verify_ids = {g: torch.randint(100, vocab, (1, g + 1), dtype=torch.long, device=dev)
                  for g in gammas}

    with torch.inference_mode():
        base = model(input_ids=prefix, use_cache=True).past_key_values
    for _ in range(a.warmup):
        c = copy.deepcopy(base)
        with torch.inference_mode():
            model(input_ids=one, past_key_values=c, use_cache=True)
    torch.cuda.synchronize()

    def win(ids):
        return statistics.median(
            [gpu_window(model, ids, base, a.repeats, 1) for _ in range(3)])

    t1 = win(one)
    tau = {}
    for g in gammas:
        tv = win(verify_ids[g])
        tau[str(g)] = {"verify_s": tv, "tau": tv / t1}
    del model, base
    torch.cuda.empty_cache()

    imp = torch.load(a.importance, map_location="cpu", weights_only=True) if a.importance else None
    rows = []
    for width in [float(x) for x in a.widths.split(",")]:
        draft = load_text_model(a.model, dtype="bfloat16", device_map={"": dev})
        narrow_model(draft, width, imp, a.pad_to)
        with torch.inference_mode():
            pd = draft(input_ids=prefix, use_cache=True).past_key_values
        for _ in range(a.warmup):
            cd = copy.deepcopy(pd)
            with torch.inference_mode():
                draft(input_ids=one, past_key_values=cd, use_cache=True)
        d1 = statistics.median(
            [gpu_window(draft, one, pd, a.repeats, 1) for _ in range(3)])
        saved = zero_mlp(draft)
        for _ in range(a.warmup):
            cd = copy.deepcopy(pd)
            with torch.inference_mode():
                draft(input_ids=one, past_key_values=cd, use_cache=True)
        z1 = statistics.median(
            [gpu_window(draft, one, pd, a.repeats, 1) for _ in range(3)])
        restore_zero(saved)
        dg = {}
        for g in gammas:
            dg[str(g)] = {
                "draft_gamma_s": g * d1,
                "rho_gamma": g * d1 / t1,
                "cost": g * d1 / t1 + tau[str(g)]["tau"],
            }
        rows.append({
            "width": width, "target_one_s": t1, "draft_one_s": d1,
            "rho_token": d1 / t1, "gamma": dg,
            "zero_mlp_one_s": z1, "rho_zero_mlp": z1 / t1,
            "tau": {g: tau[g]["tau"] for g in tau},
        })
        print(json.dumps({"width": width, "rho": rows[-1]["rho_token"],
                          "rho_zero": rows[-1]["rho_zero_mlp"]}), flush=True)
        del draft, pd
        torch.cuda.empty_cache()

    out = {"model": a.model, "prefix": a.prefix, "protocol": "gpu-reload",
           "target_one_s": t1, "target_tau": {g: tau[g]["tau"] for g in tau},
           "drafts": rows}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--importance", default=None)
    ap.add_argument("--widths", default="0.5,0.9,0.98,0.99,0.999")
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--gammas", default="1,4")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--protocol", default="reload",
                    choices=["reload", "resident", "pipelined", "gpu", "gpu-reload"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pad-to", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.protocol in ("resident", "pipelined", "gpu"):
        return run_resident(a, pipelined=a.protocol)
    if a.protocol == "gpu-reload":
        return run_reload_gpu(a)

    torch.manual_seed(0)
    dev = a.device
    model = load_text_model(a.model, dtype="bfloat16", device_map={"": dev})
    vocab = int(getattr(model.config, "vocab_size", 200000))
    prefix = torch.randint(100, vocab, (1, a.prefix), dtype=torch.long, device=dev)
    one = torch.randint(100, vocab, (1, 1), dtype=torch.long, device=dev)
    gammas = [int(x) for x in a.gammas.split(",")]

    with torch.inference_mode():
        base = model(input_ids=prefix, use_cache=True).past_key_values

    t1 = bench(model, one, base, a.repeats, 1)
    tau = {}
    for g in gammas:
        verify_ids = torch.randint(100, vocab, (1, g + 1), dtype=torch.long, device=dev)
        tv = bench(model, verify_ids, base, a.repeats, 1)
        tau[str(g)] = {"verify_s": tv["median_s"],
                       "tau": tv["median_s"] / t1["median_s"],
                       "times": tv}
    del model, base
    torch.cuda.empty_cache()

    imp = torch.load(a.importance, map_location="cpu", weights_only=True) if a.importance else None
    rows = []
    for width in [float(x) for x in a.widths.split(",")]:
        draft = load_text_model(a.model, dtype="bfloat16", device_map={"": dev})
        narrow_model(draft, width, imp, a.pad_to)
        with torch.inference_mode():
            pd = draft(input_ids=prefix, use_cache=True).past_key_values
        d1 = bench(draft, one, pd, a.repeats, 1)
        dg = {}
        for g in gammas:
            dg[str(g)] = {
                "draft_gamma_s": statistics.median(timed_forward(
                    draft, one, pd, a.repeats, g)),
                "rho_gamma": g * d1["median_s"] / t1["median_s"],
                "cost": g * d1["median_s"] / t1["median_s"] + tau[str(g)]["tau"],
            }
        saved = zero_mlp(draft)
        with torch.inference_mode():
            pz = draft(input_ids=prefix, use_cache=True).past_key_values
        z1 = bench(draft, one, pz, a.repeats, 1)
        restore_zero(saved)
        rows.append({
            "width": width,
            "target_one_s": t1["median_s"],
            "draft_one_s": d1["median_s"],
            "rho_token": d1["median_s"] / t1["median_s"],
            "gamma": dg,
            "zero_mlp_one_s": z1["median_s"],
            "rho_zero_mlp": z1["median_s"] / t1["median_s"],
            "tau": {g: tau[g]["tau"] for g in tau},
        })
        print(json.dumps({"width": width, "rho": rows[-1]["rho_token"],
                          "rho_zero": rows[-1]["rho_zero_mlp"]}), flush=True)
        del draft, pd, pz
        torch.cuda.empty_cache()

    out = {"model": a.model, "prefix": a.prefix, "target_one_s": t1["median_s"],
           "target_tau": {g: tau[g]["tau"] for g in tau}, "drafts": rows}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
