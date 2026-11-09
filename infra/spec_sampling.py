"""Real standard speculative-sampling loop (HF reference engine).

This complements :mod:`infra.spec_loop` (greedy verification).  It implements the
standard stochastic speculative sampling rule of Leviathan et al. / Chen et al.:

  * the draft samples ``gamma`` tokens from Q, storing Q at each position;
  * the target computes P at each prefix;
  * a draft token x is accepted with probability ``min(1, P(x)/Q(x))``;
  * on rejection, the replacement is sampled from ``normalize(max(0, P-Q))``;
  * if the whole block is accepted, a bonus token is sampled from P.

It records the empirical acceptance rate, the per-position acceptance
probability ``sum_v min(P(v), Q(v)) = 1 - TV(P,Q)``, the accepted length
``tau``, and a greedy top-1 mismatch measured on the same sampled prefixes.
That last quantity lets us compare standard SD with the greedy ``q`` headline
without conflating the two protocols.

The engine is intentionally simple: the target re-runs the full prefix each
block.  It is a measurement reference, not a throughput benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from infra import data
from infra.model_loader import load_text_model, load_tokenizer, model_device


def _prompt_ids(tok, domain: str, prompt: str, thinking: bool, max_prompt_tokens: int):
    spec = data.get_domain(domain)
    text = data.chat_prompt(tok, prompt, thinking=thinking) if spec.chat else prompt
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return ids[-max_prompt_tokens:]


def _prob(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits.float(), dim=-1)


@torch.inference_mode()
def sd_sample(draft, target, tok, prompt_ids, gamma: int, max_new: int,
              ddev, tdev, eos_ids, seed: int = 0, temperature: float = 1.0):
    """Run standard speculative sampling on one prompt.

    Returns a dict with empirical acceptance, per-position acceptance
    probability, tau, and a greedy top-1 mismatch measured on the sampled
    prefixes.
    """
    torch.manual_seed(seed)
    cur = list(prompt_ids)
    emitted: list[int] = []
    blocks: list[dict] = []
    n_acc = n_prop = 0
    acc_probs: list[float] = []
    greedy_hits: list[int] = []
    tv_sum = 0.0
    tv_n = 0
    draft_s = target_s = 0.0

    while len(emitted) < max_new:
        ids = torch.tensor(cur, dtype=torch.long, device=ddev)[None]
        t0 = time.time()
        o = draft(input_ids=ids, use_cache=True)
        past = o.past_key_values
        prop: list[int] = []
        q_logits: list[torch.Tensor] = []
        logits = o.logits[0, -1]
        for i in range(gamma):
            q = _prob(logits / temperature)
            x = int(torch.multinomial(q, 1).item())
            prop.append(x)
            q_logits.append(logits)
            if i < gamma - 1:
                o = draft(input_ids=torch.tensor([[x]], dtype=torch.long, device=ddev),
                          past_key_values=past, use_cache=True)
                past = o.past_key_values
                logits = o.logits[0, -1]
        if ddev.type == "cuda":
            torch.cuda.synchronize(ddev)
        draft_s += time.time() - t0

        t1 = time.time()
        xt = torch.tensor(cur, dtype=torch.long, device=tdev)[None]
        pv = torch.tensor(prop, dtype=torch.long, device=tdev)[None]
        o2 = target(input_ids=torch.cat([xt, pv], dim=1), use_cache=False)
        # logits at position len(cur)-1+i predict the token after prefix i.
        # There are gamma+1 distributions: P_1..P_gamma and the bonus P_{gamma+1}.
        tlog = o2.logits[0, len(cur) - 1: len(cur) - 1 + gamma + 1]
        if tdev.type == "cuda":
            torch.cuda.synchronize(tdev)
        target_s += time.time() - t1

        accepted = 0
        replacement = None
        block_ratios: list[float] = []
        block_greedy: list[int] = []
        for i in range(gamma):
            p = _prob(tlog[i] / temperature)
            q = _prob(q_logits[i].to(p.device) / temperature)
            x = prop[i]
            ratio = float(torch.minimum(torch.ones((), device=p.device),
                                        (p[x] / q[x].clamp_min(1e-20))).item())
            acc_probs.append(ratio)
            block_ratios.append(ratio)
            # Greedy top-1 agreement at the same prefix distribution.
            hit = int(int(p.argmax()) == int(q.argmax()))
            greedy_hits.append(hit)
            block_greedy.append(hit)
            tv_sum += float(0.5 * (p - q).abs().sum())
            tv_n += 1
            u = float(torch.rand((), device=p.device).item())
            if u < ratio:
                accepted += 1
            else:
                resid = (p - q).clamp_min(0.0)
                s = float(resid.sum())
                if s <= 0:
                    resid = p
                else:
                    resid = resid / s
                replacement = int(torch.multinomial(resid, 1).item())
                break

        if replacement is None:
            bonus = int(torch.multinomial(_prob(tlog[gamma] / temperature), 1).item())
            new = prop + [bonus]
        else:
            new = prop[:accepted] + [replacement]

        n_acc += accepted
        n_prop += gamma
        cur.extend(new)
        emitted.extend(new)
        blocks.append({"accepted": accepted, "proposed": gamma,
                       "emitted": len(new), "depth": len(blocks),
                       "ratios": block_ratios, "greedy": block_greedy})
        if new and new[-1] in eos_ids:
            break

    q_greedy_loop = 1.0 - (sum(greedy_hits) / len(greedy_hits)) if greedy_hits else None
    return {
        "alpha_empirical": n_acc / max(n_prop, 1),
        "alpha_token": (sum(acc_probs) / len(acc_probs)) if acc_probs else None,
        "E_TV_loop": (tv_sum / tv_n) if tv_n else None,
        "q_greedy_loop": q_greedy_loop,
        "tau": len(emitted) / max(len(blocks), 1),
        "steps": len(blocks),
        "tokens": len(emitted),
        "accepted": n_acc,
        "proposed": n_prop,
        "draft_seconds": draft_s,
        "target_seconds": target_s,
        "blocks": blocks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device-map-draft", default="cuda:1")
    ap.add_argument("--device-map-target", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.data:
        seqs = data.load_target_sequences(a.data, limit=a.limit)
        prompts = [{"prompt": s.get("prompt", ""), "input_ids": s["input_ids"],
                    "id": s.get("id", i)} for i, s in enumerate(seqs)]
    else:
        prompts = data.load_prompts(a.domain, limit=a.limit)

    tok = load_tokenizer(a.target)
    draft = load_text_model(a.draft, dtype=a.dtype, device_map=a.device_map_draft)
    target = load_text_model(a.target, dtype=a.dtype, device_map=a.device_map_target)
    ddev, tdev = model_device(draft), model_device(target)
    eos_ids = {tok.eos_token_id}
    print(f"draft={a.draft}@{ddev} target={a.target}@{tdev}", flush=True)

    rows = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        if "input_ids" in p:
            ids = p["input_ids"]
        else:
            ids = _prompt_ids(tok, a.domain, p["prompt"], a.thinking, a.max_prompt_tokens)
        r = sd_sample(draft, target, tok, ids, a.gamma, a.max_new, ddev, tdev, eos_ids,
                      seed=a.seed + i, temperature=a.temperature)
        r["id"] = p.get("id", i)
        rows.append(r)
        print(f"  {i+1}/{len(prompts)} tau={r['tau']:.2f} "
              f"alpha_emp={r['alpha_empirical']:.3f} alpha_tok={r['alpha_token']:.3f} "
              f"q_greedy_loop={r['q_greedy_loop']:.3f} {time.time()-t0:.0f}s", flush=True)

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "target": a.target, "draft": a.draft, "domain": a.domain,
        "gamma": a.gamma, "max_new": a.max_new, "n": len(rows),
        "temperature": a.temperature, "seed": a.seed,
        "alpha_empirical": mean("alpha_empirical"),
        "alpha_token": mean("alpha_token"),
        "E_TV_loop": mean("E_TV_loop"),
        "q_greedy_loop": mean("q_greedy_loop"),
        "tau": mean("tau"),
        "draft_seconds": sum(r["draft_seconds"] for r in rows),
        "target_seconds": sum(r["target_seconds"] for r in rows),
        "rows": rows,
    }
    # Per-depth conditional acceptance probability (reached positions only).
    depth: dict[str, list[float]] = {}
    depth_g: dict[str, list[int]] = {}
    for r in rows:
        for b in r.get("blocks", []):
            for d, ratio in enumerate(b.get("ratios", [])):
                depth.setdefault(str(d), []).append(float(ratio))
            for d, hit in enumerate(b.get("greedy", [])):
                depth_g.setdefault(str(d), []).append(int(hit))
    summary["per_depth_accept"] = {k: sum(v) / max(len(v), 1) for k, v in sorted(depth.items())}
    summary["per_depth_greedy_accept"] = {
        k: sum(v) / max(len(v), 1) for k, v in sorted(depth_g.items())}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in
                      ["alpha_empirical", "alpha_token", "E_TV_loop",
                       "q_greedy_loop", "tau", "n"]}, indent=1))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
