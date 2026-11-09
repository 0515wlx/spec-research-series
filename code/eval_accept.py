"""Measure greedy speculative-decoding acceptance of a draft model against cached teacher sequences."""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model_io import load_model

TOK = None


def get_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("/path/to/models/Qwen3.6-35B-A3B")


@torch.no_grad()
def token_matches(model, tok, seqs, gen_lens, device, bs=8, max_tok=12288):
    """seqs: list of 1d LongTensor full sequences. Returns per-seq bool arrays over generated tokens."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [None] * len(seqs)
    i = 0
    while i < len(order):
        j = i
        bin0 = len(seqs[order[i]]) // 128
        tot = 0
        while j < len(order) and len(seqs[order[j]]) // 128 == bin0 and (j - i + 1) <= bs:
            tot += len(seqs[order[j]])
            if tot > max_tok:
                break
            j += 1
        if j == i:
            j = i + 1
        ids = order[i:j]
        L = max(len(seqs[k]) for k in ids)
        x = torch.full((len(ids), L), tok.pad_token_id, dtype=torch.long)
        for r, k in enumerate(ids):
            x[r, :len(seqs[k])] = seqs[k]
        x = x.to(device)
        att = torch.ones_like(x)
        for r, k in enumerate(ids):
            att[r, len(seqs[k]):] = 0
        logits = model(input_ids=x, attention_mask=att, use_cache=False).logits
        for r, k in enumerate(ids):
            gl = gen_lens[k]
            s = len(seqs[k]) - gl
            pred = logits[r, s - 1: len(seqs[k]) - 1].argmax(-1).cpu()
            tgt = seqs[k][s:]
            out[k] = (pred == tgt).numpy()
        i = j
    return out


def simulate(match, gammas):
    """match: bool array over generated tokens. Greedy verification simulation."""
    N = len(match)
    res = {}
    for g in gammas:
        t = 0
        acc = prop = steps = 0
        lens = []
        while t < N:
            L = 0
            while L < g and t + L < N and match[t + L]:
                L += 1
            acc += L
            prop += min(g, N - t)
            lens.append(L + 1)
            steps += 1
            t += L + 1
        res[g] = {"alpha": acc / max(prop, 1), "tau": float(np.mean(lens)), "steps": steps,
                  "tokens_per_step": N / max(steps, 1)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/path/to/spec-research/data/teacher_math500_nothink.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gammas", default="1,2,3,4,5,6,8,10")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="draft")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tok = get_tok()
    recs = [json.loads(l) for l in open(a.data)]
    if a.limit:
        recs = recs[:a.limit]
    seqs, gen_lens = [], []
    ntrunc = 0
    PAD = tok.pad_token_id
    for r in recs:
        inp, out = r["input_ids"], r["output_ids"]
        while out and out[-1] == PAD:
            out = out[:-1]
        if len(out) < 8:
            continue
        seqs.append(torch.tensor(inp + out, dtype=torch.long))
        gen_lens.append(len(out))
    model, meta = load_model(a.ckpt, dtype=torch.bfloat16, device=a.device)
    model.eval()
    t0 = time.time()
    matches = token_matches(model, tok, seqs, gen_lens, a.device, bs=a.bs)
    dt = time.time() - t0
    gammas = [int(x) for x in a.gammas.split(",")]
    agg = {g: {"acc": 0, "prop": 0, "lens": []} for g in gammas}
    per_seq = []
    nmatch = ntok = 0
    for k, m in enumerate(matches):
        nmatch += int(m.sum()); ntok += len(m)
        for g in gammas:
            t = 0; N = len(m)
            while t < N:
                L = 0
                while L < g and t + L < N and m[t + L]:
                    L += 1
                agg[g]["acc"] += L
                agg[g]["prop"] += min(g, N - t)
                agg[g]["lens"].append(L + 1)
                t += L + 1
    # survival[k] = P(the next k draft tokens are ALL accepted), averaged over all
    # token positions.  This is the "long-horizon" accuracy curve, independent of gamma.
    surv = {}
    for k in [1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64]:
        tot = cnt = 0
        for m in matches:
            if len(m) < k:
                continue
            ok = np.convolve(m.astype(np.int8), np.ones(k, dtype=np.int8), "valid") == k
            tot += int(ok.sum())
            cnt += len(ok)
        surv[k] = tot / max(cnt, 1)
    summary = {"ckpt": a.ckpt, "n_seq": len(matches), "gen_tokens": ntok,
               "pos_match_rate": nmatch / max(ntok, 1), "eval_time": dt,
               "survival": surv, "per_gamma": {}}
    for g in gammas:
        d = agg[g]
        summary["per_gamma"][g] = {"alpha": d["acc"] / max(d["prop"], 1),
                                   "tau": float(np.mean(d["lens"])),
                                   "accepted": d["acc"], "proposed": d["prop"]}
    print(json.dumps(summary, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
