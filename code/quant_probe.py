"""Quick PTQ probe: how much does weight-only quantization hurt draft acceptance?"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from draft import load_dense
from quant import apply_fake_quant, quant_stats
from eval_accept import get_tok, token_matches


def simulate(m, gammas):
    out = {}
    N = len(m)
    for g in gammas:
        t = acc = prop = steps = 0
        lens = []
        while t < N:
            L = 0
            while L < g and t + L < N and m[t + L]:
                L += 1
            acc += L
            prop += min(g, N - t)
            lens.append(L + 1)
            steps += 1
            t += L + 1
        out[g] = {"alpha": acc / max(prop, 1), "tau": float(np.mean(lens))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/dense_distill_s1")
    ap.add_argument("--data", default="/path/to/spec-research/data/teacher_math500_nothink.jsonl")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--lm_head_bits", type=int, default=None)
    ap.add_argument("--skip_lm_head", action="store_true")
    ap.add_argument("--gammas", default="1,3,5,8,12,16,24,32")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="ptq")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tok = get_tok()
    recs = [json.loads(l) for l in open(a.data)][:a.limit]
    seqs, gen_lens = [], []
    for r in recs:
        inp, out = r["input_ids"], r["output_ids"]
        while out and out[-1] == tok.pad_token_id:
            out = out[:-1]
        if len(out) < 8:
            continue
        seqs.append(torch.tensor(inp + out, dtype=torch.long))
        gen_lens.append(len(out))

    model, meta = load_dense(a.ckpt, dtype=torch.bfloat16, device=a.device)
    model.eval()
    info = {"tag": a.tag, "ckpt": a.ckpt, "n_seq": len(seqs), "bits": a.bits,
            "group": a.group, "lm_head_bits": a.lm_head_bits, "skip_lm_head": a.skip_lm_head}
    if a.bits < 16:
        names = apply_fake_quant(model, bits=a.bits, group=a.group,
                                 skip_lm_head=a.skip_lm_head, lm_head_bits=a.lm_head_bits)
        info["quantized_modules"] = len(names)
        info["sizes"] = quant_stats(model)
    t0 = time.time()
    matches = token_matches(model, tok, seqs, gen_lens, a.device, bs=8)
    info["pos_match_rate"] = float(sum(int(m.sum()) for m in matches) / max(sum(len(m) for m in matches), 1))
    gammas = [int(x) for x in a.gammas.split(",")]
    agg = {g: ([], 0, 0) for g in gammas}
    for m in matches:
        for g in gammas:
            L, prop = 0, 0
            t, N = 0, len(m)
            acc = 0
            lens = []
            while t < N:
                k = 0
                while k < g and t + k < N and m[t + k]:
                    k += 1
                acc += k
                prop += min(g, N - t)
                lens.append(k + 1)
                t += k + 1
            agg[g][0].append(lens)
            agg[g] = (agg[g][0], agg[g][1] + acc, agg[g][2] + prop)
    info["per_gamma"] = {str(g): {"alpha": agg[g][1] / max(agg[g][2], 1),
                                  "tau": float(np.mean([x for l in agg[g][0] for x in l]))}
                         for g in gammas}
    info["eval_time"] = time.time() - t0
    print(json.dumps(info, indent=1))
    if a.out:
        json.dump(info, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
