"""Reload a saved teacher QAT state and measure its draft acceptance."""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from quant import apply_fake_quant
from quant_teacher import load_teacher
from packed_moe import pack_experts_inplace, patch_packed_experts
from eval_accept import get_tok, token_matches

D = "/path/to/spec-research/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/teacher_qat_w4_100.pt")
    ap.add_argument("--data", default=f"{D}/teacher_math500_nothink.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--gammas", default="1,2,3,5,8,12,16,24,32")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    obj = torch.load(a.ckpt, map_location="cpu")
    bits, group = obj["bits"], obj["group"]
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

    model = load_teacher(0)
    apply_fake_quant(model, bits=bits, group=group, lm_head_bits=bits)
    pack_experts_inplace(model, bits=bits, group=group)
    patch_packed_experts()
    missing, unexpected = model.load_state_dict(obj["state"], strict=False)
    print("missing", len(missing), "unexpected", len(unexpected), flush=True)
    model.eval()

    t0 = time.time()
    matches = token_matches(model, tok, seqs, gen_lens, "cuda:0", bs=1, max_tok=4096)
    gammas = [int(x) for x in a.gammas.split(",")]
    per = {}
    for g in gammas:
        acc = prop = 0
        lens = []
        for m in matches:
            t, N = 0, len(m)
            while t < N:
                k = 0
                while k < g and t + k < N and m[t + k]:
                    k += 1
                acc += k
                prop += min(g, N - t)
                lens.append(k + 1)
                t += k + 1
        per[str(g)] = {"alpha": acc / max(prop, 1), "tau": float(np.mean(lens))}
    surv = {}
    for k in [1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64]:
        tot = cnt = 0
        for m in matches:
            if len(m) < k:
                continue
            ok = np.convolve(m.astype(np.int8), np.ones(k, dtype=np.int8), "valid") == k
            tot += int(ok.sum()); cnt += len(ok)
        surv[str(k)] = tot / max(cnt, 1)
    info = {"ckpt": a.ckpt, "bits": bits, "group": group, "n_seq": len(seqs),
            "eval_time": time.time() - t0,
            "pos_match_rate": float(sum(int(m.sum()) for m in matches) /
                                    max(sum(len(m) for m in matches), 1)),
            "per_gamma": per, "survival": surv}
    print(json.dumps(info, indent=1))
    if a.out:
        json.dump(info, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
