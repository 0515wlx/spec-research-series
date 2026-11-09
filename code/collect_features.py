"""Collect per-token draft predictions / confidence / entropy on cached teacher sequences.

Used by the dynamic-depth analysis (REPORT §15).  Supports:
  --ckpt ckpt/moe8_keep8_distill          (normal dense/MoE draft)
  --teacher_qat ckpt/teacher_qat_w4_100.pt (packed W4 teacher QAT)
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model_io import load_model
from eval_accept import get_tok

D = "/path/to/spec-research/data"


def load_qat(path):
    from quant import apply_fake_quant
    from quant_teacher import load_teacher
    from packed_moe import pack_experts_inplace, patch_packed_experts
    obj = torch.load(path, map_location="cpu")
    model = load_teacher(0)
    apply_fake_quant(model, bits=obj["bits"], group=obj["group"], lm_head_bits=obj["bits"])
    pack_experts_inplace(model, bits=obj["bits"], group=obj["group"])
    patch_packed_experts()
    model.load_state_dict(obj["state"], strict=False)
    return model.eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--teacher_qat", default=None)
    ap.add_argument("--data", default=f"{D}/teacher_math500_nothink.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    assert bool(a.ckpt) ^ bool(a.teacher_qat)

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

    t0 = time.time()
    if a.teacher_qat:
        model = load_qat(a.teacher_qat)
        print(f"QAT loaded {time.time()-t0:.0f}s", flush=True)
    else:
        model, meta = load_model(a.ckpt, dtype=torch.bfloat16, device=a.device)
        model.eval()
        print(f"draft loaded {time.time()-t0:.0f}s", flush=True)

    preds, matches, maxps, ents, seqids, prevs, tgts = [], [], [], [], [], [], []
    t0 = time.time()
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    i = 0
    while i < len(order):
        j = i
        tot = 0
        L0 = len(seqs[order[i]]) // 128
        while j < len(order) and len(seqs[order[j]]) // 128 == L0 and (j - i + 1) <= a.bs:
            tot += len(seqs[order[j]])
            if tot > 6000:
                break
            j += 1
        if j == i:
            j = i + 1
        ids = order[i:j]
        L = max(len(seqs[k]) for k in ids)
        x = torch.full((len(ids), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(x)
        for r, k in enumerate(ids):
            x[r, :len(seqs[k])] = seqs[k]
            att[r, :len(seqs[k])] = 1
        x, att = x.to(a.device), att.to(a.device)
        logits = model(input_ids=x, attention_mask=att, use_cache=False).logits.float()
        for r, k in enumerate(ids):
            s = len(seqs[k]) - gen_lens[k]
            lg = logits[r, s - 1: len(seqs[k]) - 1]
            pr = lg.softmax(-1)
            pred = pr.argmax(-1)
            tgt = seqs[k][s:].to(pred.device)
            preds.append(pred.cpu().numpy().astype(np.int32))
            matches.append((pred == tgt).cpu().numpy().astype(np.uint8))
            maxps.append(pr.max(-1).values.cpu().numpy().astype(np.float32))
            ents.append((-(pr * pr.clamp(min=1e-12).log()).sum(-1)).cpu().numpy().astype(np.float32))
            seqids.append(np.full(len(tgt), k, dtype=np.int32))
            prevs.append(seqs[k][s - 1: len(seqs[k]) - 1].numpy().astype(np.int32))
            tgts.append(tgt.cpu().numpy().astype(np.int32))
        i = j
        if (i // 10) % 2 == 0:
            print(f"seq {i}/{len(seqs)} {time.time()-t0:.0f}s", flush=True)

    out = {k: np.concatenate(v) for k, v in [
        ("pred", preds), ("match", matches), ("maxp", maxps), ("ent", ents),
        ("seq", seqids), ("prev", prevs), ("tgt", tgts)]}
    np.savez(a.out, **out)
    print("saved", a.out, "tokens", len(out["match"]),
          "match", out["match"].mean(), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
