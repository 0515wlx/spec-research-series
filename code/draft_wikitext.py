"""Compute WikiText NLL / top-1 / KL / TV of a draft against the cached teacher."""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from wikitext_common import OUT, load_chunks, metric_arrays
from model_io import load_model
from eval_accept import get_tok


def load_draft(a):
    if a.teacher_qat:
        from quant import apply_fake_quant
        from quant_teacher import load_teacher
        from packed_moe import pack_experts_inplace, patch_packed_experts
        obj = torch.load(a.teacher_qat, map_location="cpu")
        m = load_teacher(0)
        apply_fake_quant(m, bits=obj["bits"], group=obj["group"], lm_head_bits=obj["bits"])
        pack_experts_inplace(m, bits=obj["bits"], group=obj["group"])
        patch_packed_experts()
        m.load_state_dict(obj["state"], strict=False)
        return m.eval()
    m, _ = load_model(a.ckpt, dtype=torch.bfloat16, device="cuda:0")
    m.eval()
    return m


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--teacher_qat", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--teacher", default=os.path.join(OUT, "wikitext_teacher.npz"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    assert bool(a.ckpt) ^ bool(a.teacher_qat)
    td = np.load(a.teacher)
    chunks = torch.from_numpy(td["target"])  # [n, T-1] targets only
    # We need the full chunk ids; reconstruct from target + teacher top_idx is not enough.
    # load_chunks gives the same split, and targets = chunks[:,1:].
    full = load_chunks(td["target"].shape[0], td["target"].shape[1] + 1)
    assert np.array_equal(full[:, 1:].numpy(), td["target"])

    t0 = time.time()
    model = load_draft(a)
    print(f"loaded {a.tag} {time.time()-t0:.0f}s", flush=True)
    out = {k: [] for k in ["nll", "match", "in5", "rank", "kl_td", "kl_dt", "tv", "js",
                           "ent", "p0", "q0", "pred", "target", "chunk"]}
    t0 = time.time()
    for ci in range(full.shape[0]):
        ids = full[ci:ci + 1].cuda()
        logits = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
        logp = logits.log_softmax(-1)
        tgt = torch.from_numpy(td["target"][ci:ci + 1]).long().cuda()
        top_idx = torch.from_numpy(td["top_idx"][ci:ci + 1]).long().cuda()
        top_logp = torch.from_numpy(td["top_logp"][ci:ci + 1]).float().cuda()
        m = metric_arrays(logp, tgt, top_idx, top_logp)
        for k in out:
            if k == "target":
                out[k].append(tgt.cpu().numpy().reshape(-1))
            elif k == "chunk":
                out[k].append(np.full(tgt.numel(), ci, dtype=np.int32))
            else:
                out[k].append(m[k].cpu().numpy().reshape(-1))
        if ci % 5 == 0:
            print(f"  chunk {ci+1}/{full.shape[0]} {time.time()-t0:.0f}s", flush=True)
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["meta"] = np.array([a.tag])
    np.savez(a.out or os.path.join(OUT, f"wikitext_{a.tag}.npz"), **out)
    print(f"{a.tag}: NLL {out['nll'].mean():.4f} ppl {np.exp(out['nll'].mean()):.2f} "
          f"match {out['match'].mean():.4f} in5 {out['in5'].mean():.4f} "
          f"KL_TD {out['kl_td'].mean():.4f} KL_DT {out['kl_dt'].mean():.4f} "
          f"TV {out['tv'].mean():.4f} JS {out['js'].mean():.4f} "
          f"rank {out['rank'].mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
