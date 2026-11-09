"""Run the teacher once on WikiText and cache top-k log-probs + last-layer hidden."""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from wikitext_common import TEACHER, OUT, load_chunks
from quant_teacher import load_teacher


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(OUT, "wikitext_teacher.npz"))
    a = ap.parse_args()
    chunks = load_chunks(a.n, a.seq_len)
    model = load_teacher(0)
    cap = {}
    hk = model.model.layers[-1].register_forward_hook(
        lambda mod, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    rows = {"top_idx": [], "top_logp": [], "nll": [], "ent": [], "hidden": [],
            "target": [], "chunk": []}
    t0 = time.time()
    with torch.no_grad():
        for ci in range(chunks.shape[0]):
            ids = chunks[ci:ci + 1].cuda()
            out = model(input_ids=ids, use_cache=False)
            logits = out.logits[:, :-1].float()
            logp = logits.log_softmax(-1)
            tgt = ids[:, 1:]
            tk = logp.topk(a.topk, dim=-1)
            nll = -logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            ent = -(logp.exp() * logp).sum(-1)
            h = cap["h"][:, :-1].to(torch.float16)
            rows["top_idx"].append(tk.indices.cpu().numpy().astype(np.int32))
            rows["top_logp"].append(tk.values.cpu().numpy().astype(np.float16))
            rows["nll"].append(nll.cpu().numpy().astype(np.float32))
            rows["ent"].append(ent.cpu().numpy().astype(np.float32))
            rows["hidden"].append(h.cpu().numpy().astype(np.float16))
            rows["target"].append(tgt.cpu().numpy().astype(np.int32))
            rows["chunk"].append(np.full(tgt.shape[1], ci, dtype=np.int32))
            print(f"chunk {ci+1}/{chunks.shape[0]} {time.time()-t0:.0f}s", flush=True)
    hk.remove()
    n, T = chunks.shape[0], chunks.shape[1] - 1
    out = {}
    for k, v in rows.items():
        arr = np.concatenate(v, axis=1) if k in ("top_idx", "top_logp", "hidden", "target") else np.concatenate(v)
        out[k] = arr.reshape(n, T, *arr.shape[2:]) if k in ("top_idx", "top_logp", "hidden", "target") else arr.reshape(n, T)
    out["n"] = np.array([n]); out["seq_len"] = np.array([T + 1])
    np.savez(a.out, **out)
    p0 = 1 - np.exp(out["top_logp"].astype(np.float32)).sum(-1)
    print("saved", a.out, "tokens", len(out["nll"]),
          "NLL", out["nll"].mean(), "entropy", out["ent"].mean(),
          "mean residual p0", p0.mean(), "max p0", p0.max(), flush=True)


if __name__ == "__main__":
    main()
