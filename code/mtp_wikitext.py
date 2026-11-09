"""MTP head metrics on WikiText using the cached teacher hidden states."""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from wikitext_common import OUT, load_chunks, metric_arrays
from mtp_eval import build_mtp
from quant_teacher import load_teacher
from transformers import AutoConfig
import copy
from transformers.masking_utils import create_causal_mask


def build(teacher):
    cfg = AutoConfig.from_pretrained("/path/to/models/Qwen3.6-35B-A3B")
    tc = cfg.text_config
    mcfg = copy.deepcopy(tc)
    mcfg.layer_types = ["full_attention"]
    mcfg.num_hidden_layers = 1
    mcfg._attn_implementation = "sdpa"
    mcfg._experts_implementation = "eager"
    mtp = build_mtp(tc, mcfg).to("cuda").eval()
    return mtp, mcfg


@torch.no_grad()
def mtp_logits(teacher, mtp, mcfg, tokens, hidden):
    # tokens: [1,T] current tokens; hidden: [1,T,H] pre-norm last-layer
    e = teacher.model.embed_tokens(tokens)
    x = mtp.fc(torch.cat([mtp.ne(e), mtp.nh(hidden.to(torch.bfloat16))], -1))
    T = x.shape[1]
    pos = torch.arange(T, device=x.device).unsqueeze(0)
    pe = teacher.model.rotary_emb(x, pos)
    mask = create_causal_mask(mcfg, inputs_embeds=x, attention_mask=None,
                              past_key_values=None, position_ids=pos)
    h = mtp.layer(x, position_embeddings=pe, attention_mask=mask,
                  position_ids=pos, past_key_values=None, use_cache=False)
    return teacher.lm_head(mtp.norm(h))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=os.path.join(OUT, "wikitext_teacher.npz"))
    ap.add_argument("--out", default=os.path.join(OUT, "wikitext_mtp.npz"))
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    td = np.load(a.teacher)
    n, T = td["target"].shape
    full = load_chunks(n, T + 1)
    teacher = load_teacher(0)
    mtp, mcfg = build(teacher)
    if a.verify:
        ci = 0
        tok = full[ci:ci + 1, :-1].cuda()
        h = torch.from_numpy(td["hidden"][ci:ci + 1]).float().cuda()
        lg = mtp_logits(teacher, mtp, mcfg, tok, h)[0]
        from transformers import DynamicCache
        cache = DynamicCache(config=mcfg)
        for p in [0, 1, 2, 10, 19]:
            e1 = teacher.model.embed_tokens(tok[:, p:p + 1])
            x1 = mtp.fc(torch.cat([mtp.ne(e1), mtp.nh(h[:, p:p + 1].to(torch.bfloat16))], -1))
            p4 = torch.full((4, 1, 1), p, device="cuda", dtype=torch.long)
            pe1 = teacher.model.rotary_emb(x1, p4[1:])
            h1 = mtp.layer(x1, position_embeddings=pe1, attention_mask=None,
                           position_ids=p4[0], past_key_values=cache, use_cache=True)
            l1 = teacher.lm_head(mtp.norm(h1))[0, -1]
            print(p, "chunk argmax", lg[p].argmax().item(), "token argmax", l1.argmax().item(),
                  "maxdiff", (lg[p] - l1).abs().max().item())
        return
    out = {k: [] for k in ["nll", "match", "in5", "rank", "kl_td", "kl_dt", "tv", "js",
                           "ent", "p0", "q0", "pred", "target", "chunk"]}
    t0 = time.time()
    for ci in range(n):
        tok = full[ci:ci + 1, :-1].cuda()
        h = torch.from_numpy(td["hidden"][ci:ci + 1]).float().cuda()
        logits = mtp_logits(teacher, mtp, mcfg, tok, h).float()
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
            print(f"chunk {ci+1}/{n} {time.time()-t0:.0f}s", flush=True)
    out = {k: np.concatenate(v) for k, v in out.items()}
    np.savez(a.out, **out)
    print(f"MTP: NLL {out['nll'].mean():.4f} ppl {np.exp(out['nll'].mean()):.2f} "
          f"match {out['match'].mean():.4f} in5 {out['in5'].mean():.4f} "
          f"KL_TD {out['kl_td'].mean():.4f} KL_DT {out['kl_dt'].mean():.4f} "
          f"TV {out['tv'].mean():.4f} JS {out['js'].mean():.4f} "
          f"rank {out['rank'].mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
