"""Build a reduced-expert MoE draft: keep top-k=8 / expert width 512 (same activated
FFN), but merge the teacher's 256 experts into E clusters per layer.

Initialization is usage-weighted k-means on the router rows (which experts respond to
similar inputs), then usage-weighted averaging of the members' gate/up/down weights.
The result is a normal `Qwen3_5MoeForCausalLM` with `num_experts=E`, so the stock
transformers forward / training loop can be reused.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(__file__))
from draft import TEACHER

D = "/path/to/spec-research/data"
OUT = "/path/to/spec-research/ckpt"


def build_moe_model(num_experts, teacher_dir=TEACHER, dtype=torch.bfloat16, device="cpu"):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(teacher_dir)
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    tc.num_experts = int(num_experts)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(tc)
    model = model.to_empty(device=device)
    model = model.to(dtype)
    model.config.use_cache = True
    return model, tc


def weighted_kmeans(x, w, k, iters=25, seed=0):
    """Cosine k-means on rows of x, weighted by w. Returns hard assignments."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    xn = x.float()
    xn = xn / xn.norm(dim=1, keepdim=True).clamp(min=1e-6)
    w = w.float().clamp(min=1e-8)
    n = xn.shape[0]
    # weighted k-means++ init
    centers = [torch.multinomial(w / w.sum(), 1, generator=g).item()]
    d2 = (1 - xn @ xn[centers[0]]).clamp(min=0) ** 2
    for _ in range(1, k):
        p = (d2 * w)
        if float(p.sum()) <= 0:
            c = int(torch.randint(0, n, (1,), generator=g))
        else:
            c = int(torch.multinomial(p / p.sum(), 1, generator=g))
        centers.append(c)
        d2 = torch.minimum(d2, (1 - xn @ xn[c]).clamp(min=0) ** 2)
    C = xn[centers].clone()
    for _ in range(iters):
        sim = xn @ C.T                    # (n,k)
        a = sim.argmax(1)
        newC = C.clone()
        for j in range(k):
            m = a == j
            if m.any():
                newC[j] = (xn[m] * w[m, None]).sum(0) / w[m].sum()
                newC[j] /= newC[j].norm().clamp(min=1e-6)
            else:
                # re-seed empty cluster with the worst-fit point
                fit = sim.max(1).values
                j2 = int((fit * w).argmin())
                newC[j] = xn[j2]
        if torch.allclose(newC, C, atol=1e-6):
            C = newC
            break
        C = newC
    sim = xn @ C.T
    return sim.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_experts", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_top", type=int, default=0,
                    help="keep this many most-used experts exact; cluster the rest")
    a = ap.parse_args()
    E = a.num_experts
    outdir = a.out or f"{OUT}/moe{E}_merge"
    assert 8 <= E < 256

    st = np.load(f"{D}/calib_stats.npz")
    usage = st["usage"]                      # (L, 256) routing frequency
    nL = usage.shape[0]

    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(TEACHER)
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    tc.num_experts = E
    print("building meta model E=", E, "topk=", tc.num_experts_per_tok, flush=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(tc)
    model = model.to_empty(device="cpu")
    model = model.to(torch.bfloat16)

    idx = json.load(open(f"{TEACHER}/model.safetensors.index.json"))["weight_map"]
    files = sorted(set(idx.values()))
    per_file = {}
    for k, f in idx.items():
        per_file.setdefault(f, []).append(k)

    sd = {}
    pending = {}                             # layer -> dict of teacher expert tensors
    merges = []

    def flush_layer(i):
        d = pending.pop(i)
        gu = d["gu"].float()                 # (256,1024,2048)
        dn = d["dn"].float()                 # (256,2048,512)
        gw = d["gate"].float()               # (256,2048)
        u = torch.as_tensor(usage[i], dtype=torch.float32).clamp(min=1e-8)
        K = min(a.keep_top, E)
        if K > 0:
            order = torch.argsort(-u)
            assign = torch.full((256,), -1, dtype=torch.long)
            assign[order[:K]] = torch.arange(K)
            rest = order[K:]
            if E > K:
                assign[rest] = weighted_kmeans(gw[rest], u[rest], E - K,
                                               seed=a.seed) + K
        else:
            assign = weighted_kmeans(gw, u, E, seed=a.seed)
        gu_o = torch.empty(E, gu.shape[1], gu.shape[2])
        dn_o = torch.empty(E, dn.shape[1], dn.shape[2])
        gw_o = torch.empty(E, gw.shape[1])
        sizes = []
        for c in range(E):
            m = assign == c
            sizes.append(int(m.sum()))
            if not m.any():
                m = torch.zeros_like(assign, dtype=torch.bool)
                m[torch.argmax(u)] = True
            w = u[m]
            if float(w.sum()) <= 0:
                w = torch.ones_like(w)
            w = w / w.sum()
            gu_o[c] = (gu[m] * w[:, None, None]).sum(0)
            dn_o[c] = (dn[m] * w[:, None, None]).sum(0)
            gw_o[c] = (gw[m] * w[:, None]).sum(0)
        sd[f"model.layers.{i}.mlp.experts.gate_up_proj"] = gu_o.to(torch.bfloat16)
        sd[f"model.layers.{i}.mlp.experts.down_proj"] = dn_o.to(torch.bfloat16)
        sd[f"model.layers.{i}.mlp.gate.weight"] = gw_o.to(torch.bfloat16)
        merges.append({"layer": i, "cluster_sizes": sizes,
                       "kept_exact": int(K), "dropped": int((assign < 0).sum()),
                       "top_experts": torch.argsort(-u)[:8].tolist()})

    t0 = time.time()
    for fi, f in enumerate(files):
        sa = load_file(os.path.join(TEACHER, f), device="cpu")
        for k in per_file[f]:
            if k.startswith("mtp.") or ".visual" in k:
                continue
            if k == "lm_head.weight":
                sd["lm_head.weight"] = sa[k].to(torch.bfloat16)
                continue
            if not k.startswith("model.language_model."):
                continue
            kk = "model." + k[len("model.language_model."):]
            if ".mlp.experts.gate_up_proj" in kk:
                i = int(kk.split(".")[2])
                pending.setdefault(i, {})["gu"] = sa[k]
            elif ".mlp.experts.down_proj" in kk:
                i = int(kk.split(".")[2])
                pending.setdefault(i, {})["dn"] = sa[k]
            elif kk.endswith(".mlp.gate.weight"):
                i = int(kk.split(".")[2])
                pending.setdefault(i, {})["gate"] = sa[k]
            else:
                sd[kk] = sa[k].to(torch.bfloat16)
        sa = None
        for i in list(pending.keys()):
            if all(x in pending[i] for x in ("gu", "dn", "gate")):
                flush_layer(i)
        print(f"shard {fi+1}/{len(files)}  merged {len(merges)}/{nL}  {time.time()-t0:.0f}s", flush=True)
    for i in list(pending.keys()):
        flush_layer(i)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing", len(missing), missing[:8], flush=True)
    print("unexpected", len(unexpected), unexpected[:8], flush=True)
    assert not [m for m in missing if "rotary" not in m and "inv_freq" not in m], missing[:20]

    os.makedirs(outdir, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              f"{outdir}/model.safetensors")
    meta = {"arch": "moe", "num_experts": E, "top_k": tc.num_experts_per_tok,
            "moe_inter": tc.moe_intermediate_size,
            "shared_inter": tc.shared_expert_intermediate_size,
            "init": "router-kmeans-usage-avg", "merges": merges}
    json.dump(meta, open(f"{outdir}/meta.json", "w"))
    nparam = sum(p.numel() for p in model.parameters())
    print("saved", outdir, nparam / 1e9, "B params", flush=True)


if __name__ == "__main__":
    main()
