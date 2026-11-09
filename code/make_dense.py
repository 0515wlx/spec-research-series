"""Merge teacher MoE experts into a width-matched dense FFN (train-free conversion).

Exact-for-fixed-weights merge: concatenate the top-8 most-used experts' gate/up rows and
their down columns scaled by mean routing weight, plus the shared expert scaled by mean gate.
"""
import argparse, json, os
import numpy as np
import torch
from safetensors.torch import load_file

T = "/path/to/models/Qwen3.6-35B-A3B"
D = "/path/to/spec-research/data"
OUT = "/path/to/spec-research/ckpt"


def main(sel_mode="usage"):
    idx = json.load(open(f"{T}/model.safetensors.index.json"))["weight_map"]
    files = sorted(set(idx.values()))
    print("loading", len(files), "shards")
    sa = {}
    for f in files:
        sa.update(load_file(os.path.join(T, f)))
    st = np.load(f"{D}/calib_stats.npz")
    top8, mean_w, shared_w = st["top8"], st["mean_w"], st["shared_w"]

    sd = {}
    for k, v in sa.items():
        if k.startswith("model.language_model."):
            sd["model." + k[len("model.language_model."):]] = v
        elif k == "lm_head.weight":
            sd[k] = v
    nL = 40
    meta_layers = []
    for i in range(nL):
        pre = f"model.language_model.layers.{i}.mlp"
        gu = sa[f"{pre}.experts.gate_up_proj"]        # (256, 1024, 2048)
        dn = sa[f"{pre}.experts.down_proj"]           # (256, 2048, 512)
        sg = sa[f"{pre}.shared_expert.gate_proj.weight"]  # (512, 2048)
        su = sa[f"{pre}.shared_expert.up_proj.weight"]
        sdn = sa[f"{pre}.shared_expert.down_proj.weight"]  # (2048, 512)

        e = torch.as_tensor(top8[i])
        w = torch.as_tensor(mean_w[i][e], dtype=torch.float32)
        ws = float(shared_w[i])
        gu_e = gu[e].float()   # (8,1024,2048)
        gate, up = gu_e[:, :512], gu_e[:, 512:]
        gate = gate.reshape(8 * 512, 2048)
        up = up.reshape(8 * 512, 2048)
        down = torch.cat([dn[e][k].float() * w[k] for k in range(8)] +
                         [sdn.float() * ws], dim=1)  # (2048, 4608)
        dense_gate = torch.cat([gate, sg.float()], 0)
        dense_up = torch.cat([up, su.float()], 0)

        o = f"model.layers.{i}.mlp"
        sd[f"{o}.gate_proj.weight"] = dense_gate.to(torch.bfloat16)
        sd[f"{o}.up_proj.weight"] = dense_up.to(torch.bfloat16)
        sd[f"{o}.down_proj.weight"] = down.to(torch.bfloat16)
        meta_layers.append({"layer": i, "experts": e.tolist(), "w": w.tolist(), "ws": ws})
        if i in (0, 20, 39):
            print(i, "w", np.round(w.numpy(), 3), "ws", round(ws, 3))

    # drop any leftover MoE / mtp / vision keys
    bad = [k for k in sd if ".mlp.experts" in k or "shared_expert" in k or k.startswith("mtp") or ".visual" in k]
    for k in bad:
        del sd[k]

    outdir = f"{OUT}/dense_merge_{sel_mode}"
    os.makedirs(outdir, exist_ok=True)
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in sd.items()}, f"{outdir}/model.safetensors")
    with open(f"{outdir}/meta.json", "w") as f:
        json.dump({"inter": 4608, "init": "merge", "layers": meta_layers}, f)
    print("saved", outdir, sum(v.numel() for v in sd.values()) / 1e9, "B")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", default="usage")
    main(ap.parse_args().sel)
