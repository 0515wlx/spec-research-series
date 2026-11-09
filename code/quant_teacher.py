"""Evaluate the ORIGINAL teacher model (Qwen3.6-35B-A3B) after weight-only
quantization, used as a draft against its own full-precision greedy tokens.

This is the "extreme" capacity case: same architecture / total & active params as
the target, only weight precision is reduced.  We quantize every large Linear
(attention, shared expert, lm_head) and every routed expert's fused gate_up/down
tensor on the fly (per selected expert, so no extra persistent memory).

Modes:
  --qat_steps 0   : pure PTQ (RTN)
  --qat_steps N   : short QAT, see train_teacher_qat.py (not in this script)
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from quant import apply_fake_quant, fake_quant_weight, quant_stats
from eval_accept import get_tok, token_matches

TEACHER = "/path/to/models/Qwen3.6-35B-A3B"
D = "/path/to/spec-research/data"


def patch_moe_experts(bits=4, group=128, ste=False):
    """Monkey-patch Qwen3_5MoeExperts.forward to fake-quantize each selected expert."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

    def fwd(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            cur = hidden_states[token_idx]
            gu = self.gate_up_proj[expert_idx]
            dn = self.down_proj[expert_idx]
            if bits < 16:
                gu_q = fake_quant_weight(gu, bits, group)
                dn_q = fake_quant_weight(dn, bits, group)
                if ste:
                    gu = gu + (gu_q - gu).detach()
                    dn = dn + (dn_q - dn).detach()
                else:
                    gu, dn = gu_q, dn_q
            gate, up = F.linear(cur, gu).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = F.linear(h, dn)
            h = h * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))
        return final_hidden_states

    Qwen3_5MoeExperts.forward = fwd


def load_teacher(device=0):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        TEACHER, dtype=torch.bfloat16, device_map={"": device},
        low_cpu_mem_usage=True, attn_implementation="sdpa").eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{D}/teacher_math500_nothink.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--lm_head_bits", type=int, default=None)
    ap.add_argument("--skip_lm_head", action="store_true")
    ap.add_argument("--skip_experts", action="store_true")
    ap.add_argument("--gammas", default="1,2,3,5,8,12,16,24,32")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="teacher_ptq")
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
    print(f"loaded {len(seqs)} seqs, {sum(gen_lens)} gen tokens", flush=True)

    t0 = time.time()
    model = load_teacher(0)
    print(f"teacher loaded {time.time()-t0:.0f}s", flush=True)
    info = {"tag": a.tag, "n_seq": len(seqs), "bits": a.bits, "group": a.group,
            "lm_head_bits": a.lm_head_bits, "skip_lm_head": a.skip_lm_head,
            "skip_experts": a.skip_experts}
    if a.bits < 16:
        names = apply_fake_quant(model, bits=a.bits, group=a.group,
                                 skip_lm_head=a.skip_lm_head,
                                 lm_head_bits=a.lm_head_bits)
        info["quantized_linears"] = len(names)
        if not a.skip_experts:
            patch_moe_experts(a.bits, a.group, ste=False)
        info["sizes"] = quant_stats(model)
        print("quantized linears", len(names), "sizes", info["sizes"], flush=True)

    t0 = time.time()
    matches = token_matches(model, tok, seqs, gen_lens, a.device, bs=a.bs, max_tok=4096)
    info["eval_time"] = time.time() - t0
    info["pos_match_rate"] = float(sum(int(m.sum()) for m in matches) /
                                   max(sum(len(m) for m in matches), 1))

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
    info["per_gamma"] = per
    surv = {}
    for k in [1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64]:
        tot = cnt = 0
        for m in matches:
            if len(m) < k:
                continue
            ok = np.convolve(m.astype(np.int8), np.ones(k, dtype=np.int8), "valid") == k
            tot += int(ok.sum()); cnt += len(ok)
        surv[str(k)] = tot / max(cnt, 1)
    info["survival"] = surv
    print(json.dumps(info, indent=1))
    if a.out:
        json.dump(info, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
