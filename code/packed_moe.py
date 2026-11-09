"""Packed W4 storage for the teacher's routed experts + on-the-fly dequantization.

The 32B routed-expert weights are the reason the full 35B model cannot be trained
on one 80GB GPU.  We store them as packed int4 (+ per-group scales) and dequantize
only the few selected experts in each forward.  This makes the full model ~21GB,
so the cheap non-expert parameters can be QAT-fine-tuned with AdamW on one GPU.
"""
import torch
import torch.nn.functional as F


def pack_tensor(w, bits=4, group=128):
    """w: [N, O, I] -> packed uint8 [N, O, I_pad/2], scale [N, O, I_pad/group]."""
    assert bits == 4
    wf = w.float()
    N, O, I = wf.shape
    g = min(group, I)
    pad = (-I) % g
    if pad:
        wf = F.pad(wf, (0, pad))
    wg = wf.reshape(N, O, -1, g)
    qmax = 7
    scale = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    q = (wg / scale).round().clamp(-qmax - 1, qmax).to(torch.int8)
    q = (q & 0xF).reshape(N, O, -1, 2)
    packed = (q[..., 0] | (q[..., 1] << 4)).to(torch.uint8)
    return packed, scale.squeeze(-1).to(torch.bfloat16)


def unpack_tensor(packed, scale, out_dim, in_dim, group=128):
    """packed [N,O,I_pad/2], scale [N,O,I_pad/group] -> bf16 [N,O,I]."""
    lo = (packed & 0xF).to(torch.int8)
    hi = ((packed >> 4) & 0xF).to(torch.int8)
    q = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], packed.shape[1], -1)
    q = torch.where(q >= 8, q - 16, q).float()
    # q is [N,O,I_pad]
    dq = q.reshape(q.shape[0], q.shape[1], -1, group) * scale.float().unsqueeze(-1)
    dq = dq.reshape(q.shape[0], q.shape[1], -1)
    return dq[:, :, :in_dim].to(torch.bfloat16)


def pack_experts_inplace(model, bits=4, group=128):
    """Replace each layer's bf16 expert Parameters with packed buffers (in place)."""
    nbytes = 0
    for layer in model.model.layers:
        exp = layer.mlp.experts
        if hasattr(exp, "gate_up_packed"):
            continue
        gu = exp.gate_up_proj.data
        dn = exp.down_proj.data
        gu_p, gu_s = pack_tensor(gu, bits, group)
        dn_p, dn_s = pack_tensor(dn, bits, group)
        nbytes += gu_p.numel() + dn_p.numel() + gu_s.numel() * 2 + dn_s.numel() * 2
        # drop originals first, then install buffers
        del exp.gate_up_proj
        del exp.down_proj
        exp.register_buffer("gate_up_packed", gu_p, persistent=False)
        exp.register_buffer("gate_up_scale", gu_s, persistent=False)
        exp.register_buffer("down_packed", dn_p, persistent=False)
        exp.register_buffer("down_scale", dn_s, persistent=False)
        exp._packed_bits = bits
        exp._packed_group = group
        del gu, dn, gu_p, gu_s, dn_p, dn_s
    torch.cuda.empty_cache()
    return nbytes


def patch_packed_experts():
    """Monkey-patch Qwen3_5MoeExperts.forward to dequantize selected experts."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

    def fwd(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        g = self._packed_group
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            cur = hidden_states[token_idx]
            gu = unpack_tensor(self.gate_up_packed[expert_idx:expert_idx + 1],
                               self.gate_up_scale[expert_idx:expert_idx + 1],
                               self.gate_up_packed.shape[1],
                               self.gate_up_packed.shape[2] * 2, g)[0]
            dn = unpack_tensor(self.down_packed[expert_idx:expert_idx + 1],
                               self.down_scale[expert_idx:expert_idx + 1],
                               self.down_packed.shape[1],
                               self.down_packed.shape[2] * 2, g)[0]
            gate, up = F.linear(cur, gu).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = F.linear(h, dn)
            h = h * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))
        return final_hidden_states

    Qwen3_5MoeExperts.forward = fwd
