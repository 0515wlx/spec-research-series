"""Real greedy speculative-decoding loop (sanity check of the teacher-forced acceptance estimate).

Kept deliberately simple: the teacher re-runs the full prefix each round (no KV rollback needed),
the draft proposes gamma tokens autoregressively.
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(__file__))
from draft import load_dense

T = "/path/to/models/Qwen3.6-35B-A3B"


@torch.no_grad()
def sd_generate(draft, teacher, tok, prompt_ids, gamma=5, max_new=256, device="cuda:0"):
    cur = list(prompt_ids)
    n_acc = n_prop = steps = 0
    emitted = []
    while len(emitted) < max_new:
        ids = torch.tensor(cur, device=device, dtype=torch.long)[None]
        dids = ids.to("cuda:1")
        o = draft(input_ids=dids, use_cache=True)
        cache = o.past_key_values
        prop = []
        nxt = o.logits[0, -1].argmax()
        prop.append(nxt)
        for _ in range(gamma - 1):
            o = draft(input_ids=nxt.view(1, 1).to("cuda:1"), past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            nxt = o.logits[0, -1].argmax()
            prop.append(nxt)
        to = teacher(input_ids=ids, use_cache=False)
        ver = to.logits[0, -1:].new_empty(0)
        # teacher's greedy continuation after the prefix is obtained one token at a time? No:
        # verify with a single forward over prefix+proposals (teacher forcing gives all gammas)
        pv = torch.stack(prop).view(1, -1).to(device)
        o2 = teacher(input_ids=torch.cat([ids, pv], 1), use_cache=False)
        ver = o2.logits[0, len(cur) - 1: len(cur) - 1 + gamma].argmax(-1)
        acc = 0
        for i in range(gamma):
            if prop[i].item() == ver[i].item():
                acc += 1
            else:
                break
        n_acc += acc; n_prop += gamma; steps += 1
        new = [p.item() for p in prop[:acc]]
        new.append(ver[acc].item() if acc < gamma else ver[-1].item())
        cur.extend(new)
        emitted.extend(new)
        if new[-1] in (248046, 248044):
            break
    return {"alpha": n_acc / max(n_prop, 1), "tau": len(emitted) / max(steps, 1),
            "steps": steps, "ntok": len(emitted)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--max_new", type=int, default=256)
    a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(T)
    draft, _ = load_dense(a.ckpt, dtype=torch.bfloat16, device="cuda:1")
    draft.eval()
    teacher = AutoModelForCausalLM.from_pretrained(
        T, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True).eval()
    recs = [json.loads(l) for l in open("/path/to/spec-research/data/teacher_math500_nothink.jsonl")]
    for r in recs[:a.n]:
        t0 = time.time()
        out = sd_generate(draft, teacher, tok, r["input_ids"], a.gamma, a.max_new, device="cuda:0")
        print(json.dumps(out), f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
