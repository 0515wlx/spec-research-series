"""Per-step latency vs concurrency for teacher / dense draft / native MTP head."""
import sys, os, json, copy, time, argparse, torch
sys.path.insert(0, 'code')
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from draft import load_dense
from mtp_eval import build_mtp

T = '/path/to/models/Qwen3.6-35B-A3B'


def timeit(fn, warm=1, rep=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", default="1,2,4,8,16,32,48,64")
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--out", default="out/concurrency.json")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(T)
    cfg = AutoConfig.from_pretrained(T); tc = cfg.text_config
    teacher = AutoModelForCausalLM.from_pretrained(T, dtype=torch.bfloat16, device_map={'': 0},
                                                   low_cpu_mem_usage=True).eval()
    mcfg = copy.deepcopy(tc); mcfg.layer_types = ['full_attention']; mcfg.num_hidden_layers = 1
    mcfg._attn_implementation = 'sdpa'; mcfg._experts_implementation = 'eager'
    mtp = build_mtp(tc, mcfg).to('cuda').eval()

    res = {}
    draft, _ = load_dense('ckpt/dense_distill_s1', dtype=torch.bfloat16, device='cuda:1')
    draft.eval()
    for B in [int(x) for x in a.bs.split(',')]:
        prompt = torch.randint(100, 2000, (B, a.len), device='cuda')
        one = torch.randint(100, 2000, (B, 1), device='cuda')
        k = torch.randint(100, 2000, (B, a.gamma + 1), device='cuda')
        with torch.no_grad():
            ck = teacher(input_ids=prompt, use_cache=True).past_key_values
            t_teacher = timeit(lambda: teacher(input_ids=one, past_key_values=ck, use_cache=True))
        del ck
        torch.cuda.empty_cache()
        with torch.no_grad():
            ck2 = teacher(input_ids=prompt, use_cache=True).past_key_values
            t_verify = timeit(lambda: teacher(input_ids=k, past_key_values=ck2, use_cache=True))
        del ck2
        torch.cuda.empty_cache()
        t_prefill = 0.0

        # dense draft: prefill then 1-token step (separate GPU to avoid OOM)
        p1 = prompt.to('cuda:1'); o1 = one.to('cuda:1')
        with torch.no_grad():
            dk = draft(input_ids=p1, use_cache=True).past_key_values
            t_dense = timeit(lambda: draft(input_ids=o1, past_key_values=dk, use_cache=True))

        # MTP head: 1 draft step (1 layer, needs h from teacher + token emb)
        Hd = torch.randn(1, 1, tc.hidden_size, device='cuda', dtype=torch.bfloat16)
        xin = torch.randint(100, 2000, (1, 1), device='cuda')
        p4 = torch.full((4, 1, 1), 10, device='cuda', dtype=torch.long)

        def mtp_step():
            e = teacher.model.embed_tokens(xin)
            inpM = mtp.fc(torch.cat([mtp.ne(e), mtp.nh(Hd)], -1))
            pe = teacher.model.rotary_emb(inpM, p4[1:])
            return mtp.layer(inpM, position_embeddings=pe, attention_mask=None,
                             position_ids=p4[0], past_key_values=None, use_cache=False)
        with torch.no_grad():
            t_mtp = timeit(mtp_step)

        res[B] = {"teacher_step": t_teacher, "teacher_verify": t_verify, "teacher_prefill": t_prefill,
                  "dense_step": t_dense, "mtp_step": t_mtp}
        print(B, {k: round(v, 4) for k, v in res[B].items()}, flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == "__main__":
    main()
