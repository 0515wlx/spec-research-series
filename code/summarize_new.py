"""Summarize long-gamma acceptance for the QAT / reduced-expert MoE drafts."""
import json, os, sys

D = "/path/to/spec-research/out"
MODELS = [
    ("dense-distill", "eval_dense_distill_full.json", 3.434, 6.87),
    ("dense-RL", "eval_dense_rl_full.json", 3.434, 6.87),
    ("dense-ctrl (1.17M)", "eval_dense_ctrl_full.json", 3.434, 6.87),
    ("bf16-ctrl (1.17M)", "eval_dense_bf16ctrl_full.json", 3.434, 6.87),
    ("PTQ-W4", "eval_ptq_w4_full.json", 3.434, 2.49),
    ("QAT-W4", "eval_dense_qat_w4_full.json", 3.434, 2.49),
    ("MoE8 keep8", "eval_moe8_keep8_full.json", 3.435, 6.87),
    ("MoE16 keep8", "eval_moe16_keep8_full.json", 4.442, 8.88),
]
GAMMAS = [1, 2, 3, 5, 8, 12, 16, 24, 32]
TEACHER_GB = 69.3   # 34.66B * 2 bytes


def load(name):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def main():
    rows = []
    for tag, fn, params, gb in MODELS:
        d = load(fn)
        if d is None:
            continue
        rows.append((tag, params, gb, d))
    print("| model | params(B) | bf16 GB | token top-1 | " +
          " | ".join(f"τ(γ={g})" for g in GAMMAS) + " |")
    print("|" + "---|" * (4 + len(GAMMAS)))
    for tag, params, gb, d in rows:
        taus = [d["per_gamma"][str(g)]["tau"] if str(g) in d["per_gamma"] else float("nan")
                for g in GAMMAS]
        print(f"| {tag} | {params:.2f} | {gb:.2f} | {d['pos_match_rate']*100:.2f}% | " +
              " | ".join(f"{t:.2f}" for t in taus) + " |")
    print()
    KS = [1, 2, 3, 5, 8, 16, 32, 64]
    print("| model | " + " | ".join(f"P(run≥{k})" for k in KS) + " |")
    print("|" + "---|" * (1 + len(KS)))
    for tag, params, gb, d in rows:
        sv = d.get("survival", {})
        print(f"| {tag} | " + " | ".join(
            f"{sv.get(str(k), sv.get(k, float('nan')))*100:.1f}%" for k in KS) + " |")
    print()
    print("| model | c_bw=draft/teacher bytes | " +
          " | ".join(f"speedup γ={g}" for g in [5, 8, 12, 24]) + " |")
    print("|" + "---|" * (2 + 4))
    for tag, params, gb, d in rows:
        c = gb / TEACHER_GB
        sp = []
        for g in [5, 8, 12, 24]:
            t = d["per_gamma"][str(g)]["tau"] if str(g) in d["per_gamma"] else float("nan")
            sp.append(t / (1 + g * c))
        print(f"| {tag} | {c:.4f} | " + " | ".join(f"{x:.2f}×" for x in sp) + " |")


if __name__ == "__main__":
    main()
