# The Compression Floor（2026-11-06）

> 新主论文：`paper/main.tex`（旧容量扫描论文保留为 `paper/main_v1_offtheshelf.tex`）。
> 本文件记录理论、实验、复现命令和 claim 边界。

## 1. 一句话结论

把 target 的 FFN 中间宽度压小、当作自己的 draft，在 **post-hoc**（pruning + 单层 reconstruction）下
不能回本：删掉的 channel 造成不可消除的 logit 扰动，top-1 rejection 的数值证书在 raw OOD 的
每一个压缩宽度上都超过 break-even 阈值。关键是成本模型不依赖 batch size：由
`rho_B >= rho_FLOP`、`tau_B >= 1` 得 `C_B >= gamma*rho_FLOP + 1`，所以乐观 FLOP frontier
是任意 batch size 的下界。Qwen3.5-9B 上最好的 raw-OOD speedup 只有 0.83×（99% 宽度，
measured batch-1 q）；在 10% 和 50–99% 宽度，实测 batch-1 成本满足 gamma*rho_1+tau_1>5，
即使 q=0 也不能回本。4B 与跨家族复现同样成立。
即使做标准单层 reconstruction（把 pruning 后的 down projection 用 ridge 回归拟合回原层输出），
层 MSE 降 34–36%，但 q 基本不变：删 2% 内部维度仍有 MATH q=0.016、raw OOD q=0.096–0.097。

这不是“所有 compressed draft 都不行”的结论：fine-tune 过的窄 draft、单独预训练的小 draft、
tree/top-k drafting、compute-bound serving 是不同对象，论文明确排除。

## 2. 理论链

### 2.1 为什么 exact post-hoc 宽度压缩只能是 channel deletion

FFN 为

```
y = W_d ( sigma(W_g x) ⊙ W_u x ).
```

若用线性投影 P 把 h 维中间空间压到 r 维，压缩块为

```
P^T sigma(P W_g x) ⊙ (P W_u x),  然后 W_d P^T.
```

要让它对所有 x 都等于原函数，P 必须是 signed coordinate selection；一般旋转/投影会改变
elementwise nonlinearity 的行为。因此“减小 FFN 内部维度”的严格 post-hoc 形式是
structured channel pruning。独立的 gate/up SVD 和 shared rotation 在实验里都更差
（shared rotation 在所有 rank 上 q=1）。

### 2.2 删除 channel 的扰动是精确的

删除集合 S：

```
Delta y(x) = sum_{j in S} a_j(x) W_d[:, j],   a = sigma(W_g x) ⊙ W_u x.
```

一阶重要性分数（实验中用来选 channel）：

```
s_j = E|a_j| * ||W_d[:, j]||_2.
```

### 2.3 Top-2 rejection certificate

令 m = log P(top1) - log P(top2)，Delta_m 为压缩 draft 对这个 margin 的扰动。若

```
m + Delta_m < 0,
```

则 draft 把 target 的 top-2 排到 top-1 前面，token 必被 reject。因此

```
q >= q_lb := Pr(m + Delta_m < 0).
```

这是确定性的、非空的数值下界；它不需要 Markov 松弛，也不会在 raw OOD 上 >1。
实验中 q_lb 约为观测 q 的 0.55–0.70。

### 2.4 Batch-free 成本与 break-even

设 \(T_T(B,n)\) 为 batch size \(B\) 下 target 对 \(n\) 个位置的 pass 时间，
\(T_D(B,n)\) 为 draft 的对应时间：

```
rho_B = T_D(B,1) / T_T(B,1),
tau_B = T_T(B,gamma+1) / T_T(B,1),
A(alpha,gamma) = sum_{i=0}^{gamma} alpha^i,
S_B = A(1-q,gamma) / (gamma*rho_B + tau_B).
```

这是 Su et al. (2023) batched runtime model 的 per-sequence 形式；B=1 回到 Leviathan et al.
的单流模型。break-even：

```
A(1-q,gamma) >= gamma*rho_B + tau_B.
```

因为压缩 draft 不可能低于其 nominal FLOP ratio（\(\rho_B\ge\rho_{\rm FLOP}\)），且
verification 不可能比单个 target token pass 更便宜（\(\tau_B\ge1\)），所以

```
C_B = gamma*rho_B + tau_B >= gamma*rho_FLOP + 1 = C_FLOP.
```

于是 \(q^\star_{\rm FLOP}\) 是任意 batch size 下可接受 rejection 的上界。实测 batch-1
\(\rho_1\approx0.96\)–0.98、\(\tau_1\approx1.30\)–1.35 只是比这个下界更差；
10% 和 50–99% 宽度满足 \(\gamma\rho_1+\tau_1>\gamma+1\)，是 batch-1 cost-only no-go。
batch 变大只会让 \(\tau_B\) 趋向 \(\gamma+1\)，不能把成本压到 \(C_{\rm FLOP}\) 以下。

## 3. 实验

### 3.1 设置

- 主实验：Qwen3.5-9B，target 和 compressed draft 是同一个模型；100% 宽度时 q=0。
- 复现：Qwen3.5-4B。
- 数据：MATH-500 target-generated（in-domain）；WikiText-2 raw、FineWeb-Edu raw（raw OOD）；
  9B target-generated WikiText/FineWeb（matched OOD）。
- 压缩：activation-aware structured pruning，200k target-aligned tokens 校准；
  13 个宽度 0.05–1.0。
- 统计：sequence-cluster bootstrap（2000 次），不是 i.i.d. token Hoeffding。
- 成本：名义 FLOP 比 `rho_FLOP = 1 - 0.61(1 - r/h)`；实测 batch-1 latency（512-token prefix，
  padded narrow MLP，20 次交替 forward）。

### 3.2 主结果（9B，activation-aware）

| width | rho_FLOP | q* (gamma=4) | MATH q | Wiki raw q | FineWeb raw q | Wiki gen q |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.422 | 0.3168 | 0.5846 | 0.8319 | 0.8161 | 0.7445 |
| 0.10 | 0.452 | 0.2933 | 0.4350 | 0.7782 | 0.7616 | 0.6079 |
| 0.20 | 0.513 | 0.2498 | 0.2808 | 0.7033 | 0.6914 | 0.3716 |
| 0.30 | 0.574 | 0.2102 | 0.2098 | 0.6565 | 0.6284 | 0.2258 |
| 0.50 | 0.696 | 0.1400 | 0.1297 | 0.5017 | 0.5047 | 0.1530 |
| 0.70 | 0.817 | 0.0791 | 0.0831 | 0.3618 | 0.3643 | 0.1145 |
| 0.80 | 0.878 | 0.0513 | 0.0637 | 0.2876 | 0.2926 | 0.0963 |
| 0.85 | 0.909 | 0.0379 | 0.0526 | 0.2513 | 0.2518 | 0.0862 |
| 0.90 | 0.939 | 0.0250 | 0.0399 | 0.2080 | 0.2085 | 0.0750 |
| 0.95 | 0.970 | 0.0123 | 0.0260 | 0.1502 | 0.1551 | 0.0596 |
| 0.98 | 0.988 | 0.0049 | 0.0157 | 0.0968 | 0.0959 | 0.0415 |
| 0.99 | 0.994 | 0.0024 | 0.0112 | 0.0677 | 0.0649 | 0.0334 |
| 1.00 | 1.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**证书：** raw OOD 的 0.05–0.99 全部 `q_lb > q*`（certified no-go）。例如 99% 宽度：
`q_lb=0.038–0.046`，`q*=0.0024`。MATH 在 ≥85% 宽度也被证书覆盖；30–50% 的 FLOP-model
窗口（S≈1.00–1.02）在实测 latency 下关闭。

### 3.3 实测 batch-1 成本组件

| width | rho | tau | gamma*rho+tau | S_max(q=0) | q MATH | S MATH | q Wiki |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.99 | 0.977 | 1.346 | 5.254 | 0.952 | 0.0112 | 0.93 | 0.0677 |
| 0.98 | 0.983 | 1.344 | 5.277 | 0.948 | 0.0157 | 0.92 | 0.0968 |
| 0.90 | 0.982 | 1.337 | 5.265 | 0.950 | 0.0399 | 0.88 | 0.2080 |
| 0.70 | 0.985 | 1.346 | 5.285 | 0.946 | 0.0831 | 0.80 | 0.3618 |
| 0.50 | 0.961 | 1.322 | 5.166 | 0.968 | 0.1297 | 0.75 | 0.5017 |
| 0.30 | 0.916 | 1.239 | 4.905 | 1.019 | 0.2098 | 0.67 | 0.6565 |
| 0.10 | 0.973 | 1.303 | 5.193 | 0.963 | 0.4350 | 0.42 | 0.7782 |

结论：
- 10% 和 50–99% 宽度：\(\gamma\rho+\tau>5\)，**即使 q=0 也比 target 慢**。
- 30%：成本条件勉强满足（4.91<5），但 q=0.21/0.66 给出 S=0.67/0.31。
- 没有任何压缩宽度同时满足成本条件和 rejection 条件。
- 最优 raw-OOD 点：0.83×（99% width，measured q）；cost-only 上限 0.95×。

### 3.4 4B 复现

4B 的 MLP share 更小（0.539），同样宽度的 FLOP 节省更少。raw OOD 在每个压缩宽度
（10–99%）都被证书判定 no-go；99% 宽度 `q_lb=0.039/0.042`，`q*=0.0022`；FLOP τ=1 的 `S_OOD=0.89` 是乐观值，
实测成本因 attention/embedding 路径不变而更低。完整表见论文 Appendix C。

### 3.5 匹配成本对比

对同一个 9B target：

| draft | rho | MATH q | Wiki raw q |
|---|---:|---:|---:|
| pretrained 0.8B | 0.095 | 0.0958 | 0.3713 |
| pretrained 2B | 0.237 | 0.0731 | 0.3030 |
| pretrained 4B | 0.530 | 0.0474 | 0.2236 |
| compressed 9B, width 0.20 | 0.513 | 0.2808 | 0.7033 |

同样成本下，训练过的小 draft 远好于压缩的大 draft。压缩保留 target 的“怪癖”，但丢掉了
SD 需要的 low-margin top-1 agreement。

### 3.6 Baseline

- activation-aware（主）在 50% 宽度：MATH 0.130 / Wiki 0.502。
- weight-norm 在 50% 宽度：MATH 0.679 / Wiki 0.889。
- independent SVD 在 rank 0.5：MATH 0.749 / Wiki 0.820；rank 0.1 已经 0.95。
- shared rotation：所有 rank q=1（与 Lemma 1 一致）。

### 3.7 单层 reconstruction 不能消除 floor

审稿人可能会说：实际部署会继续训练/拟合压缩后的层。我们做了标准的 post-pruning recovery：
对每个 layer，保留 activation-aware top-r channels 后，用 residual ridge regression 把 down
projection 拟合回原层输出；ridge 系数在 held-out calibration split 上按重建 MSE 选择，且以
pruned down 为起点，因此不会比不拟合更差。

结果（Qwen3.5-9B）：

| width | MSE reduction | q MATH before → after | q Wiki before → after | q FineWeb before → after |
|---:|---:|---:|---:|---:|
| 0.99 | 34.2% | 0.0112 → 0.0111 | 0.0677 → 0.0685 | 0.0649 → 0.0661 |
| 0.98 | 34.9% | 0.0157 → 0.0156 | 0.0968 → 0.0974 | 0.0959 → 0.0960 |
| 0.95 | 35.3% | 0.0260 → 0.0253 | 0.1502 → 0.1530 | 0.1551 → 0.1569 |
| 0.90 | 36.0% | 0.0399 → 0.0389 | 0.2080 → 0.2116 | 0.2085 → 0.2128 |

所有 layer 的 MSE 都改善；paired sequence bootstrap 的 Δq 都在 ±0.003 内，TOST 在 ±0.005
内等价。**结论：MSE 最优的单层拟合不改变 top-1 rejection。** 这不是优化失败，而是目标错配：
MSE 按输出能量加权，top-1 rejection 由 low-margin tokens 决定；35% 的 MSE 下降可以完全不触动
这些 low-margin flips。

因此，“删掉 2% 内部维度 → MATH q=0.016 > 0.01”在**拟合之后**仍然成立；而 0.98 width 的
break-even 阈值是 q*=0.0049，q 仍高 3 倍以上。4B 在 0.98 width 上同样：reconstruction 后
MATH q=0.018。

### 3.8 跨家族 Qwen3-4B

为避免“只测 Qwen3.5 家族”的质疑，用 Qwen3-4B-Instruct-2507（不同家族、不同 tokenizer）做小复现：
target 和 compressed draft 都是 Qwen3-4B；数据是 100 条 target-generated MATH + 100 条
retokenized raw WikiText/FineWeb。

| width | rho | q* | MATH q (q_lb) | Wiki q (q_lb) | FineWeb q (q_lb) | certified |
|---:|---:|---:|---:|---:|---:|---:|
| 0.95 | 0.967 | 0.0136 | 0.036 (0.029) | 0.155 (0.106) | 0.158 (0.109) | yes |
| 0.98 | 0.987 | 0.0054 | 0.022 (0.018) | 0.100 (0.070) | 0.104 (0.072) | yes |
| 0.99 | 0.993 | 0.0027 | 0.016 (0.012) | 0.068 (0.048) | 0.070 (0.049) | yes |

0.98 width 做 ridge reconstruction：MSE 降 35.1%，MATH q 0.0224→0.0220，Wiki 0.1005→0.1010，
FineWeb 0.1038→0.1035。跨家族 + reconstruction 后仍然 no-go。

### 3.9 Gradient-fit negative control

为了排除“只拟合 down projection 不够”的质疑，我们还试了 per-layer Adam 联合拟合
gate/up/down（20 步，lr=1e-6，gradient clip 0.1，按 held-out layer MSE 选 best state）。
验证 MSE 从 9.55e-7 降到 8.37e-7，但 full-model q 变成 MATH 0.284、Wiki 0.716、FineWeb 0.710。
full-model validation 直接拒绝这个 fit。

这不是说“训练一定不行”，而是说：**layer-wise MSE 不是 top-1 rejection 的充分选择标准**；
一个在层 MSE 上更好的 fit 可能在下游 top-1 上更差。主文因此用保守的 ridge readout 作为
recovery baseline；gradient fit 只作为 negative control 放在 Appendix B。

### 3.10 额外 OOD 与第二个跨家族模型

- **Llama-3.1-8B-Instruct**（不同 tokenizer）：0.95/0.98/0.99 width 在 MATH/Wiki/FineWeb
  全部 certified no-go。0.98 width：MATH 0.0189、Wiki 0.0807、FineWeb 0.0767；
  ridge reconstruction MSE 降 34.9%，q 0.0181/0.0812/0.0786（不变）。
- **Code（The Stack）** 和 **MT-Bench first-turn**：9B target 生成，0.98 width；
  q=0.0145/0.0629，q_lb=0.0102/0.0483，q*=0.0049，均 certified。
- 因此 evidence 不再只依赖 Qwen3.5 + 两个 prose corpus。

## 4. 复现命令

```bash
# 9B target caches / generated OOD
python -m infra.cache_target --target qwen3.5-9b \
  --data data/processed/math500_target_qwen3.5-9b.jsonl --topk 256 \
  --out out/target_cache/math500_target9b_k256.npz

python -m infra.generate_target --target qwen3.5-9b --domain wikitext \
  --generate-wikitext --data data/processed/wikitext_gen_target.jsonl \
  --max-new 512 --out data/processed/wikitext_gen_target_qwen3.5-9b.jsonl

# activation importance + width sweep
python -m infra.activation_importance --model qwen3.5-9b \
  --data data/processed/distill_controlled.jsonl --max-tokens 200000 \
  --out out/importance/qwen3.5-9b_act.pt

python -m infra.joint_scan --model qwen3.5-9b --mode width \
  --importance out/importance/qwen3.5-9b_act.pt \
  --fracs 0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.85,0.9,0.95,0.98,0.99,1.0 \
  --sets math9b:data/processed/math500_target_qwen3.5-9b.jsonl:out/target_cache/math500_target9b_k256.npz \
         wikitext:data/processed/wikitext_target.jsonl:out/target_cache/wikitext_target9b_k256.npz \
         fineweb:data/processed/fineweb_target.jsonl:out/target_cache/fineweb_target9b_k256.npz \
  --out out/rank_scan/qwen3.5-9b_actwidth

# certificate / frontier / latency / tables
python -m infra.compression_certificate --scan out/rank_scan/qwen3.5-9b_actwidth \
  --gamma 4 --mlp-share 0.6088 --out out/rank_scan/certificate_actwidth_9b.json
python -m infra.cost_frontier \
  --scan 'out/rank_scan/qwen3.5-9b_actwidth/*_frac*.json' \
  --model qwen3.5-9b --mode width --gammas 4,8 \
  --out out/rank_scan/frontier_actwidth_9b.json
python -m infra.latency_pair --model qwen3.5-9b --mode width \
  --importance out/importance/qwen3.5-9b_act.pt --fracs 0.3,0.5,0.7,0.9 \
  --len 512 --repeats 20 --out out/latency/pair_width_9b_pad.json
PYTHONPATH=. python paper/make_v2_assets.py
```

## 5. 与第二轮审稿意见的对应

| 审稿意见 | 处理 |
|---|---|
| 理论只是标准事实、Assumption 2 重述 | 换成 channel-deletion 精确扰动 + top-2 certificate + cost no-go；不再使用 TV power law 假设 |
| Markov 界 vacuous、无 certificate | `q_lb` 在 raw OOD 全宽度 > q*，非空数值证书 |
| 容量与 token 混杂 | 新主实验是同一 target 的自压缩，不涉及预训练 token 混杂；旧扫描降为 secondary |
| i.i.d. Hoeffding | 全部改为 sequence-cluster bootstrap |
| “no improvement” 没有等价检验 | 9B-target 27B vs 4B 加 paired bootstrap + TOST（margin 0.01，两者均 significant） |
| raw vs generated OOD 混用 | 明确分表，不 pool |
| q_greedy vs q_s 混用 | 定义分开，q_s 只在 Appendix E |
| 27B cost / 部署误导 | 新主结论是 compressed draft 的 cost frontier；27B 明确是 model-effect probe |
| real-loop prompt 少 | 新主实验 500 MATH + 100×4 OOD；旧 loop 保留为 pilot 并明确标注 |
| 排版/表格/编号 | 新论文重写，表格/公式/标题重排；旧 v1 保留 |

## 6. Claim 边界

- 结论限定：post-hoc structured pruning + 单层 reconstruction、dense target、batch-1、Qwen3.5 9B/4B、
  measured evaluation distributions。
- 不声称：fine-tuned compressed draft 不行；所有压缩方法不行；production engine 的 wall-clock；
  universal floor。
- 成本模型不含 continuous batching / tree verification；FLOP 曲线是 optimistic bound，
  实测 latency 是 conservative implementation。
- 旧容量扫描的结论仍是条件性的：off-the-shelf same-family 0.8B–27B、greedy/top-1 和
  token-level standard SD；不声称 universal floor 或 wall-clock speedup。

## 7. 2026-11-06 第二轮大修补充

### 7.1 结论推广到 compute-reducing post-hoc 算子族

- 证书与成本测试是 operator-agnostic 的：任意压缩算子 C 都可测 `q_lb(C)`、`q_s(C)`，套用同一
  `C_FLOP` 下界。
- 在 matched nominal FLOP 下做了 operator zoo（Qwen3.5-9B）：activation-aware width deletion
  的 rejection 低于 weight-norm、independent SVD、shared rotation；weight-only W8/W4 是 memory-only
  轴（不减少 FLOP），单独讨论。第五轮用真实 packed INT4/INT8 kernel 实测：W4 `q=0.039–0.156`、
  `rho_mem=0.35–0.43`，通过 memory roofline（`q*_mem=0.31–0.38`），但实测 `rho_1≈0.99–1.01`、
  cost 5.24–5.34；W8 近无损但 INT8 kernel 更慢（`rho_1=2.11`）。这是 kernel-dependent cost-only
  结果，不是 structural no-go。文献中 QSpec/QuantSpec/ML-SpecQD/Quasar 已经证明量化+SD 是成功的
  正交方向（1.2–2.7×，80–98% acceptance）；我们的 W4 top-1 acceptance 86–96% 与之一致。见
  the private lab experiment log。
- 相关文献：SD²（self-distilled sparse drafters）、Quasar（quantized self-speculative）、
  SpecKV（compression-aware gamma）、Conformal SQS（TV decomposition）。

### 7.2 Batch-size scan

`out/latency/batch_scan_9b.json` / `batch_scan_9b_w098.json`：B=1,4,8,16，width 0.5/0.98，
padded narrow MLP。实测 `rho_B≈0.99–1.00`、`tau_B≈1.27–1.31`，`gamma*rho_B+tau_B>5` 在所有
measured batch 上都成立；与 FLOP 下界一致。

### 7.3 K2-Horizon-3.7B 跨家族全宽度

K2-Horizon-3.7B 是标准 dense Transformer（36 层，d=2560，h=10240），target 和 compressed draft
同一模型；30 MATH + 50 WikiText。结果在 Appendix K（`tab_k2.tex`）：WikiText 5–99% 全部
`q_lb > q*`，99% width `q=0.305, q_lb=0.194, q*=0.003`；MATH ≥70% certified。
这也回应“hybrid linear attention / Qwen-specific”的质疑。

### 7.4 校准集重叠检查

`distill_controlled.jsonl` 的 745 个 unique id 包含全部 500 个 MATH eval id；WikiText/FineWeb
无重叠。用去掉 MATH 的 calibration 重跑 0.5/0.9/0.98：MATH q 变化 ≤0.0006，raw OOD 变化
≤0.0015，结论不变（Appendix `app:nomath`）。

### 7.5 第三轮补充（2026-11-06）

- 标题改为 “Structured FFN Width Reduction as a Speculative Draft Fails the Optimistic FLOP Break-even on Three Raw-Text Corpora”，
  避免在无公平端到端 serving benchmark 时宣称 wall-clock unprofitable。
- Qwen3-4B / Llama-3.1-8B 补 30%/50%/70%/90% 宽度；raw OOD 全部 certified，MATH 50–70% 仍有
  FLOP 窗口（与 9B 一致），但没有这两个模型的 matched batch-1 cost 测量。
- K2-Horizon-3.7B 全宽度（标准 dense Transformer）加入 Appendix K；Wiki 5–99% 全部 certified。
- `phi_MLP` ±10%/±20% 敏感性表（Appendix C）：50%/98% robust；10% 在 +20% 时失去
  lower-bound certificate，但 point q 仍高于 q*。
- top-1024 对 `q_lb` 的影响：certificate 只用 target top-2 与 draft 在该两 token 的 logits，
  top-256 已包含，因此 top-1024 不改变 `q_lb`；只影响 q/TV 的第三 token 翻转与 residual mass。
- sampling 的几何接受长度：98% 宽度 loop tau=4.32/4.19，用 q_s 的几何 A=4.26/4.30，误差约 3%。
- MATH-calibration 重叠控制：MATH q 变化 ≤0.0006，raw OOD ≤0.0015。
- 附录执行命令已删除（私有路径太多）；保留确定性、bootstrap、模型版本说明。

## 8. 第三轮大修补充（2026-11-06）

### 8.1 理论推广：support-selection principle + MoE

- 新增 Proposition（support selection）：post-hoc 精确压缩只能删除“恒为零”的 compute unit；
  FFN channel、MoE expert、attention head、depth 都是同一论证的实例。正式结论仍由
  Lemma 1（FFN width）和 Lemma 3（MoE）承担，Proposition 只作为统一框架。
- 新增 Lemma 3（MoE perturbation）：top-k MoE 的 active-expert reduction 有精确的
  deletion + renormalization 分解；expert deletion at fixed k 是 memory-only，不是
  compute-reducing。MoE 的 nominal FLOP ratio 为 `1 - phi_E (1 - k'/k)`。
- Lemma 1 的定位改为“对所有 block weights 的 exact-equality no-go”，并明确它不证明固定权重下
  不存在近似非坐标压缩；reconstruction 后的 certificate 仍由测量给出，不由 Lemma 2 解释。

### 8.2 MoE 实验（Qwen3.5-35B-A3B，256 experts，top-8）

目标 cache 从当前 35B 模型重建（`*_target35b_v2_k256.npz`），full model q=0。结果
（`out/rank_scan/qwen3.5-35b-a3b_moe/`）：

| k | rho_FLOP | q* | MATH q | Wiki q | FineWeb q | q_lb Wiki | q_lb FineWeb | certified |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 1.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -- |
| 6 | 0.915 | 0.0354 | 0.0178 | 0.0815 | 0.0741 | 0.0581 | 0.0496 | yes |
| 4 | 0.829 | 0.0735 | 0.0385 | 0.1762 | 0.1606 | 0.1209 | 0.1106 | yes |
| 3 | 0.786 | 0.0938 | 0.0598 | 0.2509 | 0.2283 | 0.1621 | 0.1478 | yes |
| 2 | 0.744 | 0.1150 | 0.1184 | 0.3826 | 0.3406 | 0.2231 | 0.2026 | yes |
| 1 | 0.701 | 0.1371 | 0.4956 | 0.7924 | 0.7382 | 0.3629 | 0.3461 | yes |

Expert deletion at fixed k=8 是 memory-only：keep 50% 时 Wiki q=0.354、FineWeb q=0.343，
keep 25% 时 0.512/0.519；rho_mem 分别为 0.535/0.303，但 per-token FLOPs 不变。

### 8.3 Depth reduction（Qwen3.5-9B）

| drop | rho_FLOP | q* | MATH q | Wiki q | FineWeb q | q_lb Wiki | certified |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.973 | 0.0110 | 0.0852 | 0.3237 | 0.2940 | 0.2139 | yes |
| 2 | 0.946 | 0.0223 | 0.3784 | 0.4306 | 0.4228 | 0.2595 | yes |
| 4 | 0.891 | 0.0456 | 0.6228 | 0.6173 | 0.6059 | 0.3196 | yes |
| 8 | 0.782 | 0.0960 | 0.8320 | 0.8436 | 0.8011 | 0.3756 | yes |
| 16 | 0.564 | 0.2163 | 0.8913 | 0.9613 | 0.9326 | 0.4220 | yes |

### 8.4 第二/第三模型 full width sweep

- Qwen3-4B 和 Llama-3.1-8B 补齐 5,10,20,80,85,100% 宽度；raw text 5–99% 全部 certified。
  产物：`out/rank_scan/qwen3-4b_actwidth_full/`、`llama31-8b_actwidth_full/`。
- Qwen3.5-27B full sweep：目标 cache 从 27B 重建（`*_target27b_k256.npz`），13 个宽度全部完成；
  0.85–1.0 在 3 卡上运行（单卡高宽度 OOM）。raw text 5–99% 全部 certified；99% width
  Wiki q=0.057、q_lb=0.034、q*=0.0027。产物：`out/rank_scan/qwen3.5-27b_actwidth/`。

### 8.5 校准重叠、FWER、OOD 诊断、matched cost、bytes frontier

- 校准去 MATH 控制已在 Appendix `app:nomath`；MATH q 变化 ≤0.0006，raw text ≤0.0015，
  30–50% 窗口仍在。
- FWER：9B 预指定 39 个 test（13 width × 3 raw corpus: Wiki/FineWeb/C4），Bonferroni 99.87%
  cluster CI（α=0.05/39）下 36 个 compressed raw-text cell 全部 certified；3 个 100% width
  cell 是 identity control（q=0，不计入 no-go test）。单独 MATH family 13 tests 中 5/12 为
  point-above-but-adjusted-LB-below；全模型 26/218 cell 为 bad，全部是 in-domain MATH
  （`out/paper_stats/confirm_family.json`、`out/paper_stats/certificate_audit.json`）。
- 第三个 raw corpus C4（100 chunks）在 0.98/0.99 width 上 q=0.105/0.072，全部 certified；
  产物 `out/rank_scan/qwen3.5-9b_actwidth_c4/`、`certificate_c4_9b.json`。
- OOD 诊断（`out/paper_stats/ood_stats_9b.json`）：raw Wiki/FineWeb target NLL 2.16/2.10
  （ppl 8.64/8.19），target-generated 0.187/0.208（ppl 1.21/1.23）；raw text 与 calibration
  8-gram overlap 0%，MATH 21%（因为 calibration 含 MATH eval prompts）。这不是 contamination
  certificate，只是 operational check。
- matched cost：pretrained Qwen3.5-4B 在 FineWeb raw 上 q=0.178（95% CI [0.173,0.184]），
  低于 q*=0.238；Wiki q=0.224（CI [0.218,0.229]）也低于 q*。因此 4B 在两个 raw corpus
  上都低于 frontier，而 compressed 9B @20% 为 0.703/0.691，远高于 frontier。
- 与 T2D (arXiv 2610.34103) / SD² (arXiv 2504.08838) 的关系：二者都是 prune target +
  distill/adapt on target responses，在 in-domain benchmark 上正结果；本文是 no-training
  post-hoc + raw-text frontier，结论正交。Sharma (arXiv 2606.30265) 的 exact KL certificate
  是更一般的理论版本，本文 top-2 bound 是其可测特例。
- roofline frontier：9B 的 FFN weight share 0.540 < FLOP share 0.609，所以 `rho_mem` 比
  `rho_FLOP` 更严格，`rho_roof = min(rho_FLOP, rho_mem) = rho_FLOP`（例如 50% width: rho_mem=0.730, q*_mem=0.122 vs q*_FLOP=0.140）。
  memory-bound 不会 reopen window；fused kernel 最多逼近 FLOP frontier，不能低于它。

### 8.6 复现命令

```bash
# MoE scan (4 GPUs max; 3 GPUs used here)
CUDA_VISIBLE_DEVICES=1,2,3 python -m infra.moe_scan --model qwen3.5-35b-a3b \
  --sets math:data/processed/math500_target.jsonl:out/target_cache/math500_target35b_v2_k256.npz \
         wikitext:data/processed/wikitext_target.jsonl:out/target_cache/wikitext_target35b_v2_k256.npz \
         fineweb:data/processed/fineweb_target.jsonl:out/target_cache/fineweb_target35b_v2_k256.npz \
  --topks 8,6,4,3,2,1 --prune-keeps 1.0,0.5,0.25 \
  --out out/rank_scan/qwen3.5-35b-a3b_moe

# depth scan
python -m infra.layer_scan --model qwen3.5-9b \
  --sets math9b:data/processed/math500_target_qwen3.5-9b.jsonl:out/target_cache/math500_target9b_k256.npz \
         wikitext:data/processed/wikitext_target.jsonl:out/target_cache/wikitext_target9b_k256.npz \
         fineweb:data/processed/fineweb_target.jsonl:out/target_cache/fineweb_target9b_k256.npz \
  --drops 1,2,4,8,16 --out out/rank_scan/qwen3.5-9b_layerdrop

# OOD diagnostics
python -m infra.ood_stats --model qwen3.5-9b \
  --sets math:data/processed/math500_target_qwen3.5-9b.jsonl \
         wikitext:data/processed/wikitext_target.jsonl \
         fineweb:data/processed/fineweb_target.jsonl \
         wikitext_gen:data/processed/wikitext_gen_target_qwen3.5-9b.jsonl \
         fineweb_gen:data/processed/fineweb_gen_target_qwen3.5-9b.jsonl \
  --out out/paper_stats/ood_stats_9b.json

# certificate FWER audit
python -m infra.certificate_audit --cert out/rank_scan/certificate_actwidth_9b.json \
  --scan out/rank_scan/qwen3.5-9b_actwidth --tests 26 \
  --out out/paper_stats/certificate_audit.json

# tables
python paper/make_revision2_assets.py
```

## 9. Fourth-round review closure（2026-11-07）

See the private lab experiment log for the full experiment log.  Highlights:

- Fixed-weight exactness lemma (`lem:fixed`), nested top-$k$ certificate
  (`lem:topk`), a draft-agnostic certificate, and a deployment-distribution /
  coverage theory (shift, mixture, Markov coverage, and an uncovered-region
  lower bound for data-only learners).
- Raw-text breadth: Wikipedia zh/de/fr/ja, FineWeb-2 zh, CodeParrot code,
  PG-19 books, arXiv; all certified at 99% width (one 50%-width code cell is
  point-above only).
- Physical attention-head pruning on Llama-3.1-8B and combined FFN+head /
  FFN+depth scans: all certified no-go.
- Margin-aware top-1 calibration at 98% width: gentle run 0.097→0.137
  WikiText, aggressive run 0.097→0.839; the failure is not MSE-only.
- Zero-MLP kernel ceiling: at 98% width even a free MLP leaves
  $\gamma\rho+\tau=5.10>5$.  Fair KV-cache serving loops on Llama-3.1-8B,
  Qwen3-4B, and K2-3.7B give 0.32–0.38× at 98% width; Llama at 50% width is
  0.19×.  Measured batch-1 costs for depth/head/combined are also cost-only
  no-go; MoE is nominal-only.
- Powered K2 100/100 replication: all widths 50–99% certified.

## 10. Sixth-round review closure（2026-11-07）

See the private lab experiment log for the full log. Highlights:

- Refitted-readout decomposition (`lem:refit`): the deployed ridge draft satisfies
  `Delta y = W_D a_D + (W_K - R) a_K`, which becomes
  `W_D [a_D - Sigma_DK (Sigma_K+lambda I)^-1 a_K]` for residual ridge.  The
  certificate is still measured on the composed map; the theory now closes the
  pruning-to-deployment gap.
- Leave-one-corpus-out margin-mass validation: at 98% width MAE 0.0138, all 11
  raw corpora predicted above `q*`; at 50% width MAE 0.0466, all 11 (including
  code) above `q*=0.140`.  MT-Bench (five widths) and SWE-bench are held out and
  also predicted above `q*`; the only predicted-break-even cell is in-domain MATH
  at 50%, where the measurement agrees.
- Combined width+INT4: FLOP-axis no-go survives at every raw-text width
  (50%: q=0.526 vs q*=0.140; 98%: 0.171 vs 0.0025).  The memory axis is not a
  certified failure (50% point above, lower bound below; 90/98% pass), so the
  paper treats it as an operator-selection boundary, not a quantization no-go.
- Cross-family confirmatory family: 36 9B English raw + 6 Llama + 3 K2 = 45
  tests, Bonferroni 99.889%, 45/45 certified.
- Required-vs-measured acceptance: at 99% width `alpha*=0.998` vs measured
  0.934; at 50% width 0.860 vs 0.497 (Appendix A.5 figure).
