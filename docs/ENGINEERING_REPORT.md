# Dense 草稿模型 for Qwen3.6-35B-A3B：接受率 / 平均接受长度（MATH-500）

日期：2026-10-31（初版）｜ 2026-11-01（§12 补充）｜ 机器：8×A100-80G

## 一句话结论

把 MoE 的 top-8 专家 + shared expert **无损拼宽**成一个同架构、同激活参数（3.43B）的 dense FFN 后，
只用 teacher 自生成数据做 CE 蒸馏（~1M token），dense draft 在 MATH-500（greedy、非思考模式）上就达到：

| γ（每步草稿 token 数） | 接受率 α | 平均接受长度 τ（tokens/step） |
|---|---|---|
| 3 | 70.7% | 3.12 |
| 5 | **60.1%** | **4.00** |
| 8 | 48.6% | **4.87** |
| 12 | 38.2% | 5.54 |

即：γ=5 时每次验证平均产出 4.0 个 token（3 个草稿被接受 + 1 个 teacher bonus），γ=8 时 4.9 个。
token 级 top-1 一致率 85.5%。**纯蒸馏已经足够好，不需要 RL 就有很强的接受长度。**

再用“平均接受长度期望”做 reward 做 100 步 RL（§4），还能再涨 4%~8%：
γ=5 τ=4.16 / γ=8 τ=5.15 / γ=12 τ=5.96，token 一致率 87.0%。

**但端到端速度是另一回事**（§6）：同激活参数的 dense draft 每 token 成本是 teacher 的 32%~40%，
本机实测在**任何并发（B=1~48）下都打不过模型自带的原生 MTP/EAGLE head**（1.0× vs 1.2~1.3×）；
只有在“draft 单 token 成本 ≤ teacher 的 10%~17%”的带宽受限场景、低并发下才有优势，约 **160~270 token 并发**处被反超。

**预测失败也不是“路由到冷门专家”**（§7）：路由越偏离拼宽的 top-8，token 反而越容易猜对（AUC 0.46）；
真正的信号是置信度（draft 自身熵 AUC 0.92）。动态预算只调深度只能省 ~6%~10% 算力，
要涨接受长度必须做**多候选树**（teacher 真 token 有 97.9% 落在 draft top-4 里，理想 τ(γ=8) 可到 8.3）。

**2026-11-01 补充（§12）：两个更极端的 draft 都验证过了。**
- 把 dense draft 做 **W4（group=128）weight-only QAT** 后，打包体积 6.87GB→**2.49GB**；
  纯 PTQ 的 token top-1 只掉 0.35pt（85.50%→85.15%），QAT 再训 1.17M token 后到 **86.63%**、
  τ(γ=32)=**7.08**。同预算的不量化对照是 86.76%/7.14（**-0.13pt / -0.8%**）——
  **W4 QAT 对接受率基本无损。**
- 把 teacher 的 256 experts 压到 **8 个**（top-8 全激活，激活参数不变），总参数 3.44B≈dense；
  1.17M token 蒸馏后 token top-1 **88.32%**、τ(γ=8)=**5.43**、τ(γ=32)=**7.96**。
  同预算的 dense 对照只有 85.65%/4.91/6.65（**+2.67pt，τ(γ=32) +19.7%**），
  比 dense-RL 也高 5.4%/9.9%，且 **γ 越长优势越大**。E=16（4.44B）反而没有更好
  （88.08%/7.83）——**8 个专家已经够，继续加总专家数不涨点。**
- **对原模型本身做 W4 QAT（§13）**：34.66B teacher 量化到 4-bit（packed 18.6GB）后当 draft，
  全 500 题 token top-1 **96.06%**、τ(γ=32)=**17.53**、`P(run≥32)`=**39.0%**；
  再做 100 step 部分 QAT（训 1.94B always-active 参数）后 **96.97%**、τ(32)=**19.95**、
  `P(run≥32)`=**44.0%**。这是容量维度的上界实验：γ=32 理想 τ=33，W4 QAT 到 ~60%；
  但 packed 18.6GB、`c_bw≈0.27`，带宽受限下最优加速 ~2.5×，并不比 dense MoE8 的 ~3.0× 更快。
- **深度律（§14）**：短 γ 的线性斜率精确等于 \(-p(1-p)/2\)（六个模型误差 <0.002），
  但全局是凹的（收益递减）；per-token 条件接受率随深度**上升**（selection），
  长 runs 是 stretched-exponential 重尾（β=0.72~0.91）。MTP 的 p 高、β 接近 1、
  block-start bias 小，所以 α(γ) 更接近直线、长 γ 更强。
- **动态深度（§15）**：按 confidence 聚类动态选 γ，弱 draft 收益更大
  （MoE8/dense +5~6%），teacher QAT 只有 +1.5%（confidence 双峰、β=0.91，
  fixed γ 已接近最优）。大 draft vs MTP：**τ 优势随 γ 增大**（γ=32 高 20%），
  等 c 下 teacher QAT 赢；但 W4 的 c=0.268 vs MTP 0.03，绝对 speedup 仍是
  MTP 8.5× vs 2.5×；W3 PTQ 崩到 90.2%，不能便宜地替代。
- **WikiText 分布偏移（§16）**：小 draft 在 WikiText-2 上 NLL 4.7~5.9、TV 0.65~0.68、
  next-tok top-1 23.8%~27.1%、**teacher-agree 30%~35%**；teacher QAT 是 2.10 / 0.143 / 54.9% / **83.3%**，
  MTP 是 2.74 / 0.381 / 43.3% / **62.1%**。跨模型 MATH 接受率与 TV/KL/NLL 强相关
  （r≈-0.96~-0.99），模型级
  \(p\approx1.014-0.212\text{TV}\approx1.008-0.076\sqrt{KL}\approx1.066-0.119\log\text{NLL}\)。
  **小 draft 的高 MATH 接受率是域内特化；大 draft 的长 γ 优势来自全局分布更接近 teacher。**
- **OOD 灾难（§17）**：long-γ 接受长度满足 \(\tau_\infty\approx1/q\)，
  \(q=1-\text{teacher-agree}\) 是 OOD/不一致率；7 个模型拟合 **R²=1.000000**。
  WikiText：小 draft q=0.65~0.70 → τ∞≈1.5；MTP q=0.38 → 2.6；
  teacher QAT q=0.17 → 6.0。合成注入 **5% OOD 就把 τ(32) 砍 25%~41%，
  10% 砍 41%~60%**。要 τ∞>20 需 q<5%；35B QAT 在 OOD 文本仍 q=0.17，
  scaling 只到 τ∞≈6——**naive“超大 draft”路线在 OOD 上被 residual q 卡死**。
  域内小 draft q≈12~15% → τ∞≈7~8，teacher QAT q≈3% → 33：大 draft 域内仍有价值，但不能解决 OOD。

## 1. 模型与“同激活参数”的构造

Teacher：`/path/to/models/Qwen3.6-35B-A3B`（34.66B 总参数，每 token 激活 8/256 experts + shared）。

激活参数核算（每层 FFN）：`8×3×2048×512 + 3×2048×512 = 28.3M`，40 层 = 1.13B；
加上 attention/embedding，teacher 单 token 激活 ≈ **3.46B**。

Dense draft（`code/draft.py`）：
- 完全复用 teacher 的 backbone（40 层 hybrid linear/full attention、hidden 2048、vocab 248K、embedding、lm_head、norm）。
- 只把 `mlp`（MoE block）换成 **dense FFN，intermediate = 8×512 + 512 = 4608**，恰好等于 teacher 单 token 激活的 FFN 宽度。
- 总参数 3.43B ≈ teacher 的单 token 激活参数，**per-token FLOPs 与 teacher 完全对齐**。

### 三个初始化/训练阶段
1. **Train-free 专家拼接（`code/make_dense.py`）**：在 160 条 calibration 文本上统计每层 router 使用频率，取 top-8 专家，
   把 8 个专家的 `gate/up` 行拼接、`down` 列按平均路由权重拼接，再加 shared expert（按平均 sigmoid gate 缩放）。
   由于 `sum_k w_k·E_k(x)` 在 w 固定时可被“拼宽”精确表示，这一步不是权重平均，而是**函数级的等价改写**。
2. **逐层 FFN 拟合（`code/fit_ffn.py`）**：用 10 万 token 的 teacher 中间激活，对每层 dense FFN 做 MSE 回归（~2 分钟/40 层），
   修正“路由权重随 token 变化”带来的误差。FFN 输出相对误差 0.65→0.25。
3. **在线蒸馏（`code/train_draft.py`）**：在 teacher 自己 greedy 生成的数学解答上做 hard-label CE（全参数微调，3 epoch，~2.9M token）。

| 阶段 | 训练量 | token top-1 一致率 | τ(γ=5) | τ(γ=8) |
|---|---|---|---|---|
| 专家拼接（纯 train-free） | 0 | 65.6% | 2.50 | 2.69 |
| + 逐层 FFN 拟合 | 100k tok MSE | 75.1% | 3.08 | 3.46 |
| + 蒸馏 | 2.9M tok CE | **85.5%** | **4.00** | **4.87** |
| + RL（接受长度期望） | 1.5M tok | **87.0%** | **4.16** | **5.15** |

## 2. 数据

- Prompt 池：`AI-MO/NuminaMath-CoT`（modelscope），取 40k 条后去重：精确重复 0.7%，与 MATH-500 重叠 0 条，
  13-gram 近重复率 1/5000。最终 39,716 条（`code/prep_data2.py` 有完整扫描输出）。
- 蒸馏数据：用 teacher 对 3,408 条 prompt 做**非思考模式 greedy 生成**（max 768 tok），共 ~2.1M output token。
  用 teacher 自己的输出而不是数据集里的解答，是为了让 draft 学到“teacher 会给什么 token”，这是接受率的直接目标。
- Eval：`HuggingFaceH4/MATH-500` 全部 500 题，非思考模式，teacher greedy，max 640 token（约 10% 被截断，见局限）。

## 3. 评测方法（关键）

Greedy 投机解码下，被接受的 token 一定等于 teacher 的 greedy token，因此 draft 的上下文永远是 **teacher 的真实前缀**。
于是“接受率”可以精确地一次前向算出来：把 teacher 生成的整条序列喂给 draft，逐位置比较 `argmax` 与 teacher token，
再按 γ 模拟“连续匹配 → 接受、首个不匹配 → 停止 + teacher 补 1 个 token”的过程。

- 这一等价性用**真实的投机解码循环**（`code/sd_sim.py`，teacher 每步重算前缀、draft 用 KV cache）验证过：
  实测 α=0.643 / τ=4.22，与 teacher-forcing 估计（α=0.60 / τ=3.99）一致。
- α = 接受的草稿 token / 提出的草稿 token；τ = 平均每步产出的 token 数（含 bonus token）。

## 4. RL：用“平均接受长度期望”做 reward

不需要采样，期望接受长度本身可微：
`E[τ] = Σ_blocks (1 + Σ_{k=1..γ} Π_{i≤k} p_draft(teacher_token_i | prefix))`，
其中 `p_draft` 是 draft 在 teacher 前缀下对 teacher token 的概率。用 `ce_w=0.5, rl_w=1.0, γ=5` 从蒸馏模型继续训 100 步。

结果（MATH-500 全量 500 题）：

| | 蒸馏 | + RL | 相对提升 |
|---|---|---|---|
| token top-1 | 85.5% | 87.0% | +1.5pt |
| τ(γ=3) | 3.116 | 3.200 | +2.7% |
| τ(γ=5) | 3.995 | 4.162 | +4.2% |
| τ(γ=8) | 4.866 | 5.154 | +5.9% |
| τ(γ=12) | 5.543 | 5.961 | +7.5% |

结论：**RL 有稳定但不算大的增益，且 γ 越大增益越大**（跟“奖励连续接受长度”的设计一致）。
纯蒸馏已经拿到大部分收益；RL 只在边界情况下（长草稿块）继续推一点。
原因是 CE 蒸馏已经把“逐位置匹配”压到接近这套架构的上限，真正的瓶颈是 dense 3.4B 对 3.4B-active MoE 的
**路由方差**（同一个 token 在不同 expert 组合下的输出，dense 只能学期望），不是训练目标。

## 5. 工程含义（重要）

接受长度高 ≠ 端到端一定快。speedup ≈ τ / (1 + γ·c)，c = draft 单 token 成本 / teacher 单 token 成本。

- 本机未优化的 HF/PyTorch 实现里，batch=1 实测 draft 18.2 tok/s、teacher 13.2 tok/s（c≈0.72），
  此时 γ=5 反而**没有加速**（0.8×）——因为两者都被同一套 Python/linear-attention kernel 开销支配。
- 在正常 serving（vLLM/SGLang，decode 是 memory-bound）下，teacher 要读 70GB 权重、dense draft 只读 7GB，
  c≈0.1，则 γ=5 的 wall-clock speedup ≈ 4/(1+0.5) ≈ **2.7×**；γ=8 ≈ 4.9/(1+0.8) ≈ **2.7×**。
- 换句话说：**同激活参数的 dense draft 在“接受长度”上很划算，但在“算力”上没有便宜可占**，
  它的优势来自显存/带宽（35B 总参数 vs 3.4B 总参数），不是 FLOPs。

## 6. 并发对比：多少并发会被原生 EAGLE 打败

> **更正（见 §9.2）**：下面 MTP 的接受率是“每个 block 重置 KV”的保守实现。用训练一致/真实实现里的
> **持久 KV cache**，MTP 会强很多（token 准确率 95.7%，τ(5)=5.32、τ(8)=7.44），
> 因此 §6 的结论（EAGLE/MTP 赢）只会更强，只是“它比我们准这么多”这一点当时被低估了。

先说明“原生 EAGLE”用的是哪个：这个 checkpoint 自带一个 **MTP head**（`mtp.fc` + 1 层 full-attention decoder + norm），
结构上就是 EAGLE 式的“特征融合 + 单层 draft”，而且是和 target 联合训练的。我把它按 EAGLE 的方式递归使用
（`code/mtp_eval.py`，KV 每个 block 重置，偏保守）。它的接受率：

| γ | 原生 MTP/EAGLE α | τ | 我们的 dense draft α | τ |
|---|---|---|---|---|
| 1 | **89.5%** | 1.90 | 85.8% | 1.86 |
| 3 | 66.3% | 2.99 | 73.5% | 3.20 |
| 5 | 45.2% | 3.26 | **63.4%** | **4.16** |
| 8 | 28.9% | 3.31 | **52.2%** | **5.15** |

MTP 第一跳更准（89.5% vs 85.8%），但递归 2 跳之后掉得很快（τ 在 γ=8 只有 3.31）；我们 τ 高 27%~56%。

### 实测（HF/PyTorch，batch=并发，序列 384 token，γ=5）

| 并发 B | 无投机 tok/s | 我们 tok/s | 加速 | MTP tok/s | 加速 | c=draft/teacher 单步 | 验证/单步 |
|---|---|---|---|---|---|---|---|
| 1 | 7.4 | 7.0 | 0.95× | 9.5 | **1.28×** | 0.40 | 2.37 |
| 8 | 52.8 | 52.3 | 0.99× | 69.1 | **1.31×** | 0.37 | 2.34 |
| 16 | 100.9 | 102.2 | 1.01× | 132.9 | **1.32×** | 0.36 | 2.33 |
| 32 | 188.1 | 191.7 | 1.02× | 239.7 | **1.27×** | 0.33 | 2.43 |
| 48 | 272.0 | 273.9 | 1.01× | 331.2 | **1.22×** | 0.32 | 2.55 |

**结论：在这套实现里，B=1 就已经被打败了，没有交叉点。** 我们的 dense draft 在任何并发下都只有 ~1.0×，
原生 MTP/EAGLE 稳定 1.2~1.3×。γ=8 也一样（0.92×~1.01× vs 1.20×~1.29×）。

### 为什么，以及什么条件下才能赢

单步成本 = `(γ·c + v) / τ`，c = draft 单 token 成本 / teacher 单 token 成本，v = 一次验证的成本（teacher 单步的倍数）。
我们的 τ 高，但 draft 每 token 要花掉 teacher 单步的 **32%~40%**（因为 FLOPs/token 和 teacher 一样，只是权重少 10 倍）。
盈亏平衡需要：

```
c* = (τ_ours·(γ·c_mtp + v)/τ_mtp − v)/γ
```

- v=1.2（优化过的 chunked prefill）：**c* ≈ 0.104**
- v=2.4（本机实测）：**c* ≈ 0.171**

实测 c≈0.32~0.40 → 输。只有当 draft 单 token 成本降到 teacher 的 **10%~17%** 以下，高 τ 才能补回来。

理想 serving（带宽受限、v≈1.2、c≈0.1）：两边基本打平（我们 ~2.4×，MTP ~2.4×），此时 **并发一上来我们就先输**：
dense draft 只有 7GB 权重，**~158 token/step 就进入 compute-bound**；teacher 有 70GB，要到 **~1580 token/step** 才 compute-bound。
中间这一段 c 从 0.1 涨到 1，我们在 **~160~270 token 并发（约 3~10 个请求）** 处被 MTP/EAGLE 反超；
超过 ~1580 token 后两者都变 compute-bound，投机解码整体不如不投机（MTP 输得少）。

一句话：**同激活参数的 dense draft 的收益全在“带宽/显存”，一旦 draft 不再是带宽瓶颈（并发 ≳ 几百 token）就必然被便宜的小 draft 打败；
而在本机未优化实现里，它连低并发都赢不了（c=0.4）。**

## 7. 失败原因是“路由到冷门专家”吗？——不是

在 80 条 MATH-500 序列（37,849 个生成 token）上，把 teacher 40 层的路由结果和 dense draft 的
逐 token 对错对齐（`code/route_analyze.py`，路由位置已按“位置 p 的路由影响 token p+1”对齐）：

| 特征 | 最低分位 mismatch | 最高分位 mismatch | AUC(mismatch) |
|---|---|---|---|
| **w_out**：路由权重落在“拼宽 top-8”之外的占比 | 9.5% | **4.6%** | 0.462（负相关） |
| **w_cold25**：路由权重落在最冷 25% 专家上 | 16.8% | **6.8%** | 0.447（负相关） |
| teacher 输出熵 | 0.45% | **53.3%** | **0.882** |
| draft 自身熵 | 0% | **56.9%** | **0.923** |
| draft 自身 top-1 概率 | 55.8% | 0% | 0.078（=0.922 反向） |

- **路由越“偏”（越常落到非 top-8 / 冷门专家），token 反而越容易被猜对。**
- 在 teacher 熵分桶内再算 `AUC(w_out)`：0.51 / 0.53 / 0.56 / 0.51 / 0.52 —— **路由几乎没有增量信息**。
- 另外：所有 token 都有 ≥54% 的路由权重落在 top-8 之外（top-8 只覆盖 15~20% 的权重），
  所以“draft 就是 top-8 专家”这个前提本身就不成立。

直觉解释：冷门/偏门路由往往出现在**高度模式化**的 token（数字、公式符号、格式符）上——某个专家很自信地接管；
而真正难猜的“分叉点”（“所以/因此/设 x=…”）反而是通用专家在兜底，路由很平。**路由偏差标记的是“专家自信”，不是“能力不足”。**

### 那动态预算能涨多少？

有信号，但是**置信度**而不是路由；而且要看预算花在“深度”还是“广度”上：

1. **只调深度（adaptive γ，置信度门控）**：τ 本身不变，省的是草稿算力。
   实测门控策略（p<0.7→γ=2, <0.9→γ=4, <0.98→γ=6, else γ=8）：
   - 本机（c=0.35, v=2.4）：tok/cost **+6.4%**；理想 serving（c=0.1, v=1.2）：**+10.1%**
   - 上帝视角最优 γ 的界：+35% / +40%
2. **调广度（tree，EAGLE-2 式）才是真正涨接受长度的路**：teacher 的真 token 落在 draft top-k 的比例：

| k | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| recall | 89.6% | 95.5% | **97.9%** | 99.0% | 99.5% |

   用 top-k 树、理想情况下 τ(γ=8)：k=1 → 6.0，k=2 → 7.5，**k=4 → 8.3**（实测单路径 5.15）。
   代价是树节点数（草稿算力）随 k 和 γ 指数增长。

**结论：接受率上不去不是冷门专家的问题，加预算也不会让“单路径”更准；想要更高接受长度必须上树（多候选），
而树的剪枝信号用 draft 自己的置信度就够（AUC 0.92），路由信号没有额外价值。**

## 8. 分叉点占被拒 token 的比例 & 能不能提前看出来

数据：120 条 MATH-500 序列、59,374 个生成 token（draft = distill+RL），draft token 准确率 89.5%。
“分叉点”用 **teacher 自己的 next-token 不确定度**定义（teacher top-1 概率 / 熵 / top1-top2 margin）。

### 8.1 被拒 token 的构成

| 分叉点定义 | 出现率 | P(被拒\|分叉) | P(被拒\|非分叉) | **占全部被拒** |
|---|---|---|---|---|
| teacher top1 < 0.50（严格） | 1.0% | 68.2% | 9.9% | 6.8% |
| teacher top1 < 0.90 | 9.5% | 53.5% | 6.0% | **48.5%** |
| teacher top1 < 0.99（宽） | 16.5% | 44.9% | 3.7% | 70.7% |
| teacher 熵 > q90 | 10.0% | 53.2% | 5.8% | 50.6% |
| teacher 熵 > q80 | 20.0% | 40.7% | 3.0% | 77.3% |
| top1−top2 margin < 0.10 | 1.0% | 59.0% | 10.0% | 5.6% |
| 双不确定（teacher<0.9 且 draft<0.9） | 6.0% | 62.1% | — | 35.5% |

**约一半（48.5%）的被拒 token 落在分叉点上**（这些位置只占 9.5%）；另一半落在 teacher 很自信的位置，
那里只有 6% 被拒——**这部分是纯粹的 draft 容量误差，可以靠更好的蒸馏/更大 draft 消掉**。

如果 draft 在“非分叉点”上全对（分叉点保持现状）：token 准确率 89.5% → **94.6%**，
对应 τ(γ=5) 4.16 → **5.24**、τ(γ=8) 5.15 → **7.28**。
换句话说：**剩下的一半 headroom 是 draft 能力，另一半是语言本身的歧义**。

### 8.2 分叉点能不能提前看出来？——弱可预测（AUC≈0.73）

分叉点有聚集性但很短：`P(branch[t] | branch[t-1]) = 0.229`（基准 0.095，**2.4× lift**），
run 长度 mean 1.30 / p90 2 / max 10 —— 大多数是孤立的。

只用 t 之前的信息预测 branch[t]（teacher top1<0.9）：

| 提前特征 | AUC |
|---|---|
| t-1 的 teacher 熵 | 0.711 |
| 最近 4 个 token 的 teacher 熵均值 | **0.714** |
| 最近 4 个 token 的 teacher 熵标准差 | 0.707 |
| t-1 的 teacher top1（取反） | 0.710 |
| t-1 的 draft 自身熵 | 0.687 |
| t-2 / t-3 的 teacher 熵 | 0.673 / 0.649 |
| t-1 是否被拒 | 0.560 |
| 上一 token 的字符类（换行/数字/运算符/latex） | 0.43~0.55（单独几乎没用） |
| **16 个 lag 特征做逻辑回归（按序列切分）** | **0.735**（只用 lag-1 也有 0.730） |

**可以，但只是“排序”级别的可预测**：AUC 0.73 意味着能给风险排序，不足以做精确触发。

真正有用的是少数**表层线索**（上一 token 的身份），它们把风险高度集中：

| 上一 token | 占比 | P(分叉) | P(被拒) |
|---|---|---|---|
| `" the"` | 2.9% | 24.7% | **26.4%** |
| `","` | 1.2% | 19.9% | **25.8%** |
| `" to"` | 0.6% | 24.0% | **23.5%** |
| `"\n\n"` | 2.4% | 27.7% | **22.8%** |
| `"$."` | 0.6% | 26.0% | 20.5% |
| `" is"` | 0.8% | 22.9% | 20.4% |
| `"."` | 1.2% | 17.2% | 18.8% |
| `":"` | 1.7% | 16.5% | 16.6% |
| 数字 / `" \\"` / `" ="`（公式内） | ~15% | 3% | **4~5%** |

这 12 个“句首/句读”类上一-token 覆盖了 **32.2% 的全部被拒 token**（它们只占约 12% 的位置），
而公式内部（数字、`\`、`=`）被拒率只有 4~5%。**分叉点主要发生在自然语言的句子/推理步骤边界，不在公式内部。**

另一个和任何 gating 相关的数：draft 自己 maxp>0.9 的位置占 83.3%，其中被拒只占全部被拒的 25.4%
（P=3.2%）；maxp>0.99 时只占 6.5%。**“自信但错”的 silent error 只占 12.3%**（teacher 和 draft 都自信时）。

## 9. 分叉点预测器 / 双 draft 复合 / temperature

### 9.1 训一个分叉点预测器

目标：`teacher top-1 < 0.9`（base rate 9.5%），59,374 个 token，按序列 8:2 切分。

| 可用特征 | 模型 | held-out AUC |
|---|---|---|
| 只用 t 之前的特征（lag-1..3 熵、rolling 熵、上一 token 类别、位置） | LogReg | 0.723 |
| 同上 | GBDT | **0.752** |
| draft 自己在 **t 位置的置信度**（草稿时就能算出来） | — | **0.883** |
| 两者都加 | GBDT | 0.885 |

- **提前（看到 token 之前）预测：弱**，AUC≈0.75，只能排风险。
- **在草稿时（同一位置、draft 已经算出分布）预测：强**，AUC 0.883：
  draft maxp>0.99 的位置只有 **1.7%** 是分叉点，maxp<0.5 时有 **48.3%**。
- 结论：分叉点预测器的上限就是“draft 自己的置信度”，额外再训模型只多 0.002 AUC，不值。

### 9.2 两个模型复合来解码（dense draft + 原生 MTP head）

先说一个重要更正：把 MTP 的 KV cache 改成**持久**（= 训练时的因果上下文，也是真实实现的做法），
MTP 从“便宜但一般”变成“又便宜又强”：

| 同一批 100 条序列 | token 准确率 | τ(γ=1) | τ(γ=3) | τ(γ=5) | τ(γ=8) |
|---|---|---|---|---|---|
| dense draft（3.43B） | 89.6% | 1.89 | 3.35 | 4.47 | 5.71 |
| 原生 MTP（持久 cache） | **95.7%** | 1.95 | 3.72 | **5.32** | **7.44** |
| 复合（学出来的 gate） | 95.4% | 1.90 | 3.43 | 4.64 | 7.27 |
| oracle 取二者之并 | 97.5% | 1.97 | 3.83 | 5.58 | 8.03 |

- 两个 draft 只有 10.7% 的位置不一致；在不一致处 **MTP 对 73.5%、dense 只对 16.7%**。
- 学出来的 gate（置信度 + 一致性 + lag 特征，LogReg/GBDT）AUC 0.995，但**实际收益 ≈ 0**：
  95.4% ≈ MTP 单用的 95.7%。并集 oracle 只有 +1.8pt（τ(8) 7.44→8.03，+8%），gate 一点都拿不到。
- **结论：复合不划算——因为一个 draft 严格支配另一个。** 想要复合有意义，需要两个“互补且水平相当”的 draft；
  这里 3.4B dense 在同样的 teacher 前缀下就是打不过原生 MTP（MTP 直接吃 teacher 的 hidden state，信息优势太大）。

### 9.3 提 temperature 能不能解决“自信但错”？

**greedy 下：完全没用。** `softmax(z/T)` 只改变分布形状不改变 argmax，实测 T=0.1/0.5/2/10 的 argmax 100% 相同。
“自信但错”是 top-1 错误，temperature 动不了它。

**sampling 投机解码下：反而变差**（teacher 固定 T=1，扫 draft 温度，指标 = 每 token 期望接受概率 `Σ min(p_t, q_d^T)`）：

| draft T | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 接受率 | 0.878 | 0.879 | 0.879 | 0.880 | **0.881** | 0.880 | 0.876 | 0.864 | 0.831 | 0.580 | 0.069 |

**降温确实有一点用，但只有 +0.5pt（T≈0.5），而且天花板就是 greedy top-1 一致率（0.883）**——
再往下压只会逼近 one-hot 的 `p_t(argmax)`，不会超过它。

两边一起升温也没用（T=1.5 时 0.837 < T=1 的 0.876）。

原因：这份 draft 其实**偏“欠自信”**——平均 maxp 0.937 vs teacher 0.968（正确位置 0.965、错误位置 0.694）。
升温只是把质量摊平，不会把质量搬到 teacher 的 token 上；小幅**降温**（T≈0.5~0.75）反而 +0.5pt。

那“自信但错”到底是什么？看这 1,584 个（占全部 2.7%、占被拒 25.4%）位置：
**51.3% 落在分叉点上**（base rate 9.5%），这些位置 teacher 自己也只有 0.82 的 top-1。
所以它不是“校准坏了”，而是 **draft 在 teacher 本来就不确定的地方“先下结论”**。
这类错误只能靠“多候选/换更强的 draft”，不能靠温度。


### 9.4 并集为什么更大 & 剩下的提升空间在哪

同 100 条序列（48,491 token）上的误差分解：

| 类别 | 占比 | 其中分叉点占比 | 该处 teacher top-1 |
|---|---|---|---|
| 两个都对 | 87.85% | 3.6% | 0.988 |
| 只有 MTP 对 | **7.89%** | 36.2% | 0.883 |
| 只有 dense 对 | **1.79%** | **71.6%** | 0.755 |
| 两个都错 | 2.48% | 87.5% | 0.670 |

- 分位置准确率：MTP 在**非分叉点 99.1%**、分叉点 63.7%；dense 在**非分叉点 94.1%**、分叉点 47.0%。
- **并集更大，是因为两者在“分叉点”上错的地方不一样**：dense 唯一赢的 1.79% 里 71.6% 是分叉点
  （teacher top-1 只有 0.755），这正是 MTP 也会错的地方。
- dense 的 10.4% 错误 = 2.5% 两个都错（不可救）+ 7.9% “MTP 对、dense 错”；后者里 64%（≈5.0pt）在
  **非分叉点**上——纯粹是训练/数据不够（train acc 98.4% vs val 89.6%，2.1M token 已过拟合）。

**两个“提升杠杆”的结论：**

1. **持久 cache 对 dense draft 不是杠杆**：dense 本来就用 full attention 看整个前缀（自己的 KV）。
   MTP 的 cache 之所以值 +6pt（89.5%→95.7%），是因为它缓存的是 **teacher 的 hidden state**——
   这是“信息优势”，不是“cache 技巧”。dense 想要这个优势，就必须变成“吃 teacher 特征的单层 head”，
   那就是 MTP 本身。
2. **继续蒸馏/RL 的天花板 ≈ 94.7%**：假设 dense 在非分叉点上做到 MTP 的水平（99.1%），
   整体上限 = 1 − P(错误且分叉) ≈ **94.7%**，仍然**低于 MTP 的 95.7%**（MTP 在分叉点上 63.7% vs 47.0%）。
   要再往上只能靠“多候选/树”或者换更强的 backbone，而不是继续堆同一份数据的 epoch。
3. 并集上限 97.5%（τ(8)=8.03）需要“两个互补模型 + 完美 gate”，实测 gate 拿不到任何收益（§9.2）。

## 10. 局限

- 只测了数学、非思考模式、greedy；思考模式、代码/闲聊、sampling 解码（temp>0）未测，跨域接受率会下降。
- eval 序列 max 640 token，约 10% 的解答在结尾被截断，因此没有覆盖“最终答案 \boxed{}”那一段。
- 蒸馏数据只有 2.1M token / 3408 条，训练 3 epoch 已明显过拟合（train acc 98% vs val 88%），加数据还有提升空间。
- teacher 生成是 batch=48 的 greedy，与 batch=1 greedy 可能有极少数数值差异。

## 11. 复现

```bash
python code/prep_data2.py                     # prompt 池 + 去重扫描
python code/teacher_run.py --task calib       # router 使用统计
python code/make_dense.py                     # 专家拼接（train-free dense）
python code/fit_ffn.py --tokens 100000        # 逐层 FFN 拟合
CUDA_VISIBLE_DEVICES=3 python code/teacher_run.py --task eval --max_new 640
CUDA_VISIBLE_DEVICES=4,5,6,7 python code/teacher_run.py --task gen --shard k --nshard 4 --limit 1600
torchrun --nproc_per_node=3 code/train_draft.py --ckpt ckpt/dense_ffnfit --epochs 3 --lr 1e-4
python code/eval_accept.py --ckpt ckpt/dense_distill_s1 --gammas 1,2,3,5,8,12
```

---

## 12. 更极端的两个 draft：W4 QAT 与 8-expert MoE（2026-11-01）

> 注：本节 QAT 是对 **3.43B dense draft** 做的。对 **34.66B 原模型本身**做 W4 QAT
> （真正的“容量上界”实验）见 §13。

> 目标：在 §1 的 dense draft（3.43B，激活=teacher）之外，验证两种“更极端”的压缩/结构：
> (a) **W4 weight-only QAT** 的 dense draft；(b) **只保留 8 个专家、top-8 全激活**的 MoE draft
> （激活参数与 teacher 完全一致，总参数 3.44B ≈ dense）。
> 两者都因为“容量没怎么降”，接受率大概率很好，所以重点看 **长 γ**（γ 最大到 32）。

### 12.1 设置

| 模型 | 初始化 | 训练量 | 总参数 | 激活 FFN | bf16 体积 | W4 打包体积 |
|---|---|---|---|---|---|---|
| dense-distill（§1） | top-8+shared 拼宽 + FFN fit | 2.9M tok | 3.43B | 4608 | 6.87GB | — |
| dense-ctrl（对照） | dense-ffnfit | 1.17M tok | 3.43B | 4608 | 6.87GB | — |
| bf16-ctrl（对照） | dense-distill | 1.17M tok | 3.43B | 4608 | 6.87GB | — |
| **QAT-W4** | dense-distill + W4 fake-quant | 1.17M tok | 3.43B | 4608 | — | **2.49GB** |
| **MoE8** | 256→8 experts，保留 top-8 精确权重 | 1.17M tok | **3.44B** | 8×512+512 | 6.87GB | — |
| **MoE16** | 256→16（8 个精确 + 8 个尾部聚类） | 0.97M tok | 4.44B | 8×512+512 | 8.88GB | — |

- **QAT**：`code/quant.py`，group=128 对称 int4，所有 `in/out ≥128` 的 Linear（含 lm_head）都量化；
  训练时每个 optimizer step 量化一次权重、STE 回传（`refresh_fake_quant`），再跑 80 step CE。
- **MoE8/MoE16**：`code/moe_draft.py`。把 teacher 每层 256 个 expert 按 router 方向做 usage-weighted
  k-means；MoE8 直接保留最常用的 8 个 expert 的原始权重（其余 248 个丢掉），MoE16 保留 8 个精确 +
  8 个尾部聚类（每个聚类内按 usage 加权平均 gate/up/down）。router 同步降到 8/16 行。
  **top-k 仍然是 8，expert width 仍然是 512，shared expert 不变**，所以激活参数与 teacher/dense 完全一致。
- 评测：MATH-500 全量 500 题、greedy、非思考，γ ∈ {1,2,3,5,8,12,16,24,32}；
  另外报告 **survival**：`P(连续 k 个 draft token 全部命中 teacher)`，直接看“长预测”的准确率。

### 12.2 长 γ 接受长度

| 模型 | token top-1 | τ(γ=1) | τ(γ=3) | τ(γ=5) | τ(γ=8) | τ(γ=12) | τ(γ=16) | τ(γ=24) | τ(γ=32) |
|---|---|---|---|---|---|---|---|---|---|
| dense-distill | 85.50% | 1.84 | 3.12 | 3.99 | 4.87 | 5.54 | 5.95 | 6.37 | 6.57 |
| dense-RL | 87.01% | 1.86 | 3.20 | 4.16 | 5.15 | 5.96 | 6.45 | 6.99 | 7.24 |
| dense-ctrl（同预算） | 85.65% | 1.84 | 3.13 | 4.03 | 4.91 | 5.62 | 6.02 | 6.46 | 6.65 |
| bf16-ctrl（同预算） | 86.76% | 1.86 | 3.19 | 4.14 | 5.11 | 5.91 | 6.38 | 6.90 | 7.14 |
| PTQ-W4 | 85.15% | 1.84 | 3.10 | 3.96 | 4.81 | 5.46 | 5.85 | 6.24 | 6.43 |
| **QAT-W4** | **86.63%** | 1.85 | 3.18 | 4.13 | 5.09 | 5.87 | 6.34 | 6.83 | **7.08** |
| **MoE8** | **88.32%** | 1.87 | 3.27 | 4.32 | **5.43** | **6.38** | **6.97** | **7.62** | **7.96** |
| MoE16 | 88.08% | 1.87 | 3.26 | 4.29 | 5.38 | 6.30 | 6.87 | 7.50 | 7.83 |

> `dense-ctrl`：从 dense-ffnfit（75.1%）起，用和 MoE8 **完全相同**的 1.17M token / 80 step 训一个
> bf16 dense，用来隔离“结构差异”。`bf16-ctrl`：从 dense-distill（85.5%）起，用和 QAT-W4
> **完全相同**的 1.17M token / 80 step 训一个不量化的 bf16，用来隔离“量化差异”。

长预测 survival（`P(run≥k)`）：

| 模型 | k=1 | k=2 | k=3 | k=5 | k=8 | k=16 | k=32 | k=64 |
|---|---|---|---|---|---|---|---|---|
| dense-distill | 85.5% | 75.3% | 67.6% | 56.0% | 43.8% | 25.5% | 10.7% | 2.8% |
| dense-RL | 87.0% | 77.6% | 70.3% | 59.1% | 47.1% | 28.4% | 12.6% | 3.6% |
| dense-ctrl（同预算） | 85.7% | 75.4% | 67.5% | 55.6% | 43.3% | 24.6% | 9.7% | 2.3% |
| bf16-ctrl（同预算） | 86.8% | 77.2% | 69.7% | 58.2% | 46.0% | 27.3% | 11.6% | 3.2% |
| **QAT-W4** | 86.6% | 76.9% | 69.4% | 58.0% | 45.9% | 27.2% | 11.7% | 3.2% |
| **MoE8** | **88.3%** | **79.6%** | **72.7%** | **61.8%** | **49.9%** | **30.8%** | **13.8%** | **3.8%** |
| MoE16 | 88.1% | 79.2% | 72.1% | 61.1% | 49.2% | 30.2% | 13.3% | 3.6% |

![τ vs γ](out/fig_tau.png)

![survival](out/fig_survival.png)

### 12.3 W4 QAT：量化几乎不伤接受率（同预算对照后确认）

- **纯量化（PTQ-W4，不训练）**：85.15% vs bf16 85.50%，只掉 **0.35pt**；
  τ(γ=8) 4.81 vs 4.87（-1.2%），τ(γ=32) 6.43 vs 6.57（-2.1%）。
  量化误差随 γ 略累积，但即使 γ=32 也只有 ~2%。
- **同预算 bf16 对照**（从 dense-distill 起，同样 1.17M token / 80 step，不量化）：
  86.76%，τ(γ=8)=5.11、τ(γ=32)=7.14。
- **QAT-W4**（同样 1.17M token / 80 step）：86.63%，τ(γ=8)=5.09、τ(γ=32)=7.08。
  与同预算 bf16 对照只差 **-0.13pt token、-0.4% τ(γ=8)、-0.8% τ(γ=32)**——**基本无损**。
- 所以 QAT 相对原始 dense-distill 的提升（85.50%→86.63%）主要是继续训练的红利；
  **干净的结论是：PTQ 先掉 0.35pt，QAT 把这 0.35pt 补回来；同预算下 W4 QAT ≈ bf16。**
  group=128、含 lm_head 的 W4 对 draft 接受率几乎免费，且长 γ 也没有额外崩塌。

### 12.4 少专家 MoE：8 个专家就够，且长 γ 反超 dense

- **初始化**（全量 500 题）：dense 拼宽 65.63%、dense+FFN fit 75.13%；
  MoE8（top-8 精确 + 丢掉尾部）69.42%、MoE16（+8 个尾部聚类）70.95%。
  只保留 top-8 精确专家就比 dense 拼宽高 3.8pt，说明**保留专家分块 + 可学习 router** 比“拼成一个宽 FFN”更好。
- **蒸馏后**：MoE8 88.32%、MoE16 88.08%。**E=16 比 E=8 多 1.0B 总参数，却没有更好**
  （τ(γ=32) 7.83 vs 7.96，差异在噪声内）——尾部 8 个聚类专家在训练后基本没贡献。
- **相对 dense**（同 3.44B 总参数、同激活）：
  - **vs dense-ctrl（完全相同 1.17M token / 80 step）**：token **+2.67pt**，
    τ(γ=8) **+10.6%**，τ(γ=32) **+19.7%**。而且 dense-ctrl 的初始化还更好
    （75.1% vs MoE8 的 69.4%）——**同样的训练预算、同样的总/激活参数，MoE8 结构上就是更强**。
  - vs dense-distill：token +2.82pt，τ(γ=8) +11.5%，τ(γ=32) **+21.2%**
  - vs dense-RL：token +1.31pt，τ(γ=8) +5.4%，τ(γ=32) **+9.9%**
  - 优势随 γ 单调变大（vs dense-ctrl：+7.2% @γ=5 → +19.7% @γ=32），
    因为 MoE 的 per-token 路由让长块内的错误更少累积。
- survival 也一致：MoE8 的 `P(run≥32)=13.8%` vs dense-RL 12.6% / dense-distill 10.7%；
  `P(run≥64)=3.8%` vs 3.6% / 2.8%。

### 12.5 工程含义（按 §5 的带宽受限模型估算）

`speedup ≈ τ / (1 + γ·c)`，c = draft/teacher 单 token 字节数比；teacher bf16 ≈ 69.3GB。

| 模型 | c_bw | γ=5 | γ=8 | γ=12 | γ=24 |
|---|---|---|---|---|---|
| dense-RL | 0.099 | 2.78× | 2.87× | 2.72× | 2.07× |
| dense-ctrl（同预算） | 0.099 | 2.69× | 2.74× | 2.56× | 1.91× |
| bf16-ctrl（同预算） | 0.099 | 2.77× | 2.85× | 2.70× | 2.04× |
| **QAT-W4** | **0.036** | 3.50× | 3.95× | **4.10×** | 3.67× |
| **MoE8** | 0.099 | 2.89× | 3.03× | 2.91× | 2.26× |

- **QAT 把 c 从 0.10 压到 0.036**，跨过 §6 的盈亏平衡点（c*≈0.10~0.17），
  带宽受限下 γ=8~12 的估计加速从 dense 的 ~2.7-2.9× 提到 **~4.0×**。
- **MoE8 不降内存**（c 仍 0.099），但靠更高的 τ 在同等带宽下比 dense-RL 多拿 5%~9%
  （比同预算 dense-ctrl 多 11%~20%）。
- 两者正交：**把 MoE8 也做 W4 QAT**，c 仍 ≈0.036、τ 用 MoE8 的，估计 γ=8~12 可到 **~4.2-4.5×**。
  这是下一步最值得做的组合（本次未跑）。
- 再次强调：以上都是**带宽受限**估计；三种 draft 的每 token FLOPs 都和 teacher 一样（激活参数相同），
  在 compute-bound 区间投机解码本身就不划算。

### 12.6 局限

- 只测数学、非思考、greedy；QAT/ MoE 都只训了 ~1M token、80-100 step，**没有调参/加数据**，
  绝对数值还有提升空间（尤其 MoE 的初始化还可以用“逐层输出拟合”而不是 router 聚类）。
- MoE8 丢掉了尾部 248 个 expert；E=16 用 8 个聚类近似尾部，但训练后并不更好，
  说明在当前数据/训练量下尾部专家确实可以丢。
- QAT 已经用**同预算 bf16 对照**（§12.3）隔离了训练量：同 1.17M token / 80 step 下
  QAT 只比 bf16 低 0.13pt；PTQ-W4 那一行是更保守的“纯量化、不训练”损失（-0.35pt）。
- 速度是公式估算，不是 vLLM/SGLang 实测；W4 kernel 的实际吞吐还取决于是否真的用 4-bit GEMM
  （本实验是 fake-quant + dequantized bf16 权重，**接受率**等价，但 wall-clock 未测）。

### 12.7 复现

```bash
# QAT：先看纯 PTQ，再 QAT
python code/quant_probe.py --limit 500 --bits 4 --lm_head_bits 4 --gammas 1,2,3,5,8,12,16,24,32
torchrun --nproc_per_node=3 code/train_qat.py --ckpt ckpt/dense_distill_s1 \
  --out ckpt/dense_qat_w4 --steps 80 --lr 5e-5 --bits 4 --lm_head_bits 4

# 少专家 MoE：256 -> 8（top-8 全激活，激活参数不变）
python code/moe_draft.py --num_experts 8 --keep_top 8 --out ckpt/moe8_keep8
torchrun --nproc_per_node=3 code/train_draft.py --ckpt ckpt/moe8_keep8 \
  --out ckpt/moe8_keep8_distill --steps 80 --lr 1e-4 --max_tok 6144

# 同预算对照：dense（隔离结构差异）与 bf16（隔离量化差异）
torchrun --nproc_per_node=3 code/train_draft.py --ckpt ckpt/dense_ffnfit \
  --out ckpt/dense_ctrl_1m --steps 80 --lr 1e-4 --max_tok 6144
torchrun --nproc_per_node=3 code/train_draft.py --ckpt ckpt/dense_distill_s1 \
  --out ckpt/dense_bf16ctrl_1m --steps 80 --lr 5e-5 --max_tok 6144

# 评测 + 汇总
python code/eval_accept.py --ckpt ckpt/moe8_keep8_distill \
  --gammas 1,2,3,4,5,6,8,10,12,16,20,24,32
python code/summarize_new.py
```

---

## 13. 对“原模型”本身做 W4 QAT，再当 draft（真正的极限情况）

> §12 的 QAT 是对 3.43B 的 dense draft 做的。这里按“对原模型做 QAT”补上：
> 把 **34.66B / 激活 3.4B 的 teacher 自己**量化成 W4，再用它当 draft。
> 它和 target **容量完全相同**，只剩量化噪声——这是“draft 容量”维度的上界实验。

### 13.1 设置

- **量化**：W4 group=128 对称，覆盖所有大 Linear（attention / shared expert / lm_head）
  以及 256 个 routed expert 的 fused `gate_up` / `down`。
- **显存技巧**：routed experts（32.2B）以 **packed int4** 存（16.61GB），forward 时只反量化
  当前被选中的 8 个 expert。训练时非 expert 仍保持 bf16，整模型 ~21GB；
  部署时非 expert 也 W4，packed 总量 ~18.6GB。这样一张 A100 就能做 QAT。
- **PTQ**：RTN，不训练。
- **QAT（部分）**：routed experts 冻结在 PTQ W4；只训 **1.94B 的 always-active 参数**
  （attention + router + shared expert + norms + lm_head），100 step × 512 token，
  lr 2e-5，数据仍是 teacher 自生成输出。
  完整 35B QAT 需要 FSDP；本机 8×A100 的 FSDP 初始化会把 70GB 全量搬到每张卡而 OOM，
  所以这里先给 **full QAT 的下界**。
- **评测**：MATH-500 greedy 非思考；PTQ 跑全量 500 题，QAT 跑前 100 题
  （与 PTQ 的前 100 题是同一子集，可直接比）。
- **上界**：全精度 teacher 当 draft，token 100%、τ(γ)=γ+1（理想值）。

### 13.2 长 γ 接受长度

| draft | bits | packed | token top-1 | τ(5) | τ(8) | τ(12) | τ(16) | τ(24) | τ(32) | τ(48) |
|---|---|---|---|---|---|---|---|---|---|---|
| dense MoE8（§12 参照） | 16 | 6.87GB | 88.32% | 4.32 | 5.43 | 6.38 | 6.97 | 7.62 | 7.96 | — |
| teacher **PTQ-W4**（全 500） | 4 | ~18.6GB | **96.06%** | 5.39 | 7.59 | 10.09 | 12.12 | 15.29 | 17.53 | 20.28 |
| teacher PTQ-W4（100 子集） | 4 | ~18.6GB | 96.65% | 5.48 | 7.78 | 10.45 | 12.67 | 16.36 | 19.00 | 22.72 |
| teacher **QAT-W4**（100 子集） | 4 | ~18.6GB | **96.97%** | 5.52 | 7.90 | 10.64 | 13.09 | 17.05 | **19.95** | — |
| teacher bf16（上界） | 16 | 69.3GB | 100% | 6 | 9 | 13 | 17 | 25 | **33** | 49 |

长预测 survival（`P(run≥k)`）：

| draft | k=8 | k=16 | k=32 | k=64 |
|---|---|---|---|---|
| dense MoE8 | 49.9% | 30.8% | 13.8% | 3.8% |
| teacher PTQ-W4（全 500） | 75.4% | 59.5% | 39.0% | 19.0% |
| teacher QAT-W4（100 子集） | 79.7% | 64.8% | 44.0% | 21.8% |
| teacher bf16（上界） | 100% | 100% | 100% | 100% |

![teacher QAT τ vs γ](out/fig_teacher_qat_tau.png)

![teacher QAT survival](out/fig_teacher_qat_survival.png)

### 13.3 结论

1. **原模型 W4 当 draft 已经是 near-oracle。** 全 500 题 token top-1 **96.06%**
   （dense MoE8 88.32%）；τ(γ=32)=**17.53**（dense 7.96，2.2×），
   `P(run≥32)=39.0%`（dense 13.8%），`P(run≥64)=19.0%`（dense 3.8%）。
   也就是说：**draft 容量一旦接近 target，长 γ 的接受长度可以远超任何 3.4B draft。**
   （W3 PTQ 会崩到 token 90.19%、τ(32)=9.36，见 §15.3——**W4 基本是这条路的极限**。）
2. **QAT 再涨一点，而且长 γ 涨得更多。** 同一 100 题上：
   token 96.65%→**96.97%**，τ(8) 7.78→7.90（+1.5%），τ(16) 12.67→13.09（+3.3%），
   τ(32) 19.00→**19.95（+5.0%）**，`P(run≥32)` 41.0%→**44.0%**。
   这说明 QAT 主要在修“长块内量化误差累积”，而不是短块的单 token 错误。
3. **但离上界仍有明显距离。** γ=32 的理想 τ 是 **33**，QAT-W4 只有 **19.95（~60%）**；
   瓶颈是 32B routed experts 的量化噪声，不是 router/lm_head。
   要再往上必须 **full QAT 更新 expert 权重**（FSDP/ZeRO）或更高位宽（W6/W8）。
4. **工程上 W4 原模型不是“速度 win”，而是“接受率上界”。**
   packed 18.6GB → `c_bw≈0.27`；按 §5 的带宽受限模型，最优 γ≈10-12、约 **2.5×**，
   略低于 dense MoE8 的 **3.0×**（γ=8）——因为每 token 成本高了 2.7×。
   若压到 W2/W3（c≈0.14/0.20），长 γ 的高 τ 才可能转成更高的端到端加速；
   但 W2 在 dense 上已经崩，原模型 W2 能否靠 QAT 救回来是另一个问题。

### 13.4 局限

- QAT 是**部分 QAT**：experts 冻结在 PTQ W4，只训 1.94B always-active 参数。
  这是 full 35B QAT 的**下界**；完整 QAT 应该更接近 bf16 上界。
- 只测数学/非思考/greedy；QAT 只有 100 step × 512 token（~51k token），没有调参。
- QAT 的 expert 反量化是“per-selected-expert unpack”，接受率等价于 W4 推理，
  但 wall-clock 没测；真实 4-bit kernel 的吞吐会更好。
- 速度是带宽受限公式估算；原模型激活 FLOPs 和 teacher 一样，compute-bound 下没有投机收益。

### 13.5 复现

```bash
# PTQ：全量 500 题（约 44 min / A100）
CUDA_VISIBLE_DEVICES=0 python code/quant_teacher.py --limit 500 \
  --bits 4 --lm_head_bits 4 --bs 1 \
  --gammas 1,2,3,4,5,6,8,10,12,16,20,24,32,48 \
  --out out/eval_teacher_ptq_w4_500.json

# 部分 QAT：packed experts + 训 always-active 1.94B（约 50 min / A100）
CUDA_VISIBLE_DEVICES=1 python code/train_teacher_qat.py \
  --steps 100 --max_len 512 --pack_experts \
  --parts attn,router,shared,norm,lm_head --eval_limit 100 \
  --out ckpt/teacher_qat_w4_100.pt \
  --eval_out out/eval_teacher_qat_w4_100.json

# 重新加载 QAT state 单独评测（验证可复现）
CUDA_VISIBLE_DEVICES=0 python code/eval_teacher_qat.py \
  --ckpt ckpt/teacher_qat_w4_100.pt --limit 100

python code/plot_teacher_qat.py
```

---

## 14. 深度律：接受率是“线性衰减”还是“收益递减”？

> 观察：把 block acceptance α(γ) 画出来，短 γ 段看起来像直线。
> 这一节先检验“线性”是否成立，再给出“分布微扰 → 序列累积 → 接受率”的动力学模型，
> 最后用 dense / MoE8 / teacher QAT / 原生 MTP 的数据说明为什么。

### 14.1 先定义清楚“深度”

- \(p_t=\Pr[\arg\max Q_t=\arg\max P_t]\)：第 t 个位置 draft 的 top-1 命中 teacher 的概率。
- \(S(k)=\Pr[\text{连续 k 个 draft token 全命中}]\)：survival。
- \(\alpha(\gamma)=\frac1\gamma\sum_{k=1}^{\gamma}S(k)\)：block acceptance（被接受草稿 token / 提出的草稿 token）。
- \(p_k=S(k)/S(k-1)\)：**条件**接受率（已经活到深度 k 时，第 k 个 token 的命中概率）。

**关键区分**：α(γ) 随 γ 下降；但 \(p_k\) 不一定下降。如果 \(p_k\) 上升，说明不是“越深越差”，而是“活到深处的都是容易的段”。

### 14.2 实验结果：短 γ 近似线性，但全局是凹的；\(p_k\) 反而上升

| 模型 | p | α(1) | α(5) | 局部斜率 γ=1→5 | \(-p(1-p)/2\) | β（stretched） | \(S(32)/p^{32}\) | \(p_{64}/p_1\) |
|---|---|---|---|---|---|---|---|---|
| dense-RL（500） | 87.01% | 0.8575 | 0.6345 | **-0.0558** | **-0.0565** | 0.715 | 10.80 | 1.105 |
| MoE8（500） | 88.32% | 0.8725 | 0.6659 | **-0.0516** | **-0.0516** | 0.744 | 7.34 | 1.088 |
| dense（100） | 89.64% | 0.8872 | 0.6974 | **-0.0475** | **-0.0465** | 0.765 | 4.51 | 1.072 |
| MTP（100） | 95.74% | 0.9541 | 0.8683 | **-0.0215** | **-0.0204** | 0.846 | 1.55 | 1.020 |
| teacher PTQ-W4（500） | 96.06% | 0.9598 | 0.8818 | **-0.0195** | **-0.0189** | 0.848 | 1.41 | 1.018 |
| teacher QAT-W4（100） | 96.97% | 0.9697 | 0.9083 | **-0.0154** | **-0.0147** | 0.910 | 1.18 | 1.009 |

三条硬结论：

1. **局部线性是真的，而且斜率可以被一个极简公式精确预测**：\(\alpha(\gamma)\approx p-\frac{p(1-p)}2(\gamma-1)\)。
   六个模型的实测斜率与 \(-p(1-p)/2\) 的差都在 0.001~0.002 以内（MoE8 几乎完全相等）。
   这说明短 γ 的“线性下降”不是累积动力学，而是**前缀平均的几何效应**（见 §14.3）。
2. **全局不是线性**。线性拟合的 R²：dense 在 γ≤6 是 0.997，γ≤12 降到 0.984，γ≤32 只有 0.915；
   MTP 因为 p 高、异质性小，γ≤32 仍有 0.982，但也不是严格直线。
   真正的形状是**凹的（diminishing returns）**，可以用 stretched exponential
   \(S(k)=e^{-(k/\lambda)^\beta}\) 描述，β<1；β 随 draft 质量单调上升：
   dense-RL 0.715 → MoE8 0.744 → MTP 0.846 → teacher QAT 0.910（越接近 1 越接近指数/无记忆）。
3. **条件接受率 \(p_k\) 随深度上升，不是下降**：
   dense 从 \(p_1=0.896\) 升到 \(p_{64}=0.956\)；MTP 从 0.957 升到 0.977。
   长 runs 比 i.i.d. 重尾得多：dense-RL 的 \(S(32)/p^{32}=10.8\)、\(S(64)/p^{64}\approx 265\)
   （dense-distill 更是 ~623）；MTP 只有 1.55 / ~2.9。
   这排除了“per-token 接受率随深度线性衰减”的简单图像。

![depth law](out/fig_depth_law.png)

### 14.3 动力学理论

#### (A) 微观：分布微扰 → 单 token 接受概率

设 target 的 next-token 分布为 \(P_t\)，draft 为 \(Q_t=P_t+\epsilon_t\)。greedy 下
\[
p_t=\Pr[\arg\max Q_t=\arg\max P_t]
   \approx 1-c\,\sigma_t^2,
\]
其中 \(\sigma_t^2\) 是 logit 微扰的方差（等价地，\(p_t\approx 1-\frac12\|\epsilon_t\|_1\)）。
量化误差、容量差、router 方差都体现在 \(\sigma_t^2\) 里。

#### (B) 无累积（i.i.d. 位置）：线性只是 Taylor 的一阶项

如果 \(\sigma_t^2\equiv\sigma^2\)，则 \(p_t=p\) 为常数，且
\[
S(k)=p^k,\qquad
\alpha(\gamma)=\frac1\gamma\sum_{k=1}^{\gamma}p^k
=\frac{p(1-p^\gamma)}{\gamma(1-p)}.
\]
在 \(\gamma=1\) 附近展开：
\[
\boxed{\alpha(\gamma)=p-\frac{p(1-p)}2(\gamma-1)+O((\gamma-1)^2)}.
\]
**这就是“线性衰减”的来源**：它是常数 p 模型的一阶 Taylor 项，斜率只由 p 决定。
实测六个模型的斜率都落在这条线上（§14.2），所以短 γ 的线性**不需要任何累积机制**。
但常数 p 模型在长 γ 会给出
\(\alpha(\gamma)\to p/[\gamma(1-p)]\)，是 **1/γ 尾部**，不是直线。

#### (C) 序列累积：什么情况下才会出现“per-token 线性衰减”？

真正的 within-block 累积可以写成扰动递推
\[
\epsilon_{t+1}=A\epsilon_t+\eta_t,\qquad
\sigma_{t+1}^2=\rho^2\sigma_t^2+\sigma_\eta^2,
\]
于是
\[
p_t=1-c\left(\sigma_0^2+\sigma_\eta^2\frac{1-\rho^{2t}}{1-\rho^2}\right).
\]
- 若 \(\rho\approx1\)：\(\sigma_t^2\approx\sigma_0^2+\sigma_\eta^2 t\)，所以
  \[
  p_t\approx p_0-c\sigma_\eta^2 t
  \]
  **per-token 接受率线性衰减**；对应
  \[
  S(k)\approx\exp\!\left(-a k-\tfrac12 b k^2\right)
  \]
  是**高斯型** survival（比指数快）。
- 若 \(\rho<1\)：扰动饱和，\(p_t\to p_\infty=1-c\sigma_\eta^2/(1-\rho^2)\)，长 k 又回到指数尾部。
- 若 \(\rho>1\)：误差爆炸，长 block 直接不可用。

**我们的 teacher-forcing 评测把 draft 的上下文永远重置为 teacher 前缀**，等价于 \(\rho=0\)（无累积）；
所以它测到的是“无累积上界”。在这个测量里 \(p_k\) 上升而不是下降，说明
**当前数据不在 \(\rho\approx1\) 的线性衰减区**。真实的 speculative loop 会有 within-block 累积；
现有 `sd_sim.py` 在 γ=5 的实测（α=0.643/τ=4.22 vs teacher-forcing 0.60/3.99）没有看到明显变差，
说明至少短 γ 下 \(\rho\) 不大。长 γ 的 real-loop 实验是区分 (B)/(C) 的关键。

#### (D) 观测到的凹形：clustering + 难度异质性

即使没有 within-block 累积，序列层面的两个效应也会让 α(γ) 偏离直线：

1. **错误成簇 / block-start bias**。实测
   \(P(C_t|C_{t-1})=0.911\)、\(P(C_t|I_{t-1})=0.783\)（dense）；
   MTP 是 0.960 / 0.891。因为一个 block 结束后，下一个 block 从**失败 token 之后**开始，
   所以 block 起点的命中率不是 marginal p，而是 \(P(C|I)\)：
   dense 0.763、MTP 0.886。γ 越大，block 越长、起点越偏向“失败后”，
   这会额外压低 α(γ)。一个只用这两个转移概率的一阶 Markov 链就能复现大部分曲线：

   | 模型 | P(C\|C) | P(C\|I) | block 起点 p | 实测 α(32) | Markov α(32) | 实测 τ(32) | Markov τ(32) |
   |---|---|---|---|---|---|---|---|
   | dense | 0.911 | 0.783 | 0.763 | 0.250 | 0.264 | 8.85 | 9.43 |
   | MTP | 0.960 | 0.891 | 0.886 | 0.499 | 0.521 | 16.59 | 17.66 |

   dense 的 Markov 预测在长 γ 偏高 0.03~0.05，说明还有**更长程**的难度相关；MTP 只偏高 0.02。

2. **难度异质性 / selection**。把序列看成不同“难度段”的混合：有些位置 p 高，有些低。
   于是
   \[
   S(k)=\mathbb{E}[p^k],\qquad
   p_k=\frac{\mathbb{E}[p^k]}{\mathbb{E}[p^{k-1}]},
   \]
   后者随 k **上升**——活到深处的 block 已经被选择成“容易段”。这就是 §14.2 里
   \(p_k\) 上升、\(S(k)\) 比 \(p^k\) 重尾的原因。两种可解释的参数化都拟合得很好：
   - Beta 混合 \(p\sim\mathrm{Beta}(a,b)\)：\(S(k)=B(a+k,b)/B(a,b)\)，R²≈0.993~0.9999；
   - stretched exponential \(S(k)=e^{-(k/\lambda)^\beta}\)，β<1，R²≈0.997~0.9999。
   β 是“异质性指数”：β→1 表示位置同质、接近无记忆；β 越小，长 runs 越重尾、α(γ) 越凹。

**综合图像**：
\[
\boxed{
\alpha(\gamma)=\frac1\gamma\sum_{k=1}^{\gamma}S(k),\qquad
S(k)\approx e^{-(k/\lambda)^\beta},\ \beta<1
}
\]
短 γ 时它退化成 \(\alpha\approx p-\frac{p(1-p)}2(\gamma-1)\)（线性）；
长 γ 时 \(\tau(\gamma)=1+\sum_{k=1}^{\gamma}S(k)\) **收敛到有限 \(\tau_\infty\)**，
边际收益 \(S(\gamma)\) 按 \(e^{-(\gamma/\lambda)^\beta}\) 衰减、\(\alpha(\gamma)\) 按 \(1/\gamma\) 衰减，
是**收益递减**，不是线性。
线性区之外，block-start bias（clustering）让它先掉得比常数 p 模型快一点，
selection（异质性）又让长尾比常数 p 模型高一点，两者竞争形成实测的凹形。

### 14.4 用这个理论解释原生 MTP

MTP 在三个量上都更好，所以它的曲线更接近直线、长 γ 更强：

| 量 | dense | MTP | 含义 |
|---|---|---|---|
| p（marginal） | 0.896 | **0.957** | 初始斜率 \(-p(1-p)/2\)：-0.047 vs **-0.021**，直接减半 |
| P(C\|I)（失败后下一 token） | 0.783 | **0.891** | block-start bias 更小：起点 p 0.763 vs **0.886** |
| β（stretched） | 0.72~0.78 | **0.85** | 难度异质性更小，长尾更接近指数 |
| \(S(32)/p^{32}\) | 4.5~10.8 | **1.55** | 长 runs 不再那么重尾 |
| \(p_{64}/p_1\) | 1.07~1.10 | **1.02** | selection 效应小，条件接受率几乎不随深度变 |

机制上：MTP 每一步都吃 **target 的 hidden state** \(H_t\)，而不是只吃自己上一步的 token。
用 §14.3(C) 的语言，这相当于把扰动递推的 \(\rho\) 压小、把 \(\sigma_\eta\) 压小：
**每步都从 target 表示重新注入信息，扰动不累积**。所以 MTP 的 p 高、β 接近 1、
α(γ) 在 γ≤12 都近似线性；dense draft 只能靠自己的 KV/上下文，扰动更容易累积、
且难度异质性更大（β≈0.72），所以长 γ 掉得更快。
报告 §9.2 里“MTP 的 persistent cache 带来 +6pt（89.5%→95.7%）”也可以从这个角度理解：
它不是单纯的 cache 技巧，而是**让 draft 每步都对齐 target 的表示，抑制扰动累积**。

### 14.5 对 scaling law / draft 设计的含义

1. **scaling law 的“接受率”至少有两个维度**：平均扰动 \(p\)（决定初始斜率）和异质性/聚类
   \(\beta\)（决定长 γ 尾部）。只堆容量把 p 拉高，但如果不降 β，长 γ 仍然会凹得很快。
2. **提升长 γ 的正确方向是降 β / 降 clustering**，而不只是涨平均 p：
   - MoE8 相对 dense-RL：p 0.870→0.883，β 0.715→0.744；
   - MTP：p 0.957，β 0.853；
   - teacher QAT：p 0.970，β 0.910——几乎回到无记忆指数。
   这与§12/§13 的 τ(γ) 排序完全一致。
3. **最优 γ 的停止规则**：由 \(\text{speedup}=\tau(\gamma)/(1+\gamma c)\) 求导，
   \[
   S(\gamma+1)(1+\gamma c)\approx c\,\tau(\gamma).
   \]
   因为 \(S(\gamma+1)\) 随 γ 下降，最优 γ 有限；dense 约 8（c≈0.1）、teacher QAT 约 10~12（c≈0.27），
   与§12/§13 的带宽受限估算一致。MTP 的 head 很便宜（c 更小），最优 γ 会更靠后，
   具体取决于实现的 c。**不要因为 α(γ) 在短段看起来线性就一直加大 γ**：
   线性只是局部，过了拐点每多一个 draft token 的边际收益 \(S(\gamma+1)\) 已经低于它的成本。

### 14.6 局限与可证伪预测

- 本节主分析用 teacher-forcing 的 per-token match（`composite2.npz`，100 题 dense/MTP），
  不含 within-block 扰动累积；真实 speculative loop 的长 γ 曲线可能更低。
- 一阶 Markov 模型对 dense 的长 γ 仍偏高 0.03~0.05，说明存在**长程**难度相关；
  需要 latent-state / 半 Markov 模型才能完全拟合。
- **可证伪预测**：如果 within-block 累积是主因，那么 real-loop 的 \(p_k\) 应随深度下降，
  且长 γ 的 real-loop survival 会明显低于 teacher-forcing survival；
  如果 §14.3(B)(D) 是对的，real-loop 的 \(p_k\) 仍应上升或基本持平，
  只是 block-start bias 会让绝对 α 低一点。用 `sd_sim.py` 跑 γ=16/32 并记录每个 block 的
  per-depth 命中即可判定。
- MTP 的“target hidden state 注入”如果真是抑制累积，那么把 dense draft 也改成
  “每步吃 target hidden state”（即 EAGLE/MTP 式）应该让 β 明显上升——
  这正是§9.2 里 MTP 相对 dense 的优势来源。

### 14.7 复现

```bash
python code/depth_theory.py       # 表格/拟合
python code/plot_depth_law.py     # 图 out/fig_depth_law.png
```

---

## 15. 动态深度（熵/置信度聚类）与“大 draft vs MTP”

> 问题：(a) 如果 γ 由聚类/熵动态决定，teacher QAT 的收益会不会比 MoE8 更大？
> (b) 大参数量 draft 相对原生 MTP 的优势，是不是在长 γ？

### 15.1 方法

- 100 道 MATH-500，teacher-forced per-token `match` + draft 的 `maxp`/entropy
  （`out/features_{moe8,teacher_qat}.npz`，dense/MTP 用 `composite2.npz`）。
- 策略：
  - `fixed`：全局一个 γ；
  - `cluster`：按 block 起点 draft confidence 的 4 个分位 bin，每个 bin 学一个 γ；
  - `sequential`：逐 token 看 confidence，低于阈值就停；
  - `2D cluster`：再用上“上一 token 的 teacher entropy”做 2D 聚类；
  - `oracle cluster`：用 test 标签给每个 bin 选最优 γ，作为上界。
- 5-fold CV（按 sequence 切分）。指标
  \[
  \text{speedup}=\frac{\text{accepted tokens}}{\sum (1+\gamma c)},
  \]
  c 用带宽受限的 per-token 成本：dense/MoE8 0.099，MTP 0.03，teacher QAT 0.268。

### 15.2 结果：动态深度的相对收益，弱 draft 更大

| draft | c | fixed γ | fixed speedup | cluster speedup | **gain** | oracle gain | τ fixed→dyn |
|---|---|---|---|---|---|---|---|
| dense | 0.099 | 8 | 3.195 | 3.359 | **+5.1%** | +5.6% | 5.71→6.20 |
| **MoE8** | 0.099 | 8~10 | 3.274 | 3.457 | **+5.6%** | +6.0% | 6.09→6.55 |
| MTP | 0.03 | 32 | 8.519 | 8.736 | **+2.5%** | +3.3% | 16.06→15.81 |
| **teacher QAT** | 0.268 | 10 | 2.548 | 2.586 | **+1.5%** | +2.3% | 9.32→10.04 |

**teacher QAT 的相对收益反而最小**（+1.5% vs MoE8 +5.6%）。原因不是 entropy 信号没用，而是：

- teacher QAT 的 confidence 是**双峰**的：最低 20% 的 token match 84.9%，
  其余 80% 的 match ≥99.98%。fixed γ=10 已经接近这个双峰分布的最优；
  动态 depth 只能修那 20% 的硬尾巴，净收益小。
- MoE8/dense 的 confidence 是**渐变**的：最低 20% 只有 58% match，
  后面 93% / 99.5% / 99.9% / 99.98% 逐级上升。fixed γ=10 对低置信度段太长、
  对高置信度段又太短，所以 cluster 能拿到 +5~6%。
- 这与 §14 的 β 完全一致：teacher QAT 的 β=0.910（异质性最小），
  自适应 depth 的可利用空间也最小；MoE8/dense 的 β≈0.72~0.74，空间更大。

| draft | 最低 20% confidence 的 match | 其余 80% | β（§14） |
|---|---|---|---|
| dense | 57.5% | 92.1 / 98.9 / 99.7 / 99.95% | 0.765 |
| MoE8 | 58.2% | 93.0 / 99.5 / 99.9 / 99.98% | 0.744 |
| MTP | 78.9% | ≥99.75% | 0.846 |
| teacher QAT | 84.9% | ≥99.98% | 0.910 |

学出来的 γ/bin（低 confidence → 高 confidence）：

- dense/MoE8：`[4, 10, 12, 16]` / `[4, 10, 16, 16]`
- MTP：`[20~24, 32, 32, 32]`
- teacher QAT：`[4~8, 10~12, 16, 16]`

**2D 聚类（draft confidence + 上一 token 的 teacher entropy）没有额外收益**：
dense 3.343 vs 3.359，MoE8 3.450 vs 3.457，MTP 8.802 vs 8.736，teacher QAT 2.585 vs 2.586。
也就是说 **draft 自己的 confidence 已经吃掉了几乎全部可用信号**；再加 teacher entropy
只多 0~0.8%，与 §9.1 的结论一致（额外特征只多 0.002 AUC）。

### 15.3 大 draft vs MTP：优势确实在长 γ，但要付成本

固定 γ 下，teacher QAT 相对 MTP 的 **τ 优势随 γ 单调变大**：

| γ | 1 | 2 | 3 | 5 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| τ_QAT / τ_MTP | 1.008 | 1.014 | 1.025 | 1.037 | **1.063** | 1.092 | 1.118 | 1.168 | **1.203** |

所以“大 draft 的优势在长 γ”在 **acceptance/τ** 上是成立的：γ=8 只高 6%，γ=32 高 20%。

但换成 **speedup**，要把每 token 成本 c 算进去。**在相同 c 下**，teacher QAT 一直赢：

| c | teacher QAT | MTP | MoE8 |
|---|---|---|---|
| 0.05 | **7.839** | 6.699 | 4.426 |
| 0.10 | **5.116** | 4.542 | 3.259 |
| 0.15 | **3.889** | 3.504 | 2.680 |
| 0.20 | **3.151** | 2.904 | 2.306 |
| 0.27 | **2.534** | 2.364 | 1.937 |

但 **实际 c 不是相同的**：teacher QAT W4 packed 18.6GB → c≈0.268；
MTP head 只有 ~0.83B 总参数/很小激活 → c≈0.03。按实际 c：

- MTP：8.519×（γ=32）
- teacher QAT W4：2.548×（γ=10）
- MoE8：3.274×（γ=8~10）
- dense：3.195×（γ=8）

**MTP 的绝对 speedup 仍比 teacher QAT 高 3.3×**（比 MoE8 高 2.6×），
因为它的 c 低约 9×。大 draft 的 τ 优势（γ=32 高 20%）不足以抵消 9× 的成本差。

那能不能把 teacher QAT 压到 W3，把 c 降到 ~0.20？**W3 PTQ 出现了悬崖**：

| teacher PTQ | c_bw | token top-1 | τ(8) | τ(16) | τ(32) |
|---|---|---|---|---|---|
| W4（100 题） | 0.268 | **96.65%** | 7.78 | 12.67 | **19.00** |
| W3（100 题） | ~0.20 | **90.19%** | 5.95 | 7.94 | **9.36** |

W3 的 τ(32) 从 19.0 掉到 9.36，已经只比 MoE8 的 8~9 略高，而 c 仍是 MoE8 的 2 倍——
按 c≈0.20 算 best speedup 只有 ~2.3×（γ=8），**低于 MoE8 的 3.27×，被 MoE8 支配**。所以“大 draft 的长 γ 优势”要真正兑现，需要
**W4 以下不崩的 QAT**（W3 QAT 也许能救一部分，但 PTQ 的悬崖说明风险很大），
或者 MTP 的实际 per-token 成本高于 ~0.25（例如实现 overhead 很大时）。
在 W4 + 优化过的 MTP 这个组合下，MTP 仍然赢绝对速度。

补充：**如果 MTP 的实际 c 被实现 overhead 推到 ~0.25 以上**（§6 的 HF 实现里
MTP 的实测加速只有 1.2~1.3×，c 并不低），teacher QAT 会在长 γ 反超；
但在优化过的 vLLM/SGLang 里 MTP 的 c≈0.03，仍然赢。所以“大 draft 长 γ 优势”
是否兑现，最终取决于 **MTP 的工程效率**，而不是接受率本身。

![dynamic depth](out/fig_dynamic_depth.png)

### 15.4 与 §14 的统一解释

- 动态深度的收益 \(\propto\) 异质性：\(\beta\) 越低（dense/MoE8），confidence 越渐变，
  fixed γ 越不优，cluster 收益越大（+5~6%）；\(\beta\to1\)（teacher QAT），
  confidence 双峰且 80% 接近 1，fixed γ 已接近最优，收益只有 +1.5%。
- 所以 **动态深度是“省草稿算力”的杠杆，不是“涨接受长度”的杠杆**：
  它最多把 speedup 提高 ~5%，oracle 也只有 ~6%；真正把长 γ τ 拉上去的是
  p 和 β（容量/结构/训练），不是 gating。
- 这与 §7/§8 的结论一致：confidence 是强信号（AUC 0.88~0.92），
  但用它做单路径 adaptive depth 只能省 6~10% 算力；要涨接受长度必须多候选/树，
  或者换更强的 draft。

### 15.5 建议

- **要绝对 speedup**：MTP 仍然是最优（c≈0.03，8.5×）；除非 MTP 的实际
  per-token cost 被实现 overhead 推到 0.25 以上。
- **要长 γ 的接受率 / 或 MTP 成本较高**：teacher QAT W4 是最强的（τ(32)=19.0~19.95），
  但 packed 18.6GB 是代价；不要指望 W3 能便宜地替代它。
- **性价比最高的中间点**：MoE8 + confidence-cluster 动态深度
  （c=0.099，fixed 3.27× → cluster 3.46×），比 dense 略好，远好于 teacher QAT 的 2.59×。
- **不要把动态深度用在 near-oracle draft 上**：teacher QAT 的 +1.5% 不值得引入
  gating 的复杂度和延迟；它的收益来自容量/β，不是来自 per-token 选择。

### 15.6 局限

- 100 题；confidence 是 teacher-forced 的（真实 block 内会用 draft 自己的前缀，
  可能更保守）；没有 within-block 累积。
- c 是带宽受限估计；真实 kernel 的 MTP/大 draft 吞吐可能不同。
- W3 是 PTQ，不是 QAT；W3 QAT 可能恢复一部分，但 PTQ 的悬崖说明空间不大。
- 动态 depth 的收益是相对 fixed γ 的；如果 baseline 本来就调过 γ，收益会更小。

---

## 16. WikiText 上的 NLL / KL / TV：分布偏移到底是什么样

> 目标：用 WikiText 的 NLL 和 teacher↔draft 的 KL/TV/top-k 指标，系统刻画
> “小 draft / MTP / teacher QAT” 相对 teacher 的分布偏移，并解释 §12~§15 的接受率差异。

### 16.1 设置

- 数据：从 ModelScope 下载的 **WikiText-2 raw test**（`Salesforce/wikitext`，test split），
  取前 100 × 1024 token ≈ 102k token（不是本地 demo）。
- teacher-forced：同一段文本喂给 teacher 和每个 draft，逐位置比较 next-token 分布。
- 指标：
  - NLL / ppl（draft 自己的语言建模质量）；
  - top-1 match、teacher top-1 在 draft top-5 里的 recall、teacher top-1 在 draft 里的 rank；
  - **forward KL** \(D_{KL}(P_T\|Q_D)\)、**reverse KL** \(D_{KL}(Q_D\|P_T)\)、**TV**、**JS**；
  - draft entropy（校准/不确定性）。
- teacher 的 top-256 log-probs + residual bucket；平均 residual \(p_0=1.66\%\)，
  top-1/NLL 是精确的，KL/TV 是高熵 token 上的近似。
- 模型：dense-distill、dense-RL、dense-QAT-W4、MoE8、MoE16、原生 MTP、teacher-QAT-W4。
  （teacher 本身 NLL=2.000、entropy=1.747、next-tok top-1=56.2%。）

### 16.2 结果：小 draft 在 WikiText 上是“域偏移”，不是小扰动

| draft | NLL | ppl | excess NLL | next-tok top-1 | **teacher-agree** | top-5 | KL_TD | KL_DT | TV | JS | rank | ent | MATH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| teacher（上界） | 2.000 | 7.39 | 0 | 56.2% | **100%** | 100% | 0 | 0 | 0 | 0 | 0 | 1.75 | 100% |
| dense-distill | 5.943 | 381 | 3.943 | 23.8% | **30.0%** | 49.6% | 3.948 | 3.443 | 0.682 | 0.393 | 664 | 2.45 | 85.5% |
| dense-RL | 5.155 | 173 | 3.155 | 25.2% | **32.3%** | 52.9% | 3.219 | 3.148 | 0.669 | 0.375 | 415 | 3.77 | 87.0% |
| dense-QAT-W4 | 5.123 | 168 | 3.123 | 25.1% | **32.2%** | 52.2% | 3.195 | 3.171 | 0.678 | 0.379 | 435 | 4.09 | 86.6% |
| MoE8 | 4.807 | 122 | 2.807 | 26.6% | **34.4%** | 55.7% | 2.877 | 2.999 | 0.651 | 0.357 | 298 | 3.75 | 88.3% |
| MoE16 | 4.706 | 111 | 2.706 | 27.1% | **35.2%** | 56.9% | 2.779 | 2.965 | 0.647 | 0.353 | 270 | 3.78 | 88.1% |
| **MTP** | **2.740** | **15.5** | **0.740** | **43.3%** | **62.1%** | **90.6%** | **0.776** | **1.346** | **0.381** | **0.153** | **4.68** | **2.71** | **95.7%** |
| **teacher-QAT-W4** | **2.098** | **8.15** | **0.098** | **54.9%** | **83.3%** | **98.7%** | **0.134** | **0.136** | **0.143** | **0.029** | **1.76** | **1.72** | **97.0%** |

> 两个 top-1 指标要分开：
> `next-tok top-1` = draft argmax 命中 **真实下一个 token**；
> `teacher-agree` = draft argmax 命中 **teacher 的 top-1**，这才是投机解码接受率的直接对应量。
> teacher 自己的 next-tok top-1 是 56.2%，teacher-agree 按定义 100%。

三条结论：

1. **小 draft 在 WikiText 上几乎是“另一个模型”**：NLL 4.7~5.9（teacher 2.00），
   TV 0.65~0.68（TV 上限 1.0），next-tok top-1 只有 23.8%~27.1%，
   **teacher-agree 只有 30.0%~35.2%**，teacher top-1 在 draft 里的平均 rank 270~664。
   它们在 MATH-500 上的 85%~88% 接受率是**域内**的；换到通用文本，分布已经严重偏移。
2. **MTP 和 teacher QAT 才是“全局接近 teacher”的 draft**：MTP NLL 2.74、TV 0.381、
   teacher-agree 62.1%；teacher QAT NLL 2.10、TV 0.143、teacher-agree 83.3%。
   teacher QAT 的 excess NLL 只有 **0.098 nats**，TV **0.143**——量化后的原模型仍然是最忠实的 draft。
3. **excess NLL ≈ forward KL**：例如 dense-distill 3.943 vs 3.948、MoE8 2.807 vs 2.877、
   MTP 0.740 vs 0.776、QAT 0.098 vs 0.134。这正是信息恒等式
   \(\text{NLL}_D = H(P_T) + D_{KL}(P_T\|Q_D)\)，说明 NLL 和 KL 两个指标是自洽的。

**域内 vs 域外的 teacher-agree gap**（MATH-500 减 WikiText）：
dense 族 **54~56pt**、MoE8/16 **53~54pt**、MTP **34pt**、teacher QAT **14pt**。
小 draft 的“高接受率”几乎完全是数学域内的；MTP/QAT 的通用域保真度高得多。

### 16.3 模型级规律：接受率 ≈ 1 − a·TV ≈ 1 − b·√KL ≈ 1 − c·log NLL

把 7 个模型放在一起，WikiText 的分布偏移和 MATH-500 接受率高度一致：

- Pearson \(r(\text{TV}, \text{MATH})=-0.957\)，\(r(\text{KL}, \text{MATH})=-0.994\)，
  \(r(\text{NLL}, \text{MATH})=-0.992\)；teacher-agree 与 MATH 的相关是 \(r=0.967\)。
- 线性/平方根/对数拟合：
  \[
  \boxed{
  p_{\text{MATH}} \approx 1.014 - 0.212\,\text{TV}
  \quad(R^2=0.92)
  }
  \]
  \[
  \boxed{
  p_{\text{MATH}} \approx 1.008 - 0.076\sqrt{D_{KL}(P_T\|Q_D)}
  \quad(R^2=0.97)
  }
  \]
  \[
  p_{\text{MATH}} \approx 1.066 - 0.119\log \text{NLL}
  \quad(R^2=0.98).
  \]
- 这正好接上 §14 的微观模型：TV 是 \( \frac12\|\epsilon\|_1 \)，KL 是 \( \frac12\|\epsilon\|_2^2 \)；
  若接受率损失 \(\propto\) TV，就有 \(p\approx 1-a\,\text{TV}\approx 1-b\sqrt{KL}\)。
  **模型级的“分布微扰 → 接受率”是近似线性的（对 TV），对 KL 是平方根律。**
- 注意：per-token 的 TV→match 关系更噪声（单模型 \(R^2=0.18\sim0.48\)），
  因为每个 token 的 top-1 margin/熵不同；干净的线性律是**模型级平均**。

![WikiText shift](out/fig_wikitext_shift.png)

### 16.4 per-token：TV 高处的 token 几乎全错

把每个模型的 token 按**自己的 TV 分位**分 10 个 bin，看 **next-token top-1**
（teacher-agree 的 per-token 趋势相同，只是整体高一些）。注意 bin 边界因模型而异，
这里看的是“同一模型内 TV 升高时 match 怎么掉”的趋势，不是同一 TV 区间的横向比较：

| draft | 最低 TV bin | 第 2 bin | 中间 | 最高 TV bin |
|---|---|---|---|---|
| dense-distill | 81% | 29% | 7% / 2% | 0% |
| dense-RL | 80% | 30% | 12% / 4% | 1% |
| MoE8 | 77% | 34% | 15% / 6% | 1% |
| MTP | 85% | 48% | 36% / 32% | 15% |
| teacher-QAT-W4 | 98% | 73% | 44% / 31% | 28% |

- 所有模型都是 **TV 越高、match 越低**，但曲线的截距/斜率不同：
  小 draft 在低 TV bin 也只有 ~77~81%，高 TV bin 直接 0~1%；
  teacher QAT 低 TV bin 98%、高 TV bin 仍有 28%。
- 这说明小 draft 的 WikiText 序列是 **“域内 token + 大量 OOD token”的混合**，
  而不是均匀的小扰动。这个混合正是 §14 里低 β（0.72~0.77）的来源：
  难度异质性大，长 runs 很快掉到 OOD 区。
- teacher QAT 的 TV 分布集中在 0 附近、且高 TV 处仍有 28% match，
  所以它的 β=0.91、长 γ 更接近无记忆指数；MTP 居中（β=0.846）。

### 16.5  entropy / 校准：小 draft 不是“过度自信”，而是“不确定 + 错”

- teacher entropy 1.75；teacher QAT 1.72（几乎不变）；MTP 2.71；
  小 draft 2.45~4.09。
- dense-RL / MoE8 / QAT-W4 的 entropy 3.75~4.09，说明它们在 WikiText 上**很不确定**，
  但仍然错（NLL 4.8~5.2）。这是典型的 OOD 行为：分布被摊平，top-1 不可靠。
- dense-distill 是例外：entropy 2.45（相对低）但 NLL 5.94（最高），
  即它在部分 token 上**自信地错**——和 §9.3 的 silent error 一致。
- 所以“小 draft 的失败”不是简单的温度/校准问题；它的分布形状已经变了（TV 0.7），
  单靠 temperature 救不回来（§9.3 已经验证）。

### 16.6 对 §12~§15 的统一解释

1. **小 draft 的 MATH 接受率高，是域内特化**：dense/MoE 在 MATH-500 上 85%~88%，
   但 WikiText NLL 4.7~5.9、TV 0.65~0.68、next-tok 24%~27%（teacher-agree 30%~35%）。它们学的是“teacher 在数学上的行为”，
   不是“teacher 的通用分布”。
2. **teacher QAT / MTP 的优势是全局的**：WikiText NLL 2.10/2.74、TV 0.143/0.381，
   所以在长 γ 上不仅数学强，通用文本也更稳。**大 draft 相对 MTP 的长 γ 优势，
   本质上是“全局分布更接近 teacher”**，而不只是参数量。
3. **动态深度的收益与 TV 异质性一致**：小 draft 的 TV 分布从 0 到 1 很宽，
   confidence 能分出“域内/域外”，所以 cluster 动态 depth 有 +5~6% 可拿；
   teacher QAT 的 TV 集中在 0 附近，fixed γ 已接近最优，所以只有 +1.5%。
4. **β 的物理意义**：β 低 = TV/难度异质性大 = 序列里有大量 OOD/高 TV token。
   小 draft 的 β=0.72~0.77，teacher QAT 的 β=0.91，正好对应它们的 TV 分布宽度。
5. **W3 悬崖的解释**：W3 PTQ 的 NLL/TV 会进一步恶化（未逐项测，但 τ(32) 从 19 掉到 9.4
   已经说明），分布偏移变大，接受率掉得比 W4 快；所以 W4 是“全局接近 teacher”的最低成本点。

### 16.7 局限

- WikiText-2 raw test 只取了前 102k token（完整 test 约 245k），不是全量；
  但主题覆盖已经比 demo 广，模型间排序与 §12~§15 完全一致。
- teacher top-256 + residual bucket：平均 residual 1.66%，高熵 token 上 KL/TV 是近似；
  top-1/NLL 是精确的。
- MATH 接受率来自 §12/§13 的 500/100 题评测，与 WikiText 不是同一数据；
  模型级拟合有 7 个点，相关很强但不是普适定律。
- 小 draft 本来就是在**chat 格式的数学数据**上蒸馏的，WikiText 是 raw text，
  同时有 domain shift 和 format shift；teacher/MTP/QAT 是通用的，所以这个 gap 是
  “域偏移”的上界，不完全是 FFN 近似误差。尽管如此，模型间排序与 §12~§15 完全一致。

### 16.8 复现

```bash
# 1) teacher 一次前向，缓存 top-256 / hidden / NLL
python code/teacher_wikitext.py --n 20 --seq_len 1024 --topk 256

# 2) 各 draft（dense / MoE / QAT）
python code/draft_wikitext.py --ckpt ckpt/dense_rl --tag dense_rl
python code/draft_wikitext.py --ckpt ckpt/moe8_keep8_distill --tag moe8_keep8_distill
python code/draft_wikitext.py --teacher_qat ckpt/teacher_qat_w4_100.pt --tag teacher_qat

# 3) MTP
python code/mtp_wikitext.py

# 4) 汇总 + 图
python code/analyze_wikitext.py
```

---

## 17. OOD token 的灾难性：\(\tau_\infty\approx1/q\)，以及“超大 draft”为什么救不了 OOD

> 问题：少量 OOD token 会不会对投机解码产生灾难性影响？
> 如果会，能不能直接证伪“把 draft 做得更大”这条路？

### 17.1 定义与实验

- 定义 **OOD / disagreement rate**
  \[
  q = 1 - \text{teacher-agree}
  = \Pr[\arg\max Q_D \ne \arg\max P_T].
  \]
- 用 WikiText-2 raw test 的 teacher-agree 序列直接做投机解码模拟，得到 \(S(k)\)、\(\tau(\gamma)\)、\(\tau_\infty\)。
- 合成注入：在 MATH-500 的 match 序列里，按比例 \(q\) 把 token 改成 miss
  （i.i.d.，以及 runs=3/5/10 的成簇 OOD），再模拟 \(\tau(32)\)。
- 模型：dense-distill / dense-RL / dense-QAT / MoE8 / MoE16 / MTP / teacher-QAT-W4。

### 17.2 主结果：\(\tau_\infty \approx 1/q\)（R² = 1.000000）

| draft | q（WikiText） | τ∞（γ=64） | 1/q | MATH τ(32) |
|---|---|---|---|---|
| dense-distill | 0.700 | 1.43 | 1.43 | 6.57 |
| dense-RL | 0.677 | 1.48 | 1.48 | 7.24 |
| dense-QAT-W4 | 0.678 | 1.47 | 1.47 | 7.08 |
| MoE8 | 0.656 | 1.53 | 1.52 | 7.96 |
| MoE16 | 0.648 | 1.54 | 1.54 | 7.83 |
| MTP | 0.379 | 2.63 | 2.64 | 16.59 |
| teacher-QAT-W4 | 0.167 | 5.97 | 5.99 | 19.95 |

拟合：
\[
\boxed{
\frac{1}{\tau_\infty} = 0.00087 + 0.9989\,q
\quad(R^2=1.000000)
}
\]
等价于
\[
\boxed{
\tau_\infty \approx \frac{1}{q+\epsilon},\qquad \epsilon\approx0.0009
}
\]
其中 \(p_I=1-\epsilon\approx99.91\%\) 是 in-domain token 的接受率。

**解释**：draft 在 in-domain token 上几乎总是对的（\(p_I\approx0.999\)），
长 γ 的接受长度完全由 **OOD/不一致率 q** 决定。平均 top-1 准确率 \(p=1-q\) 只是 q 的补；
long-γ 的 tail 是 \(S(k)\approx(1-q)^k\)，所以 \(\tau_\infty\approx1/q\)。

注意：**TV 小不等于 q 小**。teacher QAT 的 TV 只有 0.143，但 q=16.7%——
WikiText 上很多 token 的 top-1 margin 很小，微小的分布偏移也会翻转 argmax。
long-γ 的 master variable 是 q（argmax 翻转率），不是 TV/KL 本身；
TV/KL 只是 q 的平滑 proxy。

**同一个 law 在 MATH 上也成立**：MATH 上 teacher-agree 就是 1-q，
所以 \(1/q\) 给出长 γ 的上界，τ(32) 只是还没饱和的值：
dense-RL q≈13% → \(1/q=7.7\)，实测 τ(32)=7.24；
MoE8 q≈12% → 8.6 vs 7.96；
MTP q≈4.3% → 23.3 vs 16.59；
teacher QAT q≈3% → 33.3 vs 19.95。
**τ(32) ≤ τ∞ = 1/q 在所有模型/域上都成立**，差别只是 γ=32 是否已经饱和。

#### 17.2.1 为什么是 \(1/q\)：一个 renewal 推导（不要求 i.i.d.）

把每个位置的接受指示记成
\[
A_t = 1\{\arg\max Q_t = \arg\max P_t\}.
\]
在投机解码里，一个 block 总是从**上一次拒绝之后的 token** 开始（或序列开头）。
令
- \(R\)：连续接受 run 的长度；
- \(F\)：连续拒绝 run 的长度；
- \(p=\Pr[A_t=1]=1-q\)。

对平稳过程，block 起点是 failure-renewal point，所以：

1. 下一个 token 开始一个 success run 的概率是 \(1/\mathbb{E}[F]\)；
2. 给定 success run 开始，其期望长度是 \(\mathbb{E}[R]\)；
3. 因此每个 block 的期望接受草稿数
   \[
   \mathbb{E}[L] = \frac{\mathbb{E}[R]}{\mathbb{E}[F]}.
   \]
4. 平稳性给出
   \[
   p = \frac{\mathbb{E}[R]}{\mathbb{E}[R]+\mathbb{E}[F]}
   \quad\Longrightarrow\quad
   \frac{\mathbb{E}[R]}{\mathbb{E}[F]} = \frac{p}{1-p}.
   \]
5. 加上 block 末尾 teacher 补的 1 个 bonus token：
   \[
   \boxed{
   \tau_\infty = 1+\mathbb{E}[L]
   = 1+\frac{p}{1-p}
   = \frac{1}{1-p}
   = \frac{1}{q}.
   }
   \]

**关键点：这个推导不要求 i.i.d.**。clustering（\(P(C|C)\)、\(P(C|I)\)）会改变
\(\mathbb{E}[R]\) 和 \(\mathbb{E}[F]\)，但它们的比值被 \(p\) 固定；所以 **mean \(\tau_\infty\) 不变**。
clustering 只影响有限 γ 的曲线形状/方差（§14 的 β），不影响 \(1/q\) 的极限。
这就是为什么我们在数据里看到强 clustering，但 7 个模型的 \(\tau_\infty\) 仍然精确落在 \(1/q\) 上。

数值验证（WikiText-2，直接从 run length 算）：

| model | p | E[R] | E[F] | 1+E[R]/E[F] | 1/q |
|---|---|---|---|---|---|
| dense-distill | 0.300 | 1.648 | 3.825 | 1.431 | 1.430 |
| dense-RL | 0.323 | 1.735 | 3.625 | 1.479 | 1.477 |
| dense-QAT-W4 | 0.322 | 1.721 | 3.613 | 1.476 | 1.475 |
| MoE8 | 0.344 | 1.754 | 3.330 | 1.527 | 1.525 |
| MoE16 | 0.352 | 1.772 | 3.259 | 1.544 | 1.542 |
| MTP | 0.621 | 2.732 | 1.668 | 2.637 | 2.636 |
| teacher-QAT-W4 | 0.833 | 6.182 | 1.239 | 5.988 | 5.992 |

误差都在 0.002 以内。这是一个 **renewal reward identity**，不是拟合巧合：
只要 block 从 failure-renewal point 开始、过程平稳，mean 就严格是 \(1/q\)。

**什么时候会偏离 \(1/q\)？**
- block 不是从 failure-renewal point 开始（例如从随机位置开始，会遇到 inspection paradox，
  期望 residual run 变长/变短）；
- 过程非平稳（难度沿文档漂移，q 不是常数；这时应分段用平均 q）；
- 长程相关让“失败后”进入一个持续很难的 regime（此时 block 起点不是 renewal point，
  \(1/q\) 变成近似，有限 γ 会更差）。
所以 \(1/q\) 是**理想 renewal 极限**；实测 7 个模型/两个域都落在 0.002 内，
说明对当前 draft 和文本，renewal 近似非常好。

### 17.3 少量 OOD 就是灾难：5% 就能砍掉一半长 γ

在 MATH-500 的 match 序列里注入 OOD（i.i.d.，run=1），\(\tau(32)\)：

| 注入 q | dense-RL（base 8.85） | MoE8（base 9.23） | MTP（base 16.59） | teacher QAT（base 19.95） |
|---|---|---|---|---|
| 0% | 8.85 | 9.23 | 16.59 | 19.95 |
| 1% | 8.3（-6%） | 8.7（-6%） | 14.9（-10%） | 17.7（-11%） |
| 2% | 7.8（-12%） | 8.1（-12%） | 13.4（-19%） | 15.8（-21%） |
| **5%** | 6.6（**-25%**） | 6.8（**-26%**） | 10.4（**-37%**） | 11.7（**-41%**） |
| **10%** | 5.2（**-41%**） | 5.3（**-43%**） | 7.3（**-56%**） | 8.0（**-60%**） |
| **20%** | 3.7（**-58%**） | 3.8（**-59%**） | 4.6（**-72%**） | 4.8（**-76%**） |

- **5% OOD 就把 \(\tau(32)\) 砍掉 25%~41%；10% 砍掉 41%~60%。**
  成簇注入（runs=3/5/10）比 i.i.d. 再差 0~2 个点，但量级相同。
- 这正是 \(\tau_\infty=1/(q+\epsilon)\) 的直接后果：
  \(q=5\%\to\tau_\infty\approx20\)，\(q=10\%\to10\)，\(q=17\%\to6\)，\(q=20\%\to5\)。
- 注意：base τ 越高的 draft，**相对损失越大**（QAT -41% vs dense -25% at q=5%），
  因为它的 in-domain 部分本来可以跑很长；OOD 把长 block 直接截断。
  base τ 低的 draft 已经接近 1/q，OOD 的边际伤害反而小。

### 17.4 扩大 draft 参数量的收益：tail-dominated scaling

\[
\tau_\infty(q)=\frac{1}{q+\epsilon},\qquad
\frac{\partial \tau_\infty}{\partial q}=-\frac{1}{(q+\epsilon)^2}.
\]
**越往后，每减少一个 OOD token 的价值越大**：

- q: 0.70 → 0.17（3.4B → 35B QAT）：τ∞ 1.5 → 6.0，**+4.5（4×）**
- q: 0.17 → 0.05：τ∞ 6.0 → 19.7，**+13.7（3.3×）**
- q: 0.05 → 0.01：τ∞ 19.7 → 92，**+72（4.7×）**

也就是说：
- **3.4B → 35B 只把 q 从 ~0.68 降到 ~0.17**（WikiText OOD），τ∞ 从 1.5 到 6；
  这是一个真实的 4× 提升，但**离长 γ 需要的 τ∞>20 还差 3×**。
- 要 \(\tau_\infty>20\) 需要 \(q<5\%\)；要 \(>30\) 需要 \(q<3.3\%\)。
  35B QAT 在 WikiText 上 q=16.7%，**远没有跨过 5% 的门槛**。
- **in-domain（MATH）**：小 draft 的 teacher-agree 是 85~88%（q≈12~15%），
  \(\tau_\infty\approx1/q\approx7\sim8\)；MTP q≈4.3% → 23；
  teacher QAT q≈3% → 33。**域内加大 draft 确实有价值**（τ∞ 7.7→33），
  但要付 35B 的成本；对小 draft 来说 τ∞≈7~8 已经能跑中等 γ。
- 大 draft 的价值在域内是真实的；但在 OOD 文本上，q 只能从 0.68 降到 0.17，
  τ∞ 仍被 residual OOD 限制在 ~6，远低于 in-domain 的 20~33。
- **MTP 是一个关键反例**：它只有 ~0.8B 总参数，但它是 generalist——
  WikiText q=0.379（远好于 3.4B math-specialist 的 0.68），MATH q=0.043。
  这说明降低 q 的关键是**数据覆盖/泛化**，不是纯参数量；
  一个小的 generalist 可以在 OOD 上打赢一个大的 specialist。
  所以“超大 draft”不是唯一的路，甚至不是最好的路。

**这就是“超大 draft”路径的墙**：
- 把 draft 从 3.4B 扩到 35B，能把 OOD 率从 0.68 降到 0.17，但**最后那 0.17 的 OOD
  是容量/量化最难消掉的**；它把长 γ 的 τ∞ 死死压在 ~6。
- 想靠“继续变大”把 q 从 0.17 再降到 0.05，需要的不只是 4× 参数，
  而是**对部署分布的覆盖/泛化**；单纯 scaling 不会自动买到。
- 所以 **naive “just make the draft bigger” 在 OOD/通用文本上被证伪**：
  它付出的显存/带宽是 35B 级，但 long-γ 仍被 residual q 卡在 ~6，
  甚至不如 in-domain 的 3.4B 小 draft（q≈13%，τ∞≈7.7）。
- 这不否定“大 draft 在 in-domain 有用”；它否定的是“大 draft 能解决 OOD”。

### 17.5 对前面所有结果的统一

- **§14 深度律**：\(\beta\) 是 q 的异质性；长 γ 的极限就是 \(\tau_\infty=1/q\)。
- **§15 动态深度**：confidence gating 能在 OOD token 上省草稿算力，但**不能降低 q**；
  它只能减轻成本，不能修复 tail。
- **§16 分布偏移**：WikiText 的 TV/KL/NLL 本质上是 q 的 proxy；
  模型级 \(p\approx1-a\text{TV}\) 是 q 的线性近似，long-γ 的精确形式是 \(1/q\)。
- **§13 teacher QAT**：MATH q≈3% → τ∞≈33（τ(32)=19.95 是还没饱和）；
  WikiText q≈17% → τ∞≈6。这解释了“同一个 draft 在数学上很强、在通用文本上长 γ 很弱”。
- **W3 悬崖**：W3 PTQ 把 q 进一步推高，τ∞ 掉得更快；W4 是 q 的最低成本点。

### 17.6 结论（可写进论文的那句）

> **长 γ 投机解码的接受长度不是由平均准确率线性决定的，而是由 OOD/不一致率 q 的倒数决定：
> \(\tau_\infty\approx1/q\)。5% 的 OOD 就能把 \(\tau(32)\) 砍掉 25%~41%，
> 10% 砍掉 41%~60%。在 **in-domain** 数学上，加大 draft 确实能把 q 从 ~13% 降到 ~3%、
> \(\tau_\infty\) 从 ~7.7 提到 ~33；但在 **OOD/通用文本** 上，把 draft 从 3.4B 扩到 35B
> 只能把 q 从 ~0.68 降到 ~0.17，\(\tau_\infty\) 从 1.5 到 6——要跨过长 γ 的门槛需要 q<5%，
> 而 35B QAT 仍停在 17%。所以在 OOD 上，“继续加大 draft”被 residual OOD 卡死；
> 大 draft 的收益是**域内且 tail-dominated** 的，它不能解决 OOD。**

### 17.7 局限

- q 来自 teacher-agree on WikiText-2 raw test；对数学蒸馏的小 draft 是极端 OOD，
  真实部署分布的 q 可能介于 MATH 和 WikiText 之间。
- \(\tau_\infty=1/q\) 假设 in-domain \(p_I\approx0.999\) 且 OOD 近似 i.i.d.；
  实测 7 个模型 R²=1.000000，但仍是单一数据集；成簇 OOD 会让有限 γ 更差。
- \(\tau_\infty\) 是 γ→∞；实际 γ=8~32 的 τ 更低（§17.3 的注入表）。
- 合成注入是随机/成簇 OOD，不等价于真实 OOD token 的语义结构；
  但趋势和真实 WikiText 的 q→τ∞ 关系一致。
- teacher QAT 是 W4；W8/全精度可能把 q 降到 0.1 以下，但显存/带宽代价更高；
  “scaling 墙”是 W4 的成本-精度点，不是所有可能的 QAT。

### 17.8 复现

```bash
# 依赖 §16 的 wikitext_*.npz 和 MATH 的 features/composite npz
python code/ood_catastrophe.py
# 输出 out/ood_catastrophe.json + out/fig_ood_catastrophe.png
```

---

## 18. 结论清单（checklist）

> 用来核对“所有结论是否都写进文档”。每条给出结论、位置和关键证据。

| # | 结论 | 位置 | 关键证据 |
|---|---|---|---|
| 1 | 同激活参数的 dense draft 可行：token top-1 85.5%，τ(5)=4.00、τ(8)=4.87；纯蒸馏已拿到大部分收益 | §1–§4 | MATH-500 500 题 |
| 2 | 端到端速度 ≠ 接受长度：本机 HF 实现下 dense draft 在任何并发都打不过原生 MTP；只有带宽受限 c≤0.10~0.17 才可能赢 | §5–§6 | 并发 bench |
| 3 | 失败不是“路由到冷门专家”：路由偏差与 mismatch 负相关（AUC 0.46）；真正信号是 draft 自身熵（AUC 0.92） | §7、§9.1 | route_analysis / branch |
| 4 | 要涨长 γ 接受长度必须多候选/树：teacher 真 token 97.9% 在 draft top-4；k=4 理想 τ(8)≈8.3 | §7 | topk_recall |
| 5 | 原生 MTP 用持久 cache 后很强：token 准确率 95.7%，τ(5)=5.32、τ(8)=7.44；dense draft + MTP 复合不划算（MTP 严格支配） | §9.2 | mtp_eval / composite |
| 6 | W4 QAT dense draft 几乎无损：PTQ token 85.15%（-0.35pt），同预算 bf16 86.76% vs QAT 86.63%（-0.13pt）；体积 6.87→2.49GB | §12 | qat / controls |
| 7 | MoE8（256→8，top-8 全激活，激活参数不变）在同总参数/同预算下反超 dense：+2.67pt token、τ(32) +19.7%；E=16 没有更好 | §12 | dense-ctrl / MoE8 |
| 8 | 原模型 W4 QAT：PTQ token 96.06%、τ(32)=17.53；部分 QAT 96.97%、τ(32)=19.95；W3 PTQ 崩到 90.19%、τ(32)=9.36 | §13、§15.3 | teacher PTQ/QAT |
| 9 | 深度律：短 γ 线性斜率精确等于 \(-p(1-p)/2\)（6 模型误差<0.002）；全局凹；条件接受率 \(p_k\) 随深度上升；长 runs 是 stretched-exp，β=0.72~0.91 | §14 | depth_theory |
| 10 | 动态深度（confidence/entropy 聚类）：MoE8 +5.6%、dense +5.1%、MTP +2.5%、teacher QAT 只有 +1.5%；2D 加 teacher entropy 无额外收益 | §15 | 5-fold CV |
| 11 | 大 draft vs MTP：τ 优势随 γ 增大（γ=8 1.06→γ=32 1.20）；等 c 下 teacher QAT 赢；W4 实际 c=0.268 vs MTP 0.03，绝对 speedup MTP 8.5× vs 2.5× | §15 | cost sweep |
| 12 | WikiText-2：小 draft NLL 4.7~5.9、teacher-agree 30~35%、TV 0.65~0.68；MTP 2.74/62.1%/0.381；teacher QAT 2.10/83.3%/0.143 | §16 | 102k token |
| 13 | 模型级分布偏移律：\(p_{\text{MATH}}\approx1.014-0.212\text{TV}\approx1.008-0.076\sqrt{KL}\approx1.066-0.119\log\text{NLL}\)（R²=0.92/0.97/0.98） | §16.3 | 7 模型 |
| 14 | OOD 灾难：\(\tau_\infty\approx1/q\)，7 模型拟合 R²=1.000000；5% OOD 把 τ(32) 砍 25%~41%，10% 砍 41%~60% | §17 | WikiText + 注入 |
| 15 | Scaling wall：3.4B→35B 只把 OOD 文本 q 从 ~0.68 降到 ~0.17，τ∞ 1.5→6；要 τ∞>20 需 q<5%；35B QAT 仍在 17% | §17.4 | τ∞=1/q |
| 16 | 结论：naive“超大 draft”在 OOD 上被 residual q 卡死；域内大 draft 仍有用（q 13%→3%，τ∞ 7.7→33）；MTP 用 0.8B generalist 在 OOD 上打赢 3.4B math-specialist（q 0.38 vs 0.68） | §17.4–§17.6 | 跨域对比 |
| 17 | \(1/q\) 的推导：renewal identity，\(\mathbb{E}[L]=\mathbb{E}[R]/\mathbb{E}[F]\)，\(p=\mathbb{E}[R]/(\mathbb{E}[R]+\mathbb{E}[F])\) → \(\tau_\infty=1/q\)；不要求 i.i.d.，clustering 只影响有限 γ | **§17.2.1** | run-length 数值验证 |

**没有记录进文档的结论？** 目前没有。§1–§18 覆盖了：
模型构造/训练、接受率/长 γ、端到端速度、路由/分支/树、复合、dense QAT、MoE8/16、
原模型 QAT、深度律、动态深度、WikiText 分布偏移、OOD 灾难与 \(1/q\) 推导。
唯一还在“未来工作”的是：full 35B QAT（FSDP/ZeRO）、W2/W3 QAT、真实 speculative loop 的长 γ 验证。
