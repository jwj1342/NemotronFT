# 路线图（Roadmap）

> **当前版本：2026-06-02。** 本文档取代了上一版「必须做 RL → LB 0.88-0.90」的战略论证——
> 那个赌注已被 GRPO 实验证伪（策略熵塌缩，无梯度），见 `ITERATION_REPORT.md` §6。
> 本轮（06-02）进一步证伪了「手写求解器生成 CoT → SFT」整条路线，并通过文献调研
> 确立了新的候选方向。**旧的 RL 战略论证移到本文末尾 §8 作为历史附录。**

> 基线：`output/huikang-baseline-converted/`（公开 LB **0.85**，9500 自评 **87.79%**）。
> grader-aligned 推理：T=0, max_tokens=7680, top_p=1.0, max_model_len=8192。

---

## 0″. 🏁 最终结论（2026-06-03 bit RFT pilot — 当前最权威，修正下面 §0′ 的乐观）

§0′ 的 pass@k 翻案（bit "可救 70%、+0.2~0.7pp、仅 bit 就够 0.86"）经 RFT pilot **实证修正为
净增 ~0**。bit RFT pilot（PROBE 262 正确 CoT → SFT → T=0 holdout 三指标，baseline holdout acc 0.71）：

| checkpoint | bit 恢复 | bit 回归 | sentinel 漂移 | 净增（holdout 子集）|
|---|---|---|---|---|
| checkpoint-5（甜点, ~0.4ep） | 5 | 2 | 0 | +3 |
| checkpoint-10 / 15 | 8 / 9 | 11 / 11 | 2 / **15(崩)** | −5 / −17 |

**外推全集**：恢复 8.3%×200≈+17，回归 1.4%×**1402**≈−20 → **净 ≈ 0 或微负**。不值得提交。

**为什么 pass@k 70% 没兑现**：bit 失败题与已对题**共享同一套歧义位先验**，RFT 拉失败题 → 回归
已对题（**先验耦合**）。已对基数大（1402），1.4% 回归的绝对数（20）吃掉 8.3% 恢复（17）。这是
结构性的，非调参可解。长 CoT 专项 SFT 过拟合极快（甜点 ~0.4 epoch，之后非 bit sentinel 崩盘）。

**最终判断：0.85 是「Huikang baseline + LoRA + 固定单次评分」框架的天花板。** "救失败题"三条
候选（SFT 蒸馏 / 手写求解器 / RFT 自训练）全部验证净增 ≤0。+1pp 需换框架——更强 baseline
（Huikang 0.85 已是公开最强，无来源）或推理时策略（已确认关闭，§5）。

**RL 是否值得回头试**：pass@k 显示 bit 在 T>0 有梯度（pass@1=20%，不像 cryptarithm 熵塌缩），
故 GRPO 在 bit 上**技术可行**（修正 §8/§6 对 cryptarithm 的"RL 不能启动"）。但 RL 面临与 RFT
**同样的先验耦合限制**；KL anchor 或许能比 MLE-SFT 更温和地控回归——这是唯一没试过的角度，但
是未验证的希望，GRPO 工程成本远高于 RFT，而 RFT 已用更便宜方式探明天花板 ~0。详见
`ITERATION_REPORT.md` §8。

> 下面 §0′（pass@k 翻案）、§0（早间诊断）保留，作完整「诊断 → pass@k → pilot」摆动链的方法论
> 记录——每一层都被下一层的更强证据修正，核心教训是 **可救性/天花板只能靠 holdout 净收益外推
> 全集来定，静态求解器估计、pass@k 上界、holdout 子集 acc 都会系统性误导**。

---

## 0′. ⚡ pass@k 实证翻案（2026-06-02 晚 — 已被上方 §0″ 的 pilot 实证修正）

跑了 pass@k 探针（Huikang adapter 对失败题 T=0.9 采样 n=16，每类 50 题）。结果**推翻了 §0 基于
静态求解器的"bit 不可救 / +0.3pp 上限"结论**：

| 类别 | pass@16 | pass@1 | 含义 |
|---|---|---|---|
| **bit_manipulation** | **0.70** | 0.207 | 35/50 失败题能采到正确答案 → **能力受限，RFT 可挖** |
| equation_numeric_guess | 0.00 | 0 | 采 16 次一次不对 → **死类确认** |
| cryptarithm_deduce | 0.00 | 0 | 同上 |
| cryptarithm_guess | 0.00 | 0 | 同上 |

**bit 翻案**：失败题里 pass@16=70%，正确 CoT 是真实 bit-serial 推理（pass@1=20.7% 远高于瞎蒙
1/256）、26/35 唯一答案 well-posed、中位 6618 token 在 budget 内、20/35 求解器也解对。说明 bit 失败
大部分是**能力受限**（T=0 贪心选错歧义位，T>0 能算对）——之前静态求解器估的"净救+3/欠约束88%"
**低估一个数量级**（求解器看不到模型 T>0 能力）。

**Codex 复核结论**：bit RFT 是真实机会，潜在 **+20~70 题（+0.2~0.7pp），仅 bit 一类可能就够冲 0.86**。
但 pass@16=70% 是 **oracle 上界**，RFT 实际收割 20-50% 未验证；**核心风险=回归**（bit 已对 1402 题，
回归 3% 掉 42 题可吃掉收益）。

**受控执行方案（Codex）**：① 对全 200 bit 失败题采样确认 pass@k 稳定 + 收集正确 CoT；② 保存已对 bit
题 T=0 CoT 作 replay + 回归监控；③ 按 prompt 切 train/holdout（同题不跨集）；④ 短 pilot：residual/
second LoRA、低 LR、少步、频繁 checkpoint；⑤ 每 checkpoint 用 **T=0** 跑三指标（失败恢复 / 已对回归 /
非 bit sentinel 回归）；⑥ 净收益显著正才批 12h 全量。按 token（非样本）平衡，长度过滤优先非截断短 CoT。

**方法论铁律**：可救性必须用模型自己的 **pass@k 实证**，绝不能用更弱的静态求解器估计——后者在 bit 上
错了一个数量级。

**修正后的活路排序**：bit RFT（唯一有实证支撑、可能单独够 0.86）＞ 低风险格式修复（+0.1~0.2pp）＞
其余类（pass@16=0，死）。cryptarithm/numeric_guess 从训练目标剔除。

---

## 0. ⚠️ 早间硬结论（2026-06-02 多 agent 诊断 + 独立对抗验证 + grader 机制核实）

> **注**：下面 (2)(3) 关于 bit "不可救 / +0.3pp 上限"的部分已被上方 §0′ 的 pass@k 实证**推翻**。
> 保留作为"静态求解器会严重低估可救空间"的方法论记录。(1) grader 关闭、(4) predicted 污染仍成立。

**(1) grader 机制已确认——推理时杠杆确凿关闭。** Kaggle 官方提交格式：提交物是 LoRA
adapter（rank≤32，submission.zip），主办方**固定 harness** 加载、对每题生成**单个** response、
提取 \boxed{} 答案。参赛者无法在评分时插入多采样 / best-of-N / self-consistency / TTT。
CLAUDE.md 记的"T=0 单次"由官方提交格式佐证。→ 本文档之前列为"最高未知"的 grader 多采样杠杆
= **关闭**；路线 C（TTT / 多采样）正式除名，不再是 +1pp 的门。

**(2) 救失败题的数学上限 ≈ +0.30pp，到不了 0.86。** 三类失败题逐题诊断 + 独立对抗复算：
诊断给的"可救%"(bit 12 / numeric 12 / cryptarithm 14)经验证**全部高估 2–8 倍**（混淆了
"原则上可救 recoverable-in-principle"与"无副作用净增 net-extractable"）。真实无副作用净救：
bit ~3 / numeric ~6–12 / cryptarithm ~4 = **全场合计 12–25 题 = +0.13~0.26pp**（乐观上界 ~+0.30pp）。
**没有任何单类、也没有三类之和能贡献 +1pp。**

**(3) 失败本质 = 信息论欠约束（方向确认，但比"墙"更细致）。** bit：0/200 题全 8 位唯一确定，
fan-in≤3 全函数空间唯一率仅 2.9%；用静态先验整体替换模型 = 灾难（FIX 58 / BREAK 500 / 净 −442，
求解器 59.9% vs 模型 87.5%）。cryptarithm_guess + eq_numeric_guess 主体信息论死（每题 per-op
独立随机规则，truth 在任意 combo 下 0/120 可达）。**但 Codex 的修正成立**：部分"形式上欠约束"
的题被模型的隐式先验救了回来（歧义位先验命中 81.3%），所以"不可救"不是绝对，只是**可干净提取的
边际空间极小**。

**(4) 数据可信度警告：`predicted` 字段被污染。** eval shard 里 1402 道 grader-correct 题中有
178 道 `predicted != answer` → 该字段不是被评分的最终串（疑似 CoT 末尾的解析噪声）。凡用 predicted
算的 Hamming / near-miss / BREAK 都被高估。**可信标签只有 `correct`(布尔) 和 `answer`(真值)。**

**总含义**：在「Huikang baseline + LoRA 微调 + 固定单次评分」框架内，0.85→0.86 **数学上做不到**
（最多到 ~0.852）。真正 +1pp 需要换框架——更强 baseline（但 Huikang 0.85 已是公开最强 adapter，
我们无更强来源）或推理时策略（已确认关闭）。**现实目标：蚕食式 +0.1~0.3pp + 把 RFT/合成数据
pipeline 跑通（方法论价值）。** 详见 §3 修正后的方法分流与 §4 pass@k 探针（进行中）。

---

## 1. 当前逐类成绩（实测 9500，仍准确）

| Category | Correct / Total | Acc | 评级 |
|---|---|---|---|
| numeral | 1576 / 1576 | **100.00%** | 满分 |
| unit_conversion | 1591 / 1594 | **99.81%** | 接近满分 |
| gravity | 1591 / 1597 | **99.62%** | 接近满分 |
| cipher（字母替换） | 1560 / 1576 | **98.98%** | 接近满分 |
| equation_numeric_deduce | 542 / 596 | **90.94%** | 良好 |
| bit_manipulation | 1402 / 1602 | **87.52%** | 良好但卡 |
| equation_numeric_guess | 22 / 136 | **16.18%** | ❌ 悬崖 |
| cryptarithm_deduce | 46 / 659 | **6.98%** | ❌ 悬崖 |
| cryptarithm_guess | 10 / 164 | **6.10%** | ❌ 悬崖 |
| **总计** | **8340 / 9500** | **87.79%** | |

到 0.86 需要整体 +1.0pp。LB 公开最高 0.87；0.90+ 从未出现。

---

## 2. 本轮（06-02）确立的核心认知

### 2.1 「手写求解器生成 CoT → SFT」是死路（已量化证伪）

实现了基于 Donald Galliano 组合框架（Kaggle 讨论 #684283）的 cipher-digit / symbol-digit
求解器，并逐类别对比了**求解器准确率 vs 模型准确率**（用
`output/eval_full_huikang-baseline-converted_shard*.jsonl` 的 `correct` 字段做交叉表）：

| 类别 | 手写求解器 | 模型 | 用求解器输出 SFT 的净效果 |
|---|---|---|---|
| bit_manipulation | 59.9% | 87.5% | 能教 +58 / **会教错 −500** |
| equation_numeric_deduce | 39.8% | 90.9% | 能教 +2 / **会教错 −307** |
| equation_numeric_guess | 0% | 16.2% | 0 / −22 |
| cipher-digit | ~4% | 7% | — |

**结论**：目标模型（Nemotron + Huikang SFT）已经比我们能快速手写的任何符号求解器都强。
用更弱的求解器输出去训更强的模型，必然负迁移。这统一解释了 opd-v1/v2 停在 0.85
（`ITERATION_REPORT.md` §3）和 cryptarithm canary 负迁移的根因。

> **求解器的正确角色不是生成器，而是验证器/过滤器**——见 §3 路线 A。

### 2.2 失败的本质是欠约束（信息论层面），不是可修的 bug

- **截断只占失败的 3%**（num_tokens≥7600）。绝大多数失败是「完整但错误」的推理。
- **bit 失败题 154/200（77%）只差 1-2 个 bit**：模型 8 位里对 6-7 位，在欠约束的「歧义位」
  上按学到的频率先验猜，恰好那一两位生成器用了罕见 gate。bit 纯演绎「全 8 位唯一确定」
  的题只占 2%（多个 gate 都与样例一致，无法判定 target）。
- **cipher-digit / cryptarithm 91% 欠约束**：用答案引导的求解器能解 42%（319/823），但
  忠实性检查显示只有 12%（符号覆盖）/ 4%（纯样例双求解器一致）真正可从样例推出；
  其中 16 题纯样例求解器解出**不同**答案——证明这些题多解。

文献直接印证（[2509.01016](https://arxiv.org/html/2509.01016)）：通过少量样例是真规则归纳的
**弱信号**——45.4% 通过 held-out 样例的程序仍违反真实规则。这就是模型「靠频率先验在欠约束时
猜测」现象的学术对应。

### 2.3 推论：能涨分的题 = 模型「偶尔做对」的边界题

- 纯欠约束题（bit 歧义位、cryptarithm 91%）：模型怎么训也只能猜 → 无解。
- 已稳定做对的题：无增益。
- **真正有头空间的是「边界题」**：模型采样时偶尔答对（如 numeric_guess 16%、bit 偶尔蒙对）。
  强化这些题上「走对路」的概率，是唯一现实的杠杆。其规模需采样统计才知道（见 §4 探针）。

---

## 3. 文献调研指向的新方向（按 ROI 排序）

> 来源：2026-06-02 深度文献调研，27 篇主要文献、对抗验证 24 条断言。完整论文清单见 §7。

### 路线 A（首选）｜答案过滤的自训练 RFT / STaR / ReST-EM

**核心**：用 Huikang 模型自己（最强求解器）生成大量 CoT，只保留①最终答案正确
②（用我们已写的求解器/规则检查当**验证器**）推理在所有 in-context 样例上自洽 的链，
去重最大化 distinct 路径，再 LoRA SFT，迭代几轮。

**为什么这条绕过了 §2.1 的死结**：生成器是模型本身，不是弱求解器 → 不存在负迁移来源。
文献证据最强（STaR/RFT/ReST-EM/V-STaR 四篇全部 3-0 一致），且我们的设定比文献更有利
（文献用 held-out 单样例过滤，假阳性 45.4%；我们有**真答案**过滤，假阳性低得多）。

**两条可直接落地的机制性结论**：
1. **收益来自 distinct（不同）正确路径的数量，不是采样总量**。文献：33B 模型生成 88.7 条
   正确路径但只有 2.78 条 distinct/题时 RFT **失败**；合并多模型采样把 distinct 从 5.25
   提到 12.84，GSM8K +13.4pt。→ 高温采样、聚合多变体（如不同 adapter / seed），别只多采。
2. **忠实性过滤必须用多样例**，不能只看最终答案——否则会强化「答案蒙对但推理错」的链。
   这里复用我们已写的 `symbol_digit_solver.py` / `cipher_cot_builder.py` 当验证器：检查 CoT
   推出的规则能否解释**所有**给定样例。

**现实收益边界（务必诚实）**：只能救 §2.3 的边界题，对纯欠约束题无效。净收益是经验问题。

**所需基础设施（大部分已有）**：venv-vllm 高温采样（rollout.py 可改）、求解器当验证器、
venv 做 LoRA SFT（train.py + `--resume_adapter_dir`）、canary + 全量 eval 当门槛。

### 路线 B｜程序合成离线当数据生成器（求解器=验证器）

**核心**：Hypothesis Search / induction-vs-transduction 范式——LLM 生成假设→实现为程序→
**在样例上验证**。这是精确符号变换（bit/cipher/roman/算术）最强的范式，但它的核心机制
（运行候选程序验证、大规模采样过滤）是**测试时**机制，与我们「T=0 单次无工具」冲突。

→ 现实用法是 **离线**：在训练集上跑「大规模采样 + 样例验证 + majority vote」（文献：solve
rate 随采样预算几乎单调上升，只 9% 假阳性，majority vote 能消掉大半），把验证通过的链
通过路线 A 的 RFT 蒸馏回模型。本质上是路线 A 的一个更强的数据生成变体。

**附带洞察**：induction（归纳规则再应用）与 transduction（直接预测）解决的题大部分**不相交**，
集成强 → 训练数据里值得同时包含「显式规则 trace」和「直接预测」两种风格。

### ~~路线 C｜Test-Time Training / 多采样~~ — 已除名（grader 关闭，见 §0）

TTT 是 ARC 最高天花板技术（6x、61.9%=人类平均），机制上就是 per-instance LoRA 更新。**但
2026-06-02 已确认**：提交物是固定 adapter，主办方 harness 单次评分，参赛者无法在评分时反向传播
或多采样。→ TTT / best-of-N / self-consistency **全部不可用**。唯一残留价值：把"多采样共识"
**离线蒸馏**进 adapter（即 RFT，路线 A），但上限仍受可救空间（§0）限制。

### 路线 D｜欠约束诊断 + 先验重塑（研究性）

度量模型的规则频率先验，尝试通过 RFT 数据分布**重塑**它（上/下调某些规则族权重），看能否
在边界题上移动准确率。ConceptSearch 的「按变换概念而非输出相似度排序」可借鉴用于离线候选
选择。理论参考：贝叶斯程序归纳（Lake 2015）、Solomonoff / MDL 先验。

### 低风险零碎修复（与主线平行，期望 +0.1~0.2pp）

只修**非推理**错误：malformed 输出（9 题）、gravity 舍入、截断（3%）。到不了 0.86 但稳、不退步。

---

## 4. 推荐的下一步：先做采样探针（不烧 Kaggle 配额）

在投入路线 A 的完整 RFT 之前，先用 Huikang 模型对各类别失败题**高温采样 n=8**，统计每类
「至少采到一条正确答案」的比例。这个数字直接给出 RFT 的真实天花板（§2.3 边界题占比）：

- 若某类边界题比例高 → 路线 A 在该类有头空间，值得投入。
- 若接近 0（纯欠约束）→ 该类 RFT 也救不了，不投。

几个 GPU 小时（venv-vllm，sbatch）即可出结果。门槛遵循项目规矩：canary 验证正迁移 →
本地全量 eval 9500 确认目标类别升、高分类别不退 → 才提交 Kaggle。

---

## 5. grader 机制（已确认 2026-06-02，曾是最高杠杆待确认项）

**结论：推理时杠杆全部关闭。** 核实方式：Kaggle 官方提交格式 = LoRA adapter(rank≤32,
submission.zip)，主办方固定 harness 对每题生成单个 response 提取 boxed；`resource/` 里的
submission notebook 中那些 `SamplingParams(n=3)` 是参赛者**本地 eval** 用的，不是评分逻辑。
- 推理时**反向传播 / TTT**：❌ 固定 adapter 单次前向，不可能。
- **每题多次采样 / best-of-N / self-consistency**：❌ harness 单次评分，无法插入。
- **程序执行 / 工具调用**：❌ 无工具通道。
→ 路线 B/C 只能**离线**用（当训练数据生成器 / 蒸馏进 adapter）。这道门的关闭是本轮把
"+1pp 是否可达"从开放问题变成否定结论的关键一步（见 §0）。

---

## 6. Headroom 汇总（用于优先级）

| Task | 待解 | 整体头空间 | 现实可达性 |
|---|---|---|---|
| cryptarithm_deduce | 613 | +6.45% | 仅 ~4% 可学（91% 欠约束）→ 上限 ~+0.3% |
| bit_manipulation | 200 | +2.10% | 77% 差 1-2 位欠约束；RFT 只能救边界题 |
| cryptarithm_guess | 154 | +1.62% | 同 deduce，更难 |
| equation_numeric_guess | 114 | +1.20% | **最有希望**：无 per-题密码，combo 先验可学，模型已 16% 说明有边界题 |

---

## 7. 关键论文清单

**自训练 / 拒绝采样（路线 A）**
- STaR — Zelikman et al. 2022, [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- RFT — Yuan et al. 2023, [arXiv:2308.01825](https://arxiv.org/abs/2308.01825)（distinct-path 机制）
- ReST-EM — Singh et al. 2023, [arXiv:2312.06585](https://arxiv.org/abs/2312.06585)
- V-STaR — Hosseini et al. COLM 2024, [arXiv:2402.06457](https://arxiv.org/html/2402.06457v2)

**ARC / 程序合成 / TTT（路线 B、C）**
- Test-Time Training — Akyürek et al. ICML 2025, [arXiv:2411.07279](https://arxiv.org/abs/2411.07279)
- Induction vs Transduction — Li/Ellis et al. ARC Prize 2024 最佳论文, [arXiv:2411.02272](https://arxiv.org/abs/2411.02272)
- Hypothesis Search — Wang et al. ICLR 2024, [arXiv:2309.05660](https://arxiv.org/abs/2309.05660)
- ConceptSearch — Singhal & Shroff AAAI 2025, [arXiv:2412.07322](https://arxiv.org/abs/2412.07322)
- DreamCoder — Ellis et al. PLDI 2021, [arXiv:2006.08381](https://arxiv.org/abs/2006.08381)

**欠约束归纳 / 先验（路线 D，§2.2 印证）**
- 规则归纳错误来源分析 — [arXiv:2509.01016](https://arxiv.org/html/2509.01016)（45.4% 假阳性；NeurIPS 2025 workshop，证据中等）
- Bayesian Program Induction — Lake et al. Science 2015, [aab3050](https://www.science.org/doi/10.1126/science.aab3050)

**过程监督 / 长链可靠性**
- Let's Verify Step by Step — Lightman et al. 2023, [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
- Math-Shepherd — [arXiv:2312.08935](https://arxiv.org/abs/2312.08935)

> 注：LoRA 最佳实践方向（rank/target 选择）无文献断言通过对抗验证，证据弱；以 `CLAUDE.md`
> 自有硬事实（Mamba in_proj/out_proj 必入 target、rank≤32、SVD 转换 ~25% 有损）为准。

---

## 8.（历史附录）旧 RL 战略论证 —— 已证伪，仅作设计依据保留

> 以下是上一版 roadmap 的核心，写于 GRPO 实验**之前**。其核心预测「必须做 RL → LB 0.88-0.90」
> **未成立**：GRPO 在 Huikang adapter 上因策略熵塌缩无法启动（T=0.95/G=8 每组仅 1.6-1.8 个不同
> 序列，组内优势≈0，无梯度）。完整证伪见 `ITERATION_REPORT.md` §6，GRPO 数学与 FSDP 工程参考见
> `RL_ARCHITECTURE.md`。保留要点如下，供将来若重启 RL 时参考。

**当时的论证**：verifier reward 精确无误差 → RL 理论上限 = policy capacity × exploration，
唯一能突破「teacher 也不会」的 cryptarithm。算法选 GRPO（无 value head，组相对优势）。
分 task 独立 RL，再 OPD 整合。

**为什么没成立（gatekeeper 结论）**：Huikang 的 SFT 为对齐 T=0 grader 把策略熵压到接近 0，
恰好抹掉了 GRPO 赖以工作的探索多样性。给 RL 最后一次机会的不训练 gatekeeper 实验是：
**起点换成 Nemotron base**（剔除 Huikang LoRA），G=16、T=1.2、top_k=64，先测 rollout 统计——
通过门槛 avg distinct/G ≥ 4 且 mixed-group rate ≥ 10%；不达标则正式放弃 RL。

**与新方向的关系**：路线 A 的 RFT 其实是 RL 的「离线、稳定」版本——同样用 verifier（答案正确性）
做信号，但不依赖在线 rollout 多样性，因此不受策略熵塌缩的致命阻断。这是从 GRPO 失败中学到的
方向修正。
