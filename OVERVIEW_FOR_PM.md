# 项目速览

> ### 速记(4 句)
> - **任务**:固定 base `Nemotron-3-Nano-30B`,只交付一个 **rank≤32 LoRA adapter**,在 NVIDIA 推理 benchmark 上最大化准确率。
> - **关键约束**:grader = **T=0 贪心、单次前向、max_tokens 7680、逐题二元判分**,且**封闭**(推理时无 best-of-N / TTT / 搜索)。能力只能烤进 adapter 权重。
> - **现状**:**LB 0.86**,与第一梯队同档(0.86–0.87 一大片,个别 0.89)。差距 = **cryptarithm 这一类**,全场共识的墙。
> - **策略 & 目标**:可动类别上做可提交方法(bit→GRPO、arbitrary-glyph→operator-fingerprinting SFT)+ 严谨证伪 cryptarithm 可学性;冲 **Open Contribution Award**(方法+文档,top 10% 即可)。

---

## 1. 比赛设定 + 约束

- 提交物 = LoRA adapter(`adapter_config.json` + 权重,rank≤32),vLLM 加载到固定 base 后判分。
- grader:`T=0, top_p=1, max_tokens=7680, max_model_len=8192`,单次 greedy 生成,exact / 1e-2 数值容差判分,**无部分分**。
- **封闭推理**:提交只有 adapter 权重、解码循环固定 → **beam / lookahead / best-of-N / self-consistency / TTT 全不可用**。这是 cryptarithm 难的根因之一(搜索无法外挂,必须烤进权重)。
- 截止 **6/15**。奖项:总榜前 3(需 ~0.89+,够不到)+ **Open Contribution Award**(Best Data / RL / Fine-tuning,top 10% + 公开 notebook/writeup,各 1 台 DGX Spark)——我们的实际目标。

---

## 2. 难点:per-category + cryptarithm 墙

```
✅ ~99%    numeral / unit_conversion / gravity / cipher                 饱和
🟡 87–91%  bit_manipulation(占 ~17%,最大类)/ equation_numeric_deduce   有空间
🔴 6–16%   cryptarithm_deduce / cryptarithm_guess / equation_numeric_guess   墙
```

- **0.86 ≈ 把 well-posed / 可学类别榨干**;train acc ~87.8% vs LB 0.85(lossy 转换)→ 缺口集中在 hard 类。
- **cryptarithm 结构**:5-char LHS `[s][s][op][s][s] = result`,需**联合反解** symbol→digit bijection(10–13 符)× 运算 glyph(13 种 op)× 全局 preprocessing `(none/swap)³`。本质是 **joint CSP + 回溯搜索**。
- **模型为何不行**:需**有状态的全局搜索/回溯**;模型强在**无状态局部计算**(故 bit 87.5%:转置位列→逐 op 匹配,fits in budget)。实测 **pass@16 = 0** → 正解不在 policy support 内,不是 decoding 选错。
- 注:讨论区 "rust solver 71%" 是外部程序,封闭推理下用不上;唯一用法是拿其 trace 做 SFT——而 trace→SFT 我们试过(见下)。

---

## 3. 方法 + 教训(每行一句)

| 方法 | 结果 / 教训 |
|---|---|
| cryptarithm SFT 蒸馏 | holdout canary **负迁移**(每题独立随机 cipher,记忆不泛化)|
| solver→CoT→SFT | 自研 solver 弱于模型(bit 60% vs 87.5%)→ 训练即 **mis-teach**;solver 只配当 verifier |
| RFT / STaR 自训练 | 净 ~0:**prior coupling**(修 fail 同时回归 1402 个已对的)|
| 复刻公开 0.87 recipe | 短 generic-trace SFT → **reasoning collapse**(88%→44%,输出 token 6684→32);且我们 lossless-0.86 已等价吃掉该 recipe |
| **bit GRPO**(β=0) | holdout **净 +3/100、0 回归**(group-relative ± advantage 优于 RFT)· **提交中** |
| **operator-fingerprinting SFT**(主攻) | 失败根因 = 模型按 glyph 惯例而非从样例 fingerprint 算子;统一覆盖 crypt-concat + eqn-deduce · **提交中** |

> **方法论铁律(吃过 3 次亏)**:可救性只能靠 **holdout 净收益外推全集**判定——static-solver 估计 / pass@k 上界 / subset-acc 都系统性误导过。所以现在**不靠本地估算下结论,跑完即提交,LB 说话**。

---

## 4. 现状 · 价值 · 下一步

- **现状**:LB **0.86**;bit-GRPO 与 fingerprinting 两个 adapter 训练/评估中,过 regression gate 即提交(本周内两个 LB 读数)。
- **独特价值**:① **6 路独立证伪** cryptarithm 可学性(executability probe 6% < 25% gate / branch-budget audit / pass@16=0 / canary 负迁移 …)——社区无人系统做过;② 可动类别上的**真实可提交方法**(GRPO 净正、fingerprinting)。
- **下一步**:等两次 LB 出分;整理 writeup + notebook 申报 Open Contribution Award。

---

*技术细节:根目录 `ITERATION_REPORT.md`(逐次实验)/ `ROADMAP.md`(策略)/ `RL_ARCHITECTURE.md`(GRPO 参考)。*
