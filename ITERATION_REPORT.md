# OPD 迭代报告（`opd` 分支）

> 记录区间：2026-05-13 → 2026-06-02。
> 起点 = `output/huikang-baseline-converted/`（公开 LB **0.85**，9500 自评 **87.79%**）。
> 目标 = 把弱类目（cryptarithm + equation_numeric_guess）拉起来，LB 突破 0.85。
> **05-30 结论：三次有效提交全部停在 0.85，SFT 蒸馏与定制 solver 两条路被实验证伪，GRPO 在 Huikang 适配器上因策略熵塌缩无法启动。**
> **06-02 续（§7）：用更强的组合框架求解器再试一次，量化证伪了「手写求解器生成 CoT → SFT」整条路线（求解器全线弱于模型），并经文献调研转向 RFT/STaR 自训练。详见 §7 与 `ROADMAP.md`。**

---

## 0. TL;DR

| 路线 | 做了什么 | 结果 | 判定 |
|---|---|---|---|
| 多类目蒸馏 SFT（opd-v1） | 1399 条多类目 CoT，warmstart 0.85 | LB **0.85**（无变化） | ❌ 无增益 |
| cryptarithm 定向蒸馏（opd-v2） | 115 条 verified 扩展-DSL CoT，3× 过采样 | LB **0.85**（无变化） | ❌ 无增益 |
| Canary holdout 诊断 | cryptarithm 留出集上训练→评测 | 净 **−1 题**（负迁移） | ❌ SFT 对 cryptarithm 破产 |
| 自制 DSL solver（自动造 CoT） | 21-op DSL 解 F 簇 | 100 题 **0 解** | ❌ solver 族结构性失配 |
| GRPO（强化学习） | 手搓 GRPO + 多节点 FSDP 全流水线 | rollout 多样性塌缩，**无梯度信号** | ⚠️ 起点策略选错，见 §6 |

**一句话**：cryptarithm 的失败不是"数据不够"或"训练不够"，而是 **(a) 教师模型本身也不会**（蒸馏天花板低）+ **(b) Huikang SFT 已把策略压成近确定性**（RL 没有探索面）。LB 0.85 是当前可接受的停点。

---

## 1. 起点 baseline（实测，9500 训练集，grader-aligned T=0）

`output/huikang-baseline-converted/`，4 shard × 4×L40s vLLM 并行，T=0 / max_tokens=7680 / max_model_len=8192。

| Category | Correct / Total | Acc | 评级 |
|---|---|---|---|
| numeral | 1576 / 1576 | **100.00%** | 满分 |
| unit_conversion | 1591 / 1594 | **99.81%** | 接近满分 |
| gravity | 1591 / 1597 | **99.62%** | 接近满分 |
| cipher | 1560 / 1576 | **98.98%** | 接近满分 |
| equation_numeric_deduce | 542 / 596 | **90.94%** | 良好 |
| bit_manipulation | 1402 / 1602 | **87.52%** | 良好但卡 |
| equation_numeric_guess | 22 / 136 | **16.18%** | ❌ 悬崖 |
| cryptarithm_deduce | 46 / 659 | **6.98%** | ❌ 悬崖 |
| cryptarithm_guess | 10 / 164 | **6.10%** | ❌ 悬崖 |
| **总计** | **8340 / 9500** | **87.79%** | |

本地 87.79% 与 Huikang writeup 的 87.8% 吻合；与 LB 0.85 的 ~3% 差距来自私测集分布略难，**不是 grader 配置 mismatch**。增益空间几乎全在 3 个悬崖类目（cryptarithm_deduce 头空间最大 +6.45% LB）。

---

## 2. Kaggle 远端提交记录（本轮相关）

> 完整历史见 `kaggle competitions submissions`；下表只列与本轮 `opd` 工作相关的提交。私测分 (privateScore) 比赛期间不公开，故为空。

| 日期 (UTC) | 提交 | 描述 | 状态 | Public LB |
|---|---|---|---|---|
| 2026-05-13 11:55 | BASELINE-CONVERTED | Huikang v20 + SVD merge + 128-expert unfuse（保留 75.77% 奇异质量），复现 fork 基线 | COMPLETE | **0.85** |
| 2026-05-24 07:48 | **opd-v1** | warmstart 0.85 + 1-epoch SFT，opd_v1 数据集（1399 条多类目 CoT），lr=1e-5 linear，eff_batch=64，train_loss 240→196 | COMPLETE | **0.85** |
| 2026-05-25 06:29 | **opd-v2** | Huikang 基线 + 115 条 verified 扩展-DSL cryptarithm CoT（3× 过采样），1 epoch lr=1e-5 | COMPLETE | **0.85** |

> 对照：本项目历史 self-trained 天花板是 **0.76**（2026-05-12 v6 Unsloth recipe，6355 样本 1-epoch）。fork Huikang = 0.85。本轮三次提交均未越过 0.85。

---

## 3. 逐实验结果（本地 + 远端）

### Exp 1 — opd-v1：多类目蒸馏 SFT
- **数据**：`training/opd/runs/opd_v1/train.csv`（46054 行 ≈ 1399 条样本，覆盖 cryptarithm/bit/cipher/eq_num/gravity/numeral/unit）。
- **训练**：warmstart Huikang 0.85，1 epoch，lr=1e-5 linear，effective_batch=64。train_loss 240→196（per-token ≈ 0.66），grad_norm 18→6.6。
- **远端**：LB **0.85**（无变化）。
- **判定**：多类目混合蒸馏没有撬动任何悬崖类目；强类目本就满分、弱类目教师 CoT 质量不足以迁移。

### Exp 2 — opd-v2：cryptarithm 定向蒸馏
- **数据**：`training/opd/runs/opd_v2/train.csv`（47207 行）。核心是 115 条**经 `verify()` 验证正确**的扩展-DSL cryptarithm CoT，3× 过采样。
- **远端**：LB **0.85**（无变化）。
- **判定**：即使把数据聚焦到唯一高潜力类目、且只喂正确 CoT，仍无增益 → 触发下面的 holdout 诊断，定位"为什么 SFT 推不动 cryptarithm"。

### Exp 3 — Canary holdout 诊断（关键否定结论）⭐
在 cryptarithm 上切出 holdout，训练 `opd-canary`，在留出 30 题上对比基线：

| | 30 题总 | crypt_deduce | crypt_guess |
|---|---|---|---|
| baseline（Huikang） | 3/30 = **10.0%** | 2/24 = 8.33% | 1/6 = 16.67% |
| opd-canary (ckpt-20) | 2/30 = **6.67%** | 1/24 = 4.17% | 1/6 = 16.67% |

- **净 −1 题（负迁移）**。训练侧 train_loss 95→13（模型**能**拟合训练 CoT），但 holdout **零正迁移**甚至倒退。
- **判定**：cryptarithm 上 **SFT 蒸馏破产**——模型记住了训练样本的具体映射，但 cryptarithm 每题是**独立的随机 letter→digit + op 加密**，没有可迁移的"规则"，记忆无法泛化。这解释了 opd-v1/v2 为何卡死 0.85。

### Exp 4 — 自制 DSL solver（想绕开教师，自动造正确 CoT）
既然教师 CoT 质量不够，尝试用程序化 solver 暴力解出 cryptarithm 真值、再合成"保证正确"的 CoT。在最大的 F 簇上诊断 21-op DSL solver（100 题抽样）：

| 配置 | no_feasible | multi_feasible | timeout/cap | **solved** |
|---|---|---|---|---|
| CURR（默认 base 区间，unique=True） | 87 | 13 | 0 | **0** |
| WIDE（放宽 base 区间） | 81 | 18 | 1 | **0**（wide_helps=0） |
| unique=False 重测 | — | — | — | **0** |

- **判定**：三个维度（DSL 算子集 / base 进制范围 / 双射唯一性约束）全部 0 解。F 簇的运算不在我们 21 个算子的张成空间里 → **solver 族选错了，不是缺算子**。自动造 CoT 这条路对 84% 的失败题无效。

### Exp 5 — 未解 cryptarithm 聚类（643 题）
对 643 道基线做不出的 cryptarithm 按签名聚类：

| 簇 | 含义 | 题数 | 占比 |
|---|---|---|---|
| **F** | digit-cipher，更难的运算 | **540** | **84.0%** |
| G | digit-cipher，固定 rhs 长度 | 47 | 7.3% |
| A+B | concat（拼接型） | 52 | 8.1% |
| E | string-DSL（字符串变换） | 4 | 0.6% |

- 之前估计"string DSL 占 40%"被证伪——实际 string-DSL 只有 **4 题**，F 簇（digit cipher 硬运算）独占 **84%**。
- A+B concat 簇也实测无救：concat 盲猜匹配 0、rev_concat 仅 3、signature-aware 0 → 该任务已删。
- **判定**：任何针对性 solver/CoT 方案要有意义，必须攻克 F 簇；而 Exp 4 已证明 F 簇对我们的 solver 族不可达。

### Exp 6 — GRPO 强化学习流水线（本轮工程主体）
**动机**：SFT（记忆）和自制 solver（程序化）都死了，转向 RL——让模型在自己**已经能偶尔做对**的题上，通过组内相对优势强化正确推理路径。

**已建成**（`training/rl/`，手搓 GRPO，非 HF Trainer 子类）：
- `rollout.py`（venv-vllm，TP=4）→ `reward.py`（`verify()` 包装 + 组相对优势）→ `grpo_loss.py`（token logprob + GRPO 目标 + Schulman k3 KL）→ `train_grpo.py`（FSDP + PEFT 手写训练环）。rollout↔train 用 JSONL 解耦。
- **Hot-pool 课程**：基线 T=0 greedy 扫 959 道 RL 目标题，只留做对的 → **78 题**（crypt_deduce 46/659、crypt_guess 10/164、eq_num_guess 22/136）。
- 算法补丁：β=0（跳过 KL anchor，避开 disable_adapter+FSDP+LoRA 交互）；dynamic sampling 过滤零方差组（DAPO/NeMo-RL）。

**工程坑（已修）**：
1. **vLLM seed bug**：`SamplingParams(seed=N, n=8)` 下 8 条补全近乎相同（131/154 前 100 字符一致）。移除 seed 后恢复多样性。
2. **OOM**：1-node 4-GPU FSDP 撞 ~35GB/rank 静态天花板（44GB L40S）。max_length 4096→2200→1200 都无效。改 2-node 8-way FSDP（smoke v8）解决。

**致命阻断（核心发现）**：Huikang 适配器是**近确定性策略**。在 T=0.95 / G=8 下：
- 平均每组只产出 **1.6–1.8 个不同序列**。
- v5 distinct 计数分布 `{1:14, 2:4, 4:2, 3:1, 6:1}`（22 组）；v8 `{1:13, 2:5, 3:3, 4:1}`。
- 半数以上的组 **8 条补全完全相同** → 组内优势全 0 → GRPO **没有可学习的梯度**。

**判定**：这不是超参问题，是**起点策略选错**。Huikang 的 SFT 为对齐 grader（T=0 贪婪）把策略熵压到接近 0，恰好抹掉了 GRPO 赖以工作的探索多样性。

---

## 4. 关键结论

1. **cryptarithm 的墙是双重的**：教师模型本身 ~25-30%（蒸馏天花板低）+ 每题独立随机加密（记忆不可泛化）。全 LB 几百队无人破解 0.90+，印证这是**领域边界**而非 30B 模型局限。
2. **SFT 蒸馏路线在 cryptarithm 上证伪**：opd-v1/v2 双双 0.85，canary 负迁移 −1。
3. **自制 solver 造 CoT 路线证伪**：F 簇（84%）对 21-op DSL solver 0 解。
4. **GRPO 在 Huikang 适配器上无法启动**：策略熵塌缩，rollout 多样性 1.6–1.8 < 阈值。

---

## 5. 决策点（gatekeeper）与下一步

Codex 拍板：**停掉在 Huikang 适配器上的 vanilla GRPO**。若仍要给 RL 最后一次机会，只做**一个**不训练的 gatekeeper 实验：

| 项 | 值 |
|---|---|
| 起点模型 | **Nemotron base**（剔除 Huikang LoRA） |
| Prompt pool | ≥ 256 |
| Group size | G = 16 |
| Sampling | T=1.2, top_k=64, top_p=0.95 |
| 训练 | **不训练**，只测 rollout 统计 |
| 通过门槛 | avg distinct/G ≥ 4 **且** mixed-group rate ≥ 10% |
| 不达标 | **正式宣布 RL 不通**，转推理时优化 |

**若 gatekeeper 失败** → 接受 LB 0.85 为停点，转 best-of-N + verifier rerank / self-consistency / search-then-distill。

---

## 6. 本次提交包含的工件

> 重产物（safetensors / rollout JSONL / SLURM 日志 / 推理 dump）已被各 `runs/.gitignore` 排除；结果数据固化在本报告中。

- **eval/**：`category.py`、`eval_full_train.{py,sbatch}`、`eval_holdout.{py,sbatch}`、`eval_shard.sbatch` — grader-aligned 全量/留出评测工具。
- **training/fsdp_*.yaml**：单节点 / 多节点 / offload 三套 FSDP 配置。
- **training/rl/**：完整手搓 GRPO 流水线（pool/rollout/reward/ref_policy/grpo_loss/train_grpo/utils）+ configs + scripts（smoke v3–v8、多节点、hot/hard-pool 构建）+ README/STATUS。
- **training/opd/**：蒸馏与诊断工具——trace pool 构建、merge、canary、F-bucket 诊断（`solvers/f_diagnostic*`）、未解聚类（`solvers/cluster_unsolved.py`）、扩展 solver、CoT 模板/生成。Kaggle 提交脚本已脱敏（token 改读环境变量 `KAGGLE_API_TOKEN`）。
- **docs/**：`路线图.md`（策略）、`RL架构指南.md`（GRPO 数学 + 架构）、本报告。

---

## 7. 2026-06-02 续：求解器路线证伪 + 文献调研转向

**动机**：Kaggle 讨论 #684283（用户 Donald Galliano）公布了一套与我们旧 DSL 不同的求解器
框架——把 cipher/symbol-digit 题建模为 `AB<op>CD` 的「4 种数字排列 × ~14 运算 × ~14 输出格式」
频率排序枚举（不是旧的 letter→digit 约束求解）。这是**对的求解器家族**（旧 DSL 在 F 簇 0/100，
正因家族选错）。本轮用它再试一次「求解器造 CoT → SFT」，但结果是更彻底的证伪。

### Exp 7 — cipher-digit 答案引导求解器
- 实现 `training/opd/solvers/{symbol_digit_solver,cipher_digit_solver,cipher_cot_builder}.py`。
- 关键 bug 修复：例子需按**问题的运算符**过滤（每个运算符规则独立），运算符取自 question
  位置 2（非第一个例子）。修正后 numeric 求解率 3.6%→32.4%。
- **答案引导**生成器（用已知答案反推密码映射）在 equation_symbolic 上解出 42%（319/823），
  0.1s/题。但纯无答案求解器只有 ~4%、6.5s/题。

### Exp 8 — 忠实性检查（否定结论）⭐
对 319 条答案引导 cipher CoT 做两个独立的忠实性检查：
| 检查 | 真正可学 | 欠约束/泄漏 |
|---|---|---|
| 符号覆盖（question+answer 符号是否都在样例出现） | 38/319 (12%) | 281 (88%) |
| 纯样例求解器一致性（不给答案能否解出且一致） | **13/319 (4%)** | 290 (91%) + **16 题解出不同答案（多解）** |

- **判定**：答案引导 CoT 88% 含样例推不出的符号 → 教模型在欠约束题上自信幻觉，T=0 下最糟。
  那 16 题多解直接证明欠约束是信息论性质，非求解器弱。详见 memory `project_cryptarithm_sft_ceiling`。

### Exp 9 — 求解器 vs 模型交叉表（决定性证伪）⭐⭐
写了 bit 频率先验加权投票求解器（`bit_solver_v2.py`，先验从 600 道做对的题统计：identity 42%
/CONST 14%/AND-XOR-OR-XNOR-NOT 次之）。逐类别对比求解器与模型准确率：

| 类别 | 手写求解器 | 模型 | 用求解器输出 SFT |
|---|---|---|---|
| bit_manipulation | 59.9% (960/1602) | 87.5% | 能教 +58 / **会教错 −500** |
| equation_numeric_deduce | 39.8% | 90.9% | +2 / **−307** |
| equation_numeric_guess | 0% | 16.2% | 0 / −22 |
| cipher-digit | ~4% | 7% | — |

- **判定**：模型已比我们能手写的任何符号求解器都强。「弱求解器生成 CoT → SFT」必然净负迁移
  ——这是 opd-v1/v2 停在 0.85 的统一根因。详见 memory `project_solver_cannot_beat_model`。

### Exp 10 — 失败模式诊断
- 截断（num_tokens≥7600）仅占失败的 **3%** → 失败不是长度/格式问题。
- bit 失败题 **154/200（77%）只差 1-2 个 bit**（歧义位猜错）；bit「全 8 位唯一确定」仅 2%。
- **判定**：失败本质是欠约束，模型在歧义位按最优先验猜，信息论上无法系统修复。

### Exp 11 — 文献调研（转向依据）
深度文献调研（27 篇主要文献、对抗验证 24 条断言）指向新方向，完整论文清单与 ROI 排序见
`ROADMAP.md` §3、§7。要点：
- **答案过滤的自训练（STaR/RFT/ReST-EM/V-STaR）是首选**——模型自己生成 CoT、答案过滤、再训
  自己，绕过「弱求解器→负迁移」死结（生成器即模型本身）。收益来自 distinct 正确路径数量
  （非采样量），过滤要用多样例（45.4% 单样例假阳性）。
- 程序合成（Hypothesis Search/induction）做**离线**数据生成器；求解器降级为**验证器**。
- TTT 高天花板但受 grader 约束，待确认 grader 是否允许推理时反传/多采样（最高杠杆待确认项）。

### 本轮工件
- `training/opd/solvers/`：`symbol_digit_solver.py`、`cipher_digit_solver.py`、
  `cipher_cot_builder.py`（答案引导）、`bit_solver.py`、`bit_solver_v2.py`（频率先验投票）。
  **注**：这些作为**验证器**（路线 A 的忠实性过滤）仍有用；作为**生成器**已证伪。
- `training/opd/{gen_cipher_cot.py,build_v3_dataset.py,gen_cot_data.sbatch}` 与
  `training/train_cot_v3.csv`、`training/train_v3.sbatch`：基于已证伪路线的产物，**未提交训练**
  （证伪后主动中止，未烧 GPU/配额）。保留作为路线 A 验证器代码的来源，可按需清理。

---

## 8. 2026-06-03：bit RFT/STaR 自训练 pilot（pass@k 翻案 → 实证净增 ~0）

接 §7（求解器证伪）。文献调研指向 RFT/STaR（模型自采 CoT、答案过滤、SFT 自己），先做 pass@k
探针实证可救空间，再 pilot 验证。**结论：bit 经 pass@k 看似可救（70%），但 RFT 实际净增 ~0
——回归吃掉恢复，是先验耦合的结构性限制。0.85 确认为当前框架天花板。**

### Exp 12 — pass@k 探针（推翻"bit 不可救"，但只是看似）
Huikang adapter 对各类失败题 T=0.9 n=16 采样（每类 50 题）：

| 类别 | pass@16 | pass@1 | 解读 |
|---|---|---|---|
| bit_manipulation | **0.70** | 0.207 | 70% 失败题能采到正确答案 → 看似能力受限可救 |
| equation_numeric_guess | 0.00 | 0 | 死类（采 16 次一次不对，RFT 无可挖）|
| cryptarithm_deduce | 0.00 | 0 | 死类 |
| cryptarithm_guess | 0.00 | 0 | 死类 |

- bit 推翻 §7/早间诊断基于**静态求解器**的"bit 不可救/净救+3"——求解器看不到模型 T>0 的能力。
- 全 140 bit 失败题 PROBE（G=16 T=1.0）：pass@1=0.242，过滤后 **262 distinct 正确 CoT / 89 题**。

### Exp 13 — bit RFT pilot（PROBE → FILTER → train → eval）
- **make_holdout**：bit 切 train(140 失败+1262 已对)/holdout(60 失败+140 已对)，反泄漏断言通过。
- **FILTER**（rft_pipeline）：gate1 答案对（丢 1697 错）+ gate2（bit 用 WEAK，不让弱求解器误杀）
  + 去重 ≤4/题 → 262 CoT。
- **训练集** 825 行（262 bit 恢复 + 263 bit anchor + 300 非 bit anchor），bit token 占 84%。
- **pilot**：resume Huikang、lr=5e-5、3 epoch（长 CoT 每步 ~15min，4h 超时停在 ~step16）、save_steps=5。
  输出 output/rft-r0-pilot（**原 Huikang 0.85 adapter 完好未动，可回滚**）。
- **EVAL**（eval_holdout_rft.py）：baseline + 每 checkpoint 同口径 **T=0 重跑** holdout+sentinel
  （150 非 bit 已对题），绕过被污染的 predicted 字段。

### 结果（决定性，baseline holdout_acc=0.71）
| checkpoint | bit 恢复 | bit 回归 | sentinel 漂移 | 净增 | holdout_acc |
|---|---|---|---|---|---|
| **checkpoint-5** (~0.4ep) | 5 | 2 | 0 | **+3** | 0.72 |
| checkpoint-10 | 8 | 11 | 2 | −5 | 0.69 |
| checkpoint-15 | 9 | 11 | 15 | −17 | 0.70 |

- checkpoint-5 是甜点（holdout 净 +3）；训练稍多即过拟合崩盘（**sentinel 漂移 0→2→15**）。
- **外推全集**：恢复率 5/60=8.3%×200≈**+17**，回归率 2/140=1.4%×**1402**≈**−20** → bit 净≈**−3**；
  整体净 ≈ **0 或微负**。**不值得提交**（LB 大概率仍 0.85）。

### 根因与方法论教训
1. **pass@k 高 ≠ RFT 能净增**：正确 CoT 存在于采样分布，不代表能无副作用提取出来。
2. **先验耦合（核心）**：bit 失败题与已对题共享同一套歧义位频率先验，RFT 把失败题的 gate 选择
   往正确拉时，不可避免地动了已对题用的同一先验 → 修一个坏一个。**已对基数大（1402）时，
   1.4% 回归率的绝对数（20）吃掉 8.3% 恢复（17）。** 这是结构性的，非调参能根本解决。
3. **长 CoT 专项 SFT 过拟合极快**：甜点在 ~0.4 epoch，之后非 bit 全局漂移崩盘。
4. **验证天花板的正确指标 = holdout 净收益外推全集**，不是 pass@k、不是 holdout 子集 acc、
   更不是 train loss（loss 一路降，但 holdout 净收益早就转负）。frequent checkpoint + T=0
   holdout 净收益选点是唯一可靠的门。

### RL 可行性修正（与 §6 GRPO 对照，关乎"是否回头试 RL"）
§6 记 GRPO 死在 cryptarithm（策略熵塌缩，distinct 1.6–1.8，无梯度）。但 pass@k 显示 **bit 不同**：
T=0.9 下 pass@1=20%、distinct-correct 多 → bit 在 T>0 **有探索多样性**，G=16 组内有正有负、
reward 方差非零 → GRPO advantage 非零、**有梯度**。即 **RL（GRPO）在 bit 上技术可行**（不像
cryptarithm 那样无法启动）。但 RL 面临与 RFT **完全相同的先验耦合限制**——在 bit 失败题上做
policy-gradient 同样会动 bit 已对题的先验。RL 唯一可能的优势是 KL anchor 比 MLE-SFT 更温和地
控制回归（β 限制偏离 ref policy）——但这是未验证的希望，且 GRPO pipeline 工程成本远高于 RFT，
而 RFT pilot 已用更便宜的方式探明天花板（净 ~0）。详见 ROADMAP §0′。

### 结论
"救失败题"的最后一个有实证支撑的候选（bit，pass@16=70%）经 pilot 验证**净增 ~0**。
**0.85 是「Huikang baseline + LoRA + 固定单次评分」框架的天花板。**
