# OPD 迭代报告（`opd` 分支）

> 记录区间：2026-05-13 → 2026-05-30。
> 起点 = `output/huikang-baseline-converted/`（公开 LB **0.85**，9500 自评 **87.79%**）。
> 目标 = 把弱类目（cryptarithm + equation_numeric_guess）拉起来，LB 突破 0.85。
> **本轮结论：三次有效提交全部停在 0.85，SFT 蒸馏与定制 solver 两条路被实验证伪，GRPO 在 Huikang 适配器上因策略熵塌缩无法启动。详见下文。**

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
