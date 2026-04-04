# Distillation Mixed Training

在 baseline LoRA 基础上，混合三种数据源进行微调：regular SFT + CoT synthesis + distillation data。

## 数据源互补关系

CoT 合成数据与蒸馏数据几乎完全互补，覆盖不同的类别：

| 类别 | Regular SFT | CoT Synthesis | Distillation | CoT↔Distill 重叠 |
|------|------------|---------------|-------------|------------------|
| bit_ops | 600 | 500 | 1 | 1 |
| cipher | 600 | 500 | 183 | 0 |
| equation_symbolic | 600 | 500 | 52 | 0 |
| equation_numeric | 600 | 500 | 0 | 0 |
| gravity | 600 | 0 | 167 | 0 |
| numeral | 600 | 0 | 196 | 0 |
| unit_conv | 600 | 0 | 145 | 0 |
| **合计** | **4200** | **2000** | **744** | **1** |

- CoT synthesis 覆盖推理密集型类别（bit_ops, cipher, equation_*）
- Distillation 覆盖计算密集型类别（gravity, numeral, unit_conv）
- CoT 与 Distillation 仅 1 条 bit_ops 重叠，可忽略

## Regular 与 Distillation 的同源去重

Regular SFT 数据与 distillation 数据**同源**（distillation 是将 regular 的 prompt 喂给 teacher model 得到的），存在大量 prompt 重叠：

| 类别 | Regular | Distill | Regular↔Distill 重叠 |
|------|---------|---------|---------------------|
| cipher | 1576 | 183 | 183 |
| gravity | 1597 | 167 | 167 |
| numeral | 1576 | 196 | 196 |
| unit_conv | 1594 | 145 | 145 |
| bit_ops | 1 | 1 | 1 |
| equation_symbolic | 823 | 52 | 0（不同子类型） |
| equation_numeric | 732 | 0 | 0 |

如果同时使用两者，同一个 prompt 会被训练两次（一次只学答案，一次学推理链+答案），相当于对该样本加权了两倍。

**解决方案：优先采纳 distillation 版本。** 数据加载时按 prompt 去重：

1. 先加载 distillation 数据，建立 `prompt → (cot_answer, answer)` 索引
2. 加载 regular 数据时逐条检查：
   - **命中 distillation** → 升级为 distillation 版本（监督完整推理链 + 答案）
   - **未命中** → 保持 regular 版本（仅监督 `\boxed{answer}`）
3. CoT synthesis 数据照常加载（与 regular/distillation 互补）

最终每个 prompt 只出现一次，无重复计数。

## Loss 设计

- 所有样本 uniform weight，不区分 boxed/reasoning 的权重差异
- Sample-level normalization：每个样本的 loss_weights 归一化到 sum=1.0，确保长短样本等权贡献
- Prefix masking：prompt 部分不参与 loss 计算

三类样本的监督范围：

| 样本类型 | 监督范围 | 来源 |
|---------|---------|------|
| Regular | 仅 `\boxed{answer}` | 无蒸馏匹配的原始样本 |
| Upgraded | reasoning chain + `\boxed{answer}` | 有蒸馏匹配，优先使用蒸馏版本 |
| CoT | reasoning chain + `\boxed{answer}` | CoT 合成数据 |

## 实验结果（distill-v1）

### 训练配置

- 基于 `baseline-lora/checkpoint-100`（Kaggle 0.64）继续训练
- 2×H100，effective batch size 32，lr=5e-5，cosine schedule
- 8200 samples（3911 regular + 289 upgraded + 4000 CoT）
- 每 20 步保存 checkpoint，训练到 step 320（约 1.5 epoch）后超时
- **注意**：此版本蒸馏数据利用率仅 19.2%（155/809），已在 distill-v2 中修复

### 最佳 Checkpoint

**`best-distill-v1-step160-kaggle070`** — Kaggle 0.70（历史最高）

已保存至 `output/best-distill-v1-step160-kaggle070/`，不会被后续训练覆盖。

| 项目 | 值 |
|------|-----|
| 来源 | `output/distill-lora/checkpoint-160` |
| Kaggle Score | **0.70** |
| Local Eval | 59.3%（166/280） |
| 基础模型 | `baseline-lora/checkpoint-100`（Kaggle 0.64） |
| 训练数据 | 8200 samples = 3911 regular + 289 upgraded + 4000 CoT |
| 蒸馏利用率 | 19.2%（155/809 distill prompts matched train split） |
| 训练步数 | 160 steps（~epoch 1 峰值） |
| Batch size | 32 effective（2×H100） |
| Learning rate | 5e-5, cosine schedule |
| Loss weights | uniform（boxed=1.0, reasoning=1.0） |

### Kaggle 提交结果

| Checkpoint | Kaggle Score | Local Eval | 说明 |
|---|---|---|---|
| baseline-100 | 0.64 | — | 起点 |
| CoT mixed step-150 (旧最佳) | 0.67 | — | 之前的历史最高 |
| **distill step-160** | **0.70** | 59.3% | **新纪录，+3pp** |
| distill step-180 | 0.66 | 59.3% | 接近旧最佳 |
| distill step-240 | 0.52 | 53.6% | epoch 2 过拟合 |

### 各类别 Local Eval 趋势

| Step | bit_ops | cipher | eq_num | eq_sym | gravity | numeral | unit_conv | Overall | 阶段 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 15.0% | 52.5% | 22.5% | 0.0% | 100% | 100% | 100% | 55.7% | Epoch 1 早期 |
| 80 | 15.0% | 67.5% | 25.0% | 0.0% | 100% | 100% | 100% | 58.2% | Epoch 1 中期 |
| **160** | 5.0% | **75.0%** | **35.0%** | 0.0% | 100% | 100% | 100% | **59.3%** | **Epoch 1 峰值 ★ Kaggle 0.70** |
| 180 | 25.0% | 60.0% | 32.5% | 0.0% | 100% | 100% | 97.5% | 59.3% | Epoch 1 |
| 200 | 22.5% | 2.5% | 30.0% | 0.0% | 100% | 100% | 100% | 50.7% | Epoch 2 崩塌 |
| 240 | 35.0% | 5.0% | 35.0% | 0.0% | 100% | 100% | 100% | 53.6% | Epoch 2 过拟合 |

### 关键发现

1. **Distillation + CoT 混合训练有效**：从 0.64 基线提升到 0.70，超过旧最佳 0.67
2. **Epoch 1 是甜蜜点**：step 160 附近最优，进入 epoch 2 后迅速过拟合
3. **Cipher 是 Kaggle 分数的关键类别**：step-160（cipher 75%）和 step-180（cipher 60%）local eval 相同（59.3%），但 Kaggle 分数差 4pp（0.70 vs 0.66），说明 cipher 在测试集上权重较大
4. **Epoch 2 出现"跷跷板"效应**：cipher 从 75%→2.5% 崩塌，但 bit_ops 从 5%→35% 上升，模型容量有限导致类别间互相挤占
5. **equation_symbolic 全程 0%**：最难类别，当前数据和方法无法攻克
6. **蒸馏数据利用率低（v1 缺陷）**：旧逻辑先随机 shuffle 再取前 600 条，蒸馏 prompt 大多落在 train split 之外，仅 19.2% 被实际使用。distill-v2 已修复为优先纳入蒸馏 prompt

## Eval 数据泄漏修复

eval 脚本中发现 distillation 数据的 prompt 有 26 条（9.3%）落入 eval 区间（rows[600:640]），已在 `eval_vllm.py` 的 `load_eval_pool` 中增加过滤逻辑，排除 distillation 和 CoT 训练集中出现过的 prompt，从后续行补充。

## 用法

```bash
# 训练
sbatch distillation_training/run_distill_training.sbatch

# 评估
sbatch distillation_training/eval_vllm.sbatch
```
