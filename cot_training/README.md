# CoT Mixed Training

Chain-of-Thought 混合微调模块。在 baseline LoRA checkpoint 基础上，混合官方训练数据与合成 CoT 推理链数据进行继续训练。训练完成后使用 vLLM 批量评测所有 checkpoint。

## 1. 数据

### 1.1 数据来源

训练集由两部分混合构成：

| 类型 | 来源 | 样本数 | 格式 |
|------|------|--------|------|
| **普通样本** | `data/by_category/*.csv`（官方 train.csv 按类别拆分） | 7 类 × 500 = 3500 | `(prompt, answer)` |
| **CoT 样本** | `data/cot/{bit_ops,equation_symbolic,cipher}/level_*.jsonl`（合成数据） | ~1500 | `(prompt, cot_answer, answer)` |
| **总计** | | **~5000** | |

其中 bit_ops、cipher、equation_symbolic 三个类别同时拥有普通样本（500 条）和 CoT 样本（~500 条），其余四个类别（equation_numeric、gravity、numeral、unit_conv）仅有普通样本。

### 1.2 数据抽样与划分

使用固定随机种子 `seed=42` 确保可复现：

```
对每个类别的 CSV（按文件名排序）：
    random.Random(42).shuffle(rows)
    训练集 = rows[:500]            # 前 500 条
    验证集 = rows[500:505]          # 接下来 5 条（默认）
    剩余   = rows[505:]            # 未使用
```

- **验证集**：7 类 × N 条，与训练集零重叠（N 可通过 eval 脚本配置，默认 5，vLLM eval 用 20）
- **CoT 数据**：全部加载（不做采样），每个类别包含多个难度等级的 level_*.jsonl
- **最终训练集**：普通样本 + CoT 样本全部混合后随机 shuffle，不做课程学习排序

### 1.3 输入网络的格式

所有样本统一使用 `enable_thinking=True` 的 chat template，确保训练与推理格式一致。两类样本的监督区域不同：

**普通样本** — 仅监督最终答案，思考区域完全 mask：

```
[MASKED] <|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n
[SUPERVISED, w=1.0] \boxed{answer}<|im_end|>
```

设计原因：我们没有这些类别的 CoT ground truth，但不希望教模型"不要思考"。mask 掉 `<think>...</think>` 区域，模型在推理时可以自由产生任意思考内容，我们只监督最终答案的正确性。

**CoT 样本** — 监督完整推理链 + 答案：

```
[MASKED] <|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n<think>\n
[SUPERVISED, w=1.0] {cot_reasoning}
[SUPERVISED, w=1.0] \n</think>\n\n
[SUPERVISED, w=5.0] \boxed{answer}<|im_end|>
```

## 2. Loss 设计

### 2.1 Per-token Weighted Cross-Entropy

使用自定义 `WeightedTrainer`（继承 HuggingFace `Trainer`），替换标准的 cross-entropy loss 为按 token 加权的版本：

```python
loss = sum(CE(token_i) * weight_i) / sum(weight_i)
```

每个 token 位置携带一个 `loss_weight`：

| 位置 | 权重 |
|------|------|
| Prefix（用户输入 + think 标签） | 0.0（masked，不参与 loss） |
| CoT 推理链 token | `reasoning_loss_weight`（默认 1.0） |
| `\boxed{answer}` token | `boxed_loss_weight`（默认 5.0） |
| Padding | 0.0 |

`\boxed{}` 部分的 5× 权重确保模型不仅学会模仿推理模式，更要**得出正确的最终答案**。

### 2.2 样本级归一化

**核心问题**：普通样本的监督区域平均 ~11 tokens，而 CoT 样本平均 ~1260 tokens。如果按 token 平均 loss，CoT 推理链会占据 93.9% 的 loss 贡献，普通样本几乎被淹没。

**解决方案**：每个样本的 `loss_weights` 在预处理时归一化为 sum=1.0。这样无论序列长短，每个样本对 batch loss 的贡献相等：

```
归一化前：Regular 占 1.6% | CoT reasoning 占 93.9% | CoT boxed 占 4.5%
归一化后：Regular 占 66%  | CoT 占 34%（与样本数比例一致）
```

归一化保持了样本内部的相对权重比（boxed : reasoning = 5 : 1）。

### 2.3 Decomposed Loss Logging

除了总加权 loss（`train/loss`），`WeightedTrainer` 还分别记录 boxed tokens 和 reasoning tokens 的**未加权平均 CE loss**，写入 TensorBoard：

| 指标 | 含义 |
|------|------|
| `train/loss` | 总加权 loss（用于梯度更新） |
| `train/boxed_loss` | `\boxed{answer}` 部分的平均 CE（答案质量） |
| `train/reasoning_loss` | 推理链 + 普通样本部分的平均 CE（推理质量） |

实现方式：dataset 层面每个 token 携带一个 `is_boxed` boolean mask，`compute_loss` 中用它分离两类 tokens 的 unweighted CE。这些分拆指标仅用于监控，**不影响梯度计算**。

**如何解读**：
- boxed_loss 下降 → 最终答案越来越准
- reasoning_loss 下降 → 推理过程学习进展
- boxed_loss 收敛但 reasoning_loss 还在降 → 推理能力仍在提升
- reasoning_loss 反转上升 → 可能开始过拟合

### 2.4 实现结构

```
dataset.py          MixedDataset: 加载、tokenize、计算 loss_weights + is_boxed 并归一化
weighted_trainer.py WeightedTrainer: 自定义 compute_loss + decomposed loss logging
                    WeightedDataCollator: 处理 loss_weights / is_boxed 的 padding
train.py            主脚本：模型加载、训练编排（无 in-training eval）
eval_vllm.py        vLLM 评测脚本：训练后批量评测所有 checkpoint
```

## 3. 训练原则

### 3.1 从 Baseline 继续训练

从 `baseline-lora/checkpoint-100`（Kaggle 得分 0.64 的最优 baseline）加载 LoRA adapter 继续训练，而非从头开始。

从 checkpoint 继续训练优于从零开始的原因：
- 7 类中只有 3 类有 CoT 数据，其余 4 类只能靠 answer-only 样本 — 从零开始等于浪费 baseline 已学的能力
- baseline 已学会基础格式（`\boxed{}` 输出）和简单类别（numeral 满分、gravity 满分）
- 实测验证：150 步就从 Kaggle 0.64 提升到 0.67

### 3.2 不做课程学习

虽然 CoT 数据包含多个难度等级（level 1-4），但混合训练中**不做课程排序**，所有样本随机 shuffle。原因：

- 普通样本没有难度分级，无法与 CoT level 对齐
- Shuffle 使每个 batch 包含多样化的样本类型，梯度更稳定
- 之前的纯课程学习实验效果不理想

### 3.3 防止灾难性遗忘

之前的纯 CoT 训练（仅 1000 条 CoT 样本，无普通数据）导致 Kaggle 得分从 0.64 降至 0.48。本次通过混合训练解决：

- 7 个类别的普通样本持续提供监督信号，保持已学能力
- 3 个类别额外获得 CoT 推理链训练，增强思考能力
- 样本级归一化确保普通样本不被长序列 CoT 数据淹没

### 3.4 NEFTune 正则化

训练时在 embedding 层注入均匀噪声（`alpha=5.0`），仅在 `model.training` 时生效，eval 时自动跳过。提高小数据集微调的泛化性。

## 4. 训练配置

```bash
python cot_training/train.py \
    --checkpoint_dir            output/baseline-lora/checkpoint-100 \
    --regular_samples_per_class 500 \
    --max_length                2048 \
    --boxed_loss_weight         5.0 \
    --reasoning_loss_weight     1.0 \
    --num_epochs                1 \
    --batch_size                1 \
    --grad_acc                  16 \
    --lr                        5e-5 \
    --neftune_alpha             5.0 \
    --save_steps                20 \
    --save_total_limit          -1 \
    --save_only_model
```

| 参数 | 值 | 说明 |
|------|-----|------|
| Effective batch size | 1 × 16 = 16 | batch_size=1（长序列）× grad_acc=16 |
| 总步数 | ~313 | ~5000 样本 / 16 × 1 epoch |
| LR schedule | cosine, warmup 5% | 从 baseline 继续训练，LR 较低（5e-5 vs baseline 2e-4） |
| save_steps | 20 | 每 20 步保存 checkpoint（约 16 个） |
| save_only_model | True | 只保存 adapter 权重（~1.7GB vs ~10GB） |
| 资源 | 2× H100, 28 CPUs, 160G RAM, 24h | `run_cot_training.sbatch` |

**注意**：训练中**不做 eval**。评测改为训练完成后使用 vLLM 批量评测所有 checkpoint（见第 6 节）。

## 5. 端到端流程

提交一个 sbatch 即可完成训练 + 评测，所有指标写入同一个 TensorBoard run：

```
sbatch cot_training/run_cot_training.sbatch
│
│  ┌─────────────────────────────────────────────────────────┐
│  │  训练阶段（venv/, 2× H100, ~24h）                       │
│  │                                                         │
│  │  train.py                                               │
│  │    ├── 每 5 步 → TensorBoard:                           │
│  │    │     train/loss                                     │
│  │    │     train/boxed_loss     ← \boxed{} 部分 CE        │
│  │    │     train/reasoning_loss ← 推理链部分 CE            │
│  │    │     train/learning_rate, train/grad_norm            │
│  │    ├── 每 20 步 → 保存 checkpoint（~1.7GB, adapter only）│
│  │    └── 训练结束 → 写 .tb_run_dir 标记文件                │
│  └─────────────────────────────────────────────────────────┘
│
│  训练成功后自动提交 eval job
│
▼
sbatch cot_training/eval_vllm.sbatch（自动）
│
│  ┌─────────────────────────────────────────────────────────┐
│  │  评测阶段（venv-vllm/, 1× H100, ~2-4h）                 │
│  │                                                         │
│  │  eval_vllm.py                                           │
│  │    ├── 读取 .tb_run_dir → 打开同一个 TensorBoard run     │
│  │    ├── 自动发现所有 checkpoint-*                          │
│  │    ├── vLLM 推理（7680 tokens, 每类 20 样本, ~900 tok/s） │
│  │    ├── 每个 checkpoint → TensorBoard:                    │
│  │    │     eval/accuracy                                  │
│  │    │     eval/accuracy_{category}（7 类分别）             │
│  │    └── 输出对比表 + JSONL 详细结果                        │
│  └─────────────────────────────────────────────────────────┘
```

**TensorBoard 中的效果**：一个 run 里同时看到 loss 下降曲线和 eval 准确率变化，共享同一条 x 轴（global_step）。

## 6. TensorBoard 监控

### 6.1 记录的指标

| 指标 | 来源 | 阶段 | 频率 |
|------|------|------|------|
| `train/loss` | WeightedTrainer（总加权 loss） | 训练 | 每 5 步 |
| `train/boxed_loss` | WeightedTrainer（`\boxed{}` 部分 unweighted CE） | 训练 | 每 5 步 |
| `train/reasoning_loss` | WeightedTrainer（推理链 + 普通样本 unweighted CE） | 训练 | 每 5 步 |
| `train/learning_rate` | Trainer 内置 | 训练 | 每 5 步 |
| `train/grad_norm` | Trainer 内置 | 训练 | 每 5 步 |
| `eval/accuracy` | eval_vllm.py（vLLM 推理） | 评测 | 每 20 步 |
| `eval/accuracy_{category}` | eval_vllm.py（7 类分别） | 评测 | 每 20 步 |

### 6.2 日志写入路径

训练和评测写入同一个 TensorBoard event 目录：

1. 训练时，HuggingFace Trainer 创建 `output/cot-lora/runs/{timestamp}/`
2. 训练结束时，`train.py` 将该路径写入 `output/cot-lora/.tb_run_dir`
3. 评测时，`eval_vllm.py` 读取 `.tb_run_dir`，打开同一个 `SummaryWriter`

`run_cot_training.sbatch` 中有后台 watcher 每 30s 自动在 `~/tensorboard_logs/` 下创建软链接，使 TensorBoard 能实时发现训练日志。

## 7. vLLM 评测 Pipeline

### 7.1 为什么用 vLLM

NemotronH 是 Mamba-Attention 混合架构，HuggingFace `model.generate()` 的 KV cache 与之不兼容，推理速度仅 **1.4 tok/s**。Kaggle 官方评测使用 vLLM，我们本地评测也切换到 vLLM：

| | HF transformers | vLLM 0.17.1 |
|---|---|---|
| 推理速度 | ~1.4 tok/s | **~900 tok/s** |
| 35 样本 × 7680 tokens | ~2 小时 | **~2 分钟** |
| KV cache | 不兼容 | 原生支持（FlashAttention v3 + Mamba cache） |
| LoRA 加载 | PeftModel.from_pretrained | LoRARequest（热切换） |
| 与 Kaggle 对齐 | 部分对齐 | 完全对齐 |

### 7.2 双 venv 架构

由于 vLLM 需要 torch 2.10+，而训练依赖 mamba-ssm 绑定 torch 2.5.1，两者不兼容。因此使用两个独立虚拟环境：

| 用途 | venv | 核心包 | 备注 |
|------|------|--------|------|
| **训练** | `venv/` | torch 2.5.1, transformers 5.3.0, mamba-ssm 2.2.4, peft | 需要 mamba-ssm CUDA kernel |
| **推理** | `venv-vllm/` | torch 2.10.0, vLLM 0.17.1, peft | 需要 `module load opencv/4.13.0`（在激活 venv 前） |

### 7.3 评测脚本

`eval_vllm.py` 支持两种 checkpoint 指定方式：

```bash
# 自动发现目录下所有 checkpoint-*
python cot_training/eval_vllm.py \
    --scan_dir output/cot-lora \
    --checkpoint_dir output/baseline-lora/checkpoint-100 \
    --max_tokens 7680 \
    --eval_samples_per_class 20

# 手动指定多个 checkpoint
python cot_training/eval_vllm.py \
    --checkpoint_dir output/cot-lora/checkpoint-100 \
    --checkpoint_dir output/cot-lora/checkpoint-200 \
    --max_tokens 7680
```

vLLM 引擎只加载一次，通过 `LoRARequest` 热切换不同 adapter，避免重复加载 60GB 的 base model。

### 7.4 评测配置（Kaggle 对齐）

| 参数 | 值 | 说明 |
|------|-----|------|
| max_tokens | 7680 | 与 Kaggle 官方一致 |
| temperature | 0.0 | 贪心解码 |
| top_p | 1.0 | 无采样 |
| max_model_len | 8192 | 模型最大上下文 |
| gpu_memory_utilization | 0.85 | 与 Kaggle 一致 |
| eval_samples_per_class | 20 | 7 类 × 20 = 140 样本 |
| 评测指标 | `\boxed{}` 提取 + 数值 1% 容差 / 字符串匹配 | 复用 `baseline/eval.py` |

## 8. 评测结果

### 8.1 vLLM Eval（7680 tokens，每类 5 条）

| Checkpoint | 总计 | bit_ops | cipher | eq_num | eq_sym | gravity | numeral | unit_conv |
|------------|------|---------|--------|--------|--------|---------|---------|-----------|
| baseline checkpoint-100 | 23/35 (65.7%) | 1/5 | 3/5 | 4/5 | 0/5 | 5/5 | 5/5 | 5/5 |
| **CoT mixed checkpoint-150** | **25/35 (71.4%)** | 1/5 | **5/5** | 4/5 | 0/5 | 5/5 | 5/5 | 5/5 |

### 8.2 Kaggle 提交分数

| Checkpoint | Kaggle 分数 |
|---|---|
| baseline checkpoint-100 | 0.64 |
| **CoT mixed checkpoint-150** | **0.67** |

### 8.3 关键观察

- **cipher 从 60% → 100%**：合成的 cipher CoT 数据效果显著
- **gravity、numeral、unit_conv 保持满分**：无灾难性遗忘
- **bit_ops、equation_symbolic 仍为 0-20%**：7680 tokens 也不够完成推理，需改进 CoT 数据质量
- checkpoint-150 只训练了 ~0.37 epoch，训练尚未完成

## 9. 使用方式

```bash
# Step 1: 提交训练（2× H100, 24h, 每 20 步保存 checkpoint）
sbatch cot_training/run_cot_training.sbatch

# Step 2: 训练完成后，评测所有 checkpoint（1× H100, vLLM）
sbatch cot_training/eval_vllm.sbatch

# Step 3: 查看 eval 结果
cat logs/eval-vllm-<JOBID>.out

# Step 4: 打包最佳 checkpoint 提交 Kaggle
bash scripts/build_submission.sh output/cot-lora/checkpoint-XXX output/
bash scripts/submit.sh output/submission.zip "description"
```

## 10. 文件说明

| 文件 | 用途 |
|------|------|
| `train.py` | 训练主脚本：模型加载、MixedDataset 构建、WeightedTrainer 训练 |
| `dataset.py` | MixedDataset：加载 regular + CoT 数据，tokenize，计算 loss_weights + is_boxed |
| `weighted_trainer.py` | WeightedTrainer（per-token weighted CE + decomposed logging）+ WeightedDataCollator |
| `eval_vllm.py` | vLLM 评测：自动发现 checkpoint，batch 推理，输出对比表 |
| `run_cot_training.sbatch` | 训练 sbatch（2× H100, 24h, save_steps=20） |
| `eval_vllm.sbatch` | vLLM eval sbatch（1× H100, 4h, 每类 20 样本） |
