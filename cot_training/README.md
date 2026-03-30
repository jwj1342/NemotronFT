# CoT Mixed Training

Chain-of-Thought 混合微调模块。在 baseline LoRA checkpoint 基础上，混合官方训练数据与合成 CoT 推理链数据进行继续训练。

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
    验证集 = rows[500:505]          # 接下来 5 条
    剩余   = rows[505:]            # 未使用
```

- **验证集**：7 类 × 5 条 = 35 条，与训练集零重叠
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

### 2.3 实现结构

```
dataset.py          MixedDataset: 加载、tokenize、计算 loss_weights 并归一化
weighted_trainer.py WeightedTrainer: 自定义 compute_loss
                    WeightedDataCollator: 处理 loss_weights 的 padding
train.py            主脚本：模型加载、训练编排、eval callback
```

## 3. 训练原则

### 3.1 从 Baseline 继续训练

从 `baseline-lora/checkpoint-100`（Kaggle 得分 0.64 的最优 baseline）加载 LoRA adapter 继续训练，而非从头开始。模型已具备基础的数学推理能力，CoT 混合训练在此基础上增强。

### 3.2 不做课程学习

虽然 CoT 数据包含多个难度等级（level 1-4），但混合训练中**不做课程排序**，所有样本随机 shuffle。原因：

- 普通样本没有难度分级，无法与 CoT level 对齐
- Shuffle 使每个 batch 包含多样化的样本类型，梯度更稳定
- 之前的纯课程学习实验（旧版 `CotCurriculumDataset`）效果不理想

### 3.3 防止灾难性遗忘

之前的纯 CoT 训练（仅 1000 条 CoT 样本，无普通数据）导致 Kaggle 得分从 0.64 降至 0.48。本次通过混合训练解决：

- 7 个类别的普通样本持续提供监督信号，保持已学能力
- 3 个类别额外获得 CoT 推理链训练，增强思考能力
- 样本级归一化确保普通样本不被长序列 CoT 数据淹没

### 3.4 NEFTune 正则化

训练时在 embedding 层注入均匀噪声（`alpha=5.0`），仅在 `model.training` 时生效，eval 时自动跳过。提高小数据集微调的泛化性。

### 3.5 评估对齐 Kaggle

Eval 使用 `max_new_tokens=7680`，与 Kaggle 官方评测一致（贪心解码，`top_p=1.0`）。每 200 步评估一次，整个训练过程约 3-4 次 eval。

## 4. 默认训练配置

```bash
python cot_training/train.py \
    --checkpoint_dir            output/baseline-lora/checkpoint-100 \
    --regular_samples_per_class 500 \
    --max_length                2048 \
    --boxed_loss_weight         5.0 \
    --reasoning_loss_weight     1.0 \
    --num_epochs                2 \
    --batch_size                1 \
    --grad_acc                  16 \
    --lr                        5e-5 \
    --neftune_alpha             5.0 \
    --eval_steps                200 \
    --eval_max_new_tokens       7680
```

| 参数 | 值 | 说明 |
|------|-----|------|
| Effective batch size | 1 × 16 = 16 | batch_size=1（长序列）× grad_acc=16 |
| 总步数 | ~625 | ~5000 样本 / 16 × 2 epochs |
| LR schedule | cosine, warmup 5% | 从 baseline 继续训练，LR 较低（5e-5 vs baseline 2e-4） |
| 资源 | 1× H100, 14 CPUs, 80G RAM, 8h | `run_cot_training.sbatch` |

## 5. TensorBoard 监控

### 5.1 日志写入路径

HuggingFace Trainer 的 `TensorBoardCallback` 将 event 文件写入：

```
output/cot-lora/runs/{timestamp}/events.out.tfevents.*
```

注意：`TrainingArguments` 的 `logging_dir` 参数**不会被 TensorBoardCallback 使用**。实际写入路径由 `TENSORBOARD_LOGGING_DIR` 环境变量决定，未设置时 fallback 到 `{output_dir}/runs/{timestamp}/`。

### 5.2 自动软链接

为了让 TensorBoard（指向 `/home/jwj/tensorboard_logs/`）能及时发现新 run，`run_cot_training.sbatch` 中启动了一个后台 watcher：

```bash
# 每 30s 检查 output/cot-lora/runs/ 下的新目录，自动创建软链接
while true; do
    for d in "$TB_RUN_DIR"/*/; do
        name="$(basename "$d")"
        [ -L "$TB_LINK_DIR/$name" ] || ln -sf "$d" "$TB_LINK_DIR/$name"
    done
    sleep 30
done
```

训练结束后自动清理该后台进程。

### 5.3 监控时间线

```
sbatch 提交
  ↓
后台 watcher 启动
  ↓ (~30s)
软链接自动创建 → TensorBoard 发现新 run
  ↓
每 5 步:  train/loss, train/learning_rate, train/grad_norm
每 200 步: eval/accuracy, eval/accuracy_{category} (7 类)
  ↓
训练结束 → final eval → watcher 清理
```

### 5.4 记录的指标

| 指标 | 来源 | 频率 |
|------|------|------|
| `train/loss` | WeightedTrainer（加权 loss） | 每 5 步 |
| `train/learning_rate` | Trainer 内置 | 每 5 步 |
| `train/grad_norm` | Trainer 内置 | 每 5 步 |
| `eval/accuracy` | EvalCallback | 每 200 步 + final |
| `eval/accuracy_{category}` | EvalCallback（7 类分别） | 每 200 步 + final |

Eval 使用 `max_new_tokens=7680`（与 Kaggle 对齐），每次约 26 分钟（35 样本），整个训练过程约 3-4 次 eval。
