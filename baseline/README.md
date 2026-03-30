# Baseline — Answer-Only LoRA Fine-tuning

第一阶段训练：对 NemotronH 进行 answer-only 的 LoRA 微调。模型学习在给定 prompt 后直接输出 `\boxed{answer}`，不含思考过程。

## 1. 训练配置

| 参数 | 值 |
|------|-----|
| GPU | 1× H100 |
| 时间限制 | 4h |
| 内存 | 80G |
| 数据采样 | 分层采样：7 类别各 400 条，共 2800 条 |
| max_steps | -1（完整 1 epoch，约 175 步） |
| batch_size | 4，grad_acc=4（等效 batch 16） |
| max_length | 512 |
| 学习率 | 2e-4，cosine decay |
| warmup | `warmup_ratio=0.05`（总步数的 5%） |
| 优化器 | AdamW Fused |
| 精度 | BF16 |
| NEFTune | noise_alpha=5.0 |

## 2. LoRA 配置

| 参数 | 值 |
|------|----|
| rank | 32 |
| alpha | 16（scaling ratio = 0.5） |
| dropout | 0.05 |
| target_modules | `in_proj`, `out_proj`, `up_proj`, `down_proj` |
| 可训练参数 | ~880M / 32.5B（2.71%） |
| 模型加载方式 | `device_map="balanced"`（多卡模型并行，非 DDP） |

## 3. 数据集划分

数据按 7 个任务类别从各自 CSV 中进行分层采样（seed=42 固定）：

- **训练集**：每类前 400 条，共 2800 条
- **Eval pool**：每类剩余样本，共 6700 条（与训练集完全无交集）

| 类别 | 总量 | 训练 | Eval pool |
|------|------|------|-----------|
| bit_ops | 1602 | 400 | 1202 |
| cipher | 1576 | 400 | 1176 |
| equation_numeric | 732 | 400 | 332 |
| equation_symbolic | 823 | 400 | 423 |
| gravity | 1597 | 400 | 1197 |
| numeral | 1576 | 400 | 1176 |
| unit_conv | 1594 | 400 | 1194 |
| **合计** | **9500** | **2800** | **6700** |

## 4. 数据格式与监督 Loss

每条样本格式化为 ChatML 对话，**仅对 assistant 回复部分计算 loss**（user prefix 的 label 全部设为 -100，PyTorch 自动跳过）：

```
<|im_start|>user
{prompt}
Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`<|im_end|>
<|im_start|>assistant
\boxed{answer}<|im_end|>
← 只有这部分 token 参与 loss 计算 →
```

Loss 类型为标准 **Cross-Entropy next-token prediction**。chat template 的 `<think></think>` 前缀已被主动 strip，训练目标中不含任何 think token。

## 5. 训练中评测（Eval Callback）

使用 HuggingFace `model.generate()` 进行 in-training eval：

| 参数 | 值 |
|------|----|
| eval_steps | 25（每 25 步触发一次） |
| eval_samples_per_class | 5（共 35 条） |
| eval_batch_size | 8（batch generate + left-padding） |
| max_new_tokens | 256 |
| 评测指标 | 官方 metric：`\boxed{}` 提取 + 数值 1% 容差 / 字符串精确匹配 |
| 日志 | TensorBoard（`eval/accuracy` + 各类别准确率） |

评测数据从 eval pool 中每类取前 5 条（共 35 条），与训练集完全独立。

**注意**：训练结束后保存的是**最后一步的权重**（step 175），而非 eval 最佳的 checkpoint。最佳权重为 `checkpoint-100`。

## 6. 训练结果

### Eval 准确率（每 25 步，每类 5 条，共 35 条）

| Step | 总计 | bit_ops | cipher | eq_num | eq_sym | gravity | numeral | unit_conv |
|------|------|---------|--------|--------|--------|---------|---------|-----------|
| 25 | 6/35 (17%) | 0/5 | 0/5 | 1/5 | 0/5 | 0/5 | **5/5** | 0/5 |
| 50 | 5/35 (14%) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | **5/5** | 0/5 |
| 75 | 5/35 (14%) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | **5/5** | 0/5 |
| **100** | **7/35 (20%)** | **1/5** | 0/5 | 0/5 | 0/5 | 0/5 | **5/5** | **1/5** |
| 125 | 6/35 (17%) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | **5/5** | 1/5 |
| 150 | 6/35 (17%) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | **5/5** | 1/5 |
| 175 | 5/35 (14%) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | **5/5** | 0/5 |
| final | 5/35 (14%) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | **5/5** | 0/5 |

### vLLM Eval（7680 tokens，每类 5 条 — Kaggle 对齐）

使用 `cot_training/eval_vllm.py` 重新评测（与 Kaggle 评测条件一致）：

| Checkpoint | 总计 | bit_ops | cipher | eq_num | eq_sym | gravity | numeral | unit_conv | Kaggle |
|------------|------|---------|--------|--------|--------|---------|---------|-----------|--------|
| **checkpoint-100** | **23/35 (65.7%)** | 1/5 | 3/5 | 4/5 | 0/5 | 5/5 | 5/5 | 5/5 | **0.64** |

注意：in-training eval（max_new_tokens=256）只显示 20% 准确率，而 Kaggle 对齐的 vLLM eval（7680 tokens）达到 65.7%。差异来自推理型任务（gravity、unit_conv 等）需要足够的 token 才能完成思考链。

### 关键观察

- **numeral** 类稳定满分，该类任务已被充分学习
- **gravity、unit_conv** 在 7680 tokens 下满分，但 256 tokens 下为 0 — answer-only 训练后模型自发学会了（不完美的）推理
- **bit_ops、equation_symbolic** 即使 7680 tokens 也接近 0%，需要 CoT 训练
- 最佳 eval 出现在 **step 100（epoch 0.57）**，之后略有下降

### 保存的 checkpoint

- `checkpoint-100` — 最佳 eval，Kaggle 得分 **0.64**，作为第二阶段 CoT 混合训练的起始权重
- `checkpoint-175` — 最后一步

## 7. 文件说明

| 文件 | 用途 |
|------|------|
| `train.py` | 训练主脚本：数据加载、LoRA 配置、Eval Callback、NEFTune |
| `eval.py` | 评测工具库：`extract_final_answer()`、`verify()`、`evaluate_samples()`（被 cot_training 复用） |
| `run_baseline.sbatch` | SBATCH 作业脚本（1× H100, 4h，训练 + 自动打包） |

## 8. 使用方式

```bash
# 训练
sbatch baseline/run_baseline.sbatch

# 查看日志
tail -f logs/nemotron-baseline-<JOBID>.out

# 手动打包
bash scripts/build_submission.sh output/baseline-lora output/
```

## 9. 已知兼容性问题

| 问题 | 修复方式 |
|------|----------|
| `mamba-ssm` CUDA 扩展与 torch 2.9+ ABI 不兼容 | 先安装 mamba-ssm，让 pip 自动选择兼容的 torch 版本 |
| `mamba-ssm` 导入 `GreedySearchDecoderOnlyOutput`（transformers 5.x 已删除） | `train.py` 顶部注入 dummy 占位类 |
| `TrainingArguments` 不再支持 `no_cuda` 参数 | 已删除该参数 |
| `warmup_steps` 固定值难以适配不同步数 | 改用 `warmup_ratio=0.05`（自适应） |
| max_length=1024 导致 GPU 利用率仅 45% | 改为 max_length=512，吞吐翻倍 |
| batch_size=8 触发 OOM | 改为 batch_size=4 + grad_acc=4 维持等效 batch=16 |
| 逐条 eval generate 导致超时 | 改为 batch generate（bs=8），每轮 eval ~17min |
| TensorBoard 日志目录为空 | `logging_dir` 被 Trainer 忽略，实际写入 `output_dir/runs/`。已创建符号链接 |
| GPU 利用率稳定 ~50% | 30B MoE 模型仅 3B 参数活跃，属正常现象 |

## 10. TODO

- [ ] **eval.py 切换到 vLLM**：当前 `evaluate_samples()` 使用 HF `model.generate()`（1.4 tok/s），应改为使用 `venv-vllm/` + vLLM 推理（~900 tok/s），与 `cot_training/eval_vllm.py` 对齐
- [ ] **load_best_model_at_end**：自动保存 eval 最佳 checkpoint，避免手动对比
