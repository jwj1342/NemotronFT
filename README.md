# NVIDIA Nemotron-3-Nano-30B LoRA Fine-tuning

Kaggle 比赛：[NVIDIA Nemotron 3 Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)

对 `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` 进行 LoRA 微调，提交符合比赛格式的 adapter。

---

## 项目结构

```
NemotronFT/
├── model/                        # 下载的基础模型权重（~60GB BF16）
├── data/
│   ├── train.csv                 # 原始训练集（~69k 条，含 id/prompt/answer）
│   ├── test.csv                  # 测试集（34 条）
│   ├── by_category/              # 按 7 类别拆分的 CSV
│   │   ├── bit_ops.csv           # 1602 条
│   │   ├── cipher.csv            # 1576 条
│   │   ├── equation_numeric.csv  # 732 条
│   │   ├── equation_symbolic.csv # 823 条
│   │   ├── gravity.csv           # 1597 条
│   │   ├── numeral.csv           # 1576 条
│   │   └── unit_conv.csv         # 1594 条
│   └── cot/                      # 合成 CoT 训练数据（由 cot_synthesis/ 生成）
│       ├── bit_ops/              # 500 条 bit_ops CoT 样本
│       └── equation_symbolic/    # 500 条 equation_symbolic CoT 样本
├── baseline/
│   ├── train.py                  # LoRA 训练脚本（含 eval callback）
│   ├── eval.py                   # 评测模块（官方 metric，batch generate）
│   └── run_baseline.sbatch       # SBATCH 作业脚本（训练 + 自动打包）
├── cot_synthesis/                # CoT 合成数据生成脚本（详见 cot_synthesis/README.md）
│   ├── bit_ops.py                # 位操作任务 CoT 生成器
│   ├── equation_symbolic.py      # 符号方程任务 CoT 生成器
│   └── README.md                 # 生成脚本使用说明
├── output/                       # 训练产物输出目录
│   ├── baseline-lora/            # 正式训练 adapter 权重
│   ├── debug-lora/               # 调试训练 adapter 权重
│   └── submission.zip            # 比赛提交文件
├── scripts/
│   ├── download_model.sh         # 从 HuggingFace 下载模型
│   ├── setup_training_env.sh     # 安装训练依赖到 venv（幂等）
│   ├── build_submission.sh       # 将 adapter 打包为 submission.zip
│   └── submit.sh                 # 上传 submission.zip 到 Kaggle
├── logs/                         # 所有 SBATCH 作业的 .out / .err 日志
├── official-doc/                 # 比赛官方 notebook（metric、submission demo）
├── resource/                     # 社区参考 notebook（其他选手的训练方案，仅供参考）
└── venv/                         # Python 虚拟环境（python/3.11.5）
```

---

## 完整流程

### Step 1：下载模型

```bash
# 方式一：直接传入 token
bash scripts/download_model.sh hf_YOUR_TOKEN

# 方式二：通过环境变量
export HF_TOKEN=hf_YOUR_TOKEN
bash scripts/download_model.sh
```

模型下载至 `model/`，约 60GB，包含 13 个 safetensors 分片。

---

### Step 2：调试训练（验证流程）

用单卡 + 少量步数快速跑通，确认代码无误：

```bash
sbatch baseline/run_baseline.sbatch
```

当前 sbatch 脚本为正式训练配置（1× H100，1 epoch，max_length=512），跑完约 26 分钟。
查看日志：

```bash
tail -f logs/nemotron-baseline-<JOBID>.out
```

调试成功的标志（日志末尾）：
```
[done] Training complete.
Training exit code: 0
Done. Submission saved to: .../output/submission.zip
```

---

### Step 3：正式训练

直接提交 sbatch 脚本（已为正式训练配置）：

```bash
sbatch baseline/run_baseline.sbatch
```

训练完成后会自动调用 `build_submission.sh` 打包生成 `output/submission.zip`。

---

### Step 4：手动打包（可选）

也可以对任意已训练完成的 adapter 单独打包：

```bash
bash scripts/build_submission.sh output/baseline-lora output/
```

脚本会自动校验：
- `adapter_config.json` 是否存在
- LoRA rank 是否 ≤ 32（超出则报错退出）
- adapter 权重文件是否存在

---

### Step 5：提交到 Kaggle

```bash
export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxx
bash scripts/submit.sh output/submission.zip "your message"
```

查看历史提交记录：

```bash
source venv/bin/activate
export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxx
kaggle competitions submissions -c nvidia-nemotron-model-reasoning-challenge
```

**注意事项：**
- 每天最多 5 次提交机会
- kaggle 库需要 **2.0.0+** 才支持 `KGAT_` 格式 token（`setup_training_env.sh` 不安装它，需手动或由 `submit.sh` 自动安装）
- `adapter_config.json` 中的 `base_model_name_or_path` 为本地路径，不影响评测（vLLM 用自己的 base model 路径）

---

## Baseline 训练配置

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

### 数据集划分

数据按 7 个任务类别（`bit_ops` / `cipher` / `equation_numeric` / `equation_symbolic` / `gravity` / `numeral` / `unit_conv`）从各自 CSV 中进行分层采样（seed=42 固定）：

- **训练集**：每类前 400 条，共 2800 条
- **Eval pool**：每类剩余样本，共 6700 条（与训练集完全无交集）

具体划分如下：

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

### 训练中评测（Eval Callback）

| 参数 | 值 |
|------|----|
| eval_steps | 25（每 25 步触发一次） |
| eval_samples_per_class | 5（共 35 条） |
| eval_batch_size | 8（batch generate 加速） |
| max_new_tokens | 256 |
| 评测指标 | 官方 metric：`\boxed{}` 提取 + 数值 1% 容差 / 字符串精确匹配 |
| 日志 | TensorBoard（`eval/accuracy` + 各类别准确率） |

评测数据从 eval pool 中每类取前 5 条（共 35 条），与训练集完全独立，无数据泄漏。使用 batched `model.generate()` + left-padding 并行推理。

**注意**：当前训练结束后保存的是**最后一步的权重**（step 175），而非 eval 最佳的 checkpoint。如需最佳权重，需手动从 `checkpoint-100` 重新打包。后续可考虑加入 `load_best_model_at_end` 逻辑。

### LoRA 配置

| 参数 | 值 |
|------|----|
| rank | 32 |
| alpha | 16（scaling ratio = 0.5） |
| dropout | 0.05 |
| target_modules | `in_proj`, `out_proj`, `up_proj`, `down_proj` |
| 可训练参数 | ~880M / 32.5B（2.71%） |
| 模型加载方式 | `device_map="balanced"`（多卡模型并行，非 DDP） |

### 数据格式与监督 Loss

每条样本格式化为 ChatML 对话，**仅对 assistant 回复部分计算 loss**（user prefix 的 label 全部设为 -100，PyTorch 自动跳过）：

```
<|im_start|>user
{prompt}
Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`<|im_end|>
<|im_start|>assistant
\boxed{answer}<|im_end|>
← 只有这部分 token 参与 loss 计算 →
```

Loss 类型为标准 **Cross-Entropy next-token prediction**，目标是让模型在给定 prompt 后直接输出正确的 `\boxed{answer}`。chat template 的 `<think></think>` 前缀已被主动 strip，训练目标中不含任何 think token。

---

## 比赛提交规范

提交文件为 `submission.zip`，评测时由 vLLM 加载 LoRA adapter 进行推理：

| 评测参数 | 值 |
|----------|----|
| max_lora_rank | 32 |
| max_tokens | 7680 |
| temperature | 0.0 |
| top_p | 1.0 |
| max_model_len | 8192 |
| gpu_memory_utilization | 0.85 |

zip 必须包含的文件：
- `adapter_config.json`（必须，且 rank ≤ 32）
- `adapter_model.safetensors`（adapter 权重）

---

## 依赖包（集群离线安装）

| 包 | 版本 | 用途 |
|----|------|------|
| torch | 2.5.1 | 深度学习框架（由 mamba-ssm 依赖版本决定） |
| transformers | 5.3.0 | 模型加载与训练 |
| peft | 0.18.1 | LoRA adapter |
| accelerate | 1.13.0 | 训练加速 |
| mamba-ssm | 2.2.4 | Mamba 架构 CUDA kernel |
| causal-conv1d | 1.5.4 | mamba-ssm 依赖 |
| tensorboard | 2.20.0 | 训练/评测指标可视化 |
| kaggle | 2.0.0 | 提交工具（需 2.0.0+ 以支持 KGAT_ token） |

安装顺序：**先装 `mamba-ssm`，再装其余包**（保证 torch 版本由 mamba-ssm 依赖决定，避免 ABI 不兼容）。

---

## 训练结果记录

| Job ID | 配置摘要 | 训练时长 | 最终 loss | 最佳 Eval | Kaggle 分数 |
|--------|----------|----------|-----------|-----------|-------------|
| 10934369 | 6类×200, max_len=1024, batch=1, grad_acc=8 | ~49 min | 4.507 | — | 0.61 |
| 10961113 | 6类×200, max_len=512, batch=4, grad_acc=4, NEFTune | ~26 min | 2.233 | — | 0.62 |
| 11061119 | 7类×400, eval callback, 逐条 generate | 超时(2h) | — | — | — |
| 11072751 | 7类×400, batch eval(bs=4), 2h | 超时(2h) | — | — | — |
| **11088934** | **7类×400, batch eval(bs=8), 4h** | **3h20m ✓** | **2.118** | **7/35 (20%) @ step 100** | **提交中** |

### Job 11088934 详细记录（首次完整 1 epoch）

**Loss 曲线：**

| Step | Epoch | Loss | LR |
|------|-------|------|----|
| 10 | 0.06 | 3.653 | 2.00e-4 |
| 20 | 0.11 | 2.984 | 1.98e-4 |
| 30 | 0.17 | 2.115 | 1.93e-4 |
| 40 | 0.23 | 2.245 | 1.84e-4 |
| 50 | 0.29 | 2.132 | 1.73e-4 |
| 60 | 0.34 | 2.094 | 1.59e-4 |
| 70 | 0.40 | 1.832 | 1.42e-4 |
| 80 | 0.46 | 2.027 | 1.24e-4 |
| 90 | 0.51 | 1.811 | 1.06e-4 |
| 100 | 0.57 | 1.812 | 8.68e-5 |
| 110 | 0.63 | 1.874 | 6.84e-5 |
| 120 | 0.69 | 1.900 | 5.11e-5 |
| 130 | 0.74 | 1.906 | 3.56e-5 |
| 140 | 0.80 | 2.022 | 2.23e-5 |
| 150 | 0.86 | 1.903 | 1.19e-5 |
| 160 | 0.91 | 1.804 | 4.55e-6 |
| 170 | 0.97 | 2.081 | 6.44e-7 |

**Eval 准确率（每 25 步，每类 5 条，共 35 条）：**

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

**关键观察：**
- **numeral** 类稳定满分（5/5），该类任务已被充分学习
- **其他 6 类准确率接近 0**，偶有零星正确（step 100 时 bit_ops、unit_conv 各答对 1 题）
- Loss 从 3.65 降至 ~1.8-2.0，模型在学习，但 answer-only 训练（无 CoT 推理过程）对推理型任务效果有限
- 最佳 eval 出现在 **step 100（epoch 0.57）**，之后略有下降，可能存在轻微过拟合
- Eval 样本量较小（每类仅 5 条），结果存在一定波动

**保存的 checkpoint：**
- `checkpoint-100` — 最佳 eval（7/35 = 20%），已用于 Kaggle 提交
- `checkpoint-175` — 最后一步
- `adapter_model.safetensors` — 最后一步权重（Trainer 默认保存）

---

### 已知兼容性问题与修复

| 问题 | 修复方式 |
|------|----------|
| `mamba-ssm` CUDA 扩展与 torch 2.9+ ABI 不兼容 | 先安装 mamba-ssm，让 pip 自动选择兼容的 torch 版本 |
| `mamba-ssm` 导入 `GreedySearchDecoderOnlyOutput`（transformers 5.x 已删除） | `train.py` 顶部注入 dummy 占位类 |
| `TrainingArguments` 不再支持 `no_cuda` 参数 | 已删除该参数 |
| `warmup_steps` 固定值难以适配不同步数 | 改用 `warmup_ratio=0.05`（自适应） |
| max_length=1024 导致 GPU 利用率仅 45% | 序列均值 168 token（p99=312），改为 max_length=512 后利用率显著提升，吞吐翻倍 |
| batch_size=8 触发 OOM（H100 80G） | 模型权重 + 优化器状态已占 ~78G，改为 batch_size=4 + grad_acc=4 维持等效 batch=16 |
| 逐条 eval generate 导致 2h 超时 | 35 条 × ~190s/条 ≈ 110 分钟。改为 batch generate（bs=4）并行推理加速 |
| batch eval(bs=4) 仍然较慢（~30min/轮） | 提升 eval_batch_size 至 8，每轮 eval 降至 ~17min（提速 47%） |
| TensorBoard 日志目录为空 | `logging_dir` 被 Trainer 忽略，实际写入 `output_dir/runs/`。已创建符号链接 |
| GPU 利用率稳定 ~50% | 30B MoE 模型仅 3B 参数活跃，单卡顺序执行各层，属正常现象 |

---

## 下一步计划

1. **CoT 数据训练**：将 `data/cot/` 中的 1000 条合成 CoT 样本加入训练，让模型学习推理过程而非仅记忆答案
2. **增加训练轮次**：当前只训练了 1 epoch，可尝试 2-3 epoch 观察是否继续改善
3. **Best model 自动保存**：在 TrainingArguments 中加入 `load_best_model_at_end` 逻辑
4. **增大 eval 样本量**：当前每类仅 5 条，波动较大，可增至 10-20 条获得更稳定评估
