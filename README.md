# NVIDIA Nemotron-3-Nano-30B LoRA Fine-tuning

Kaggle 比赛：[NVIDIA Nemotron 3 Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)

对 `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` 进行 LoRA 微调，提交符合比赛格式的 adapter。

---

## 项目结构

```
NemotronFT/
├── model/                        # 下载的基础模型权重（~60GB BF16）
├── data/
│   ├── train.csv                 # 训练集（~69k 条，含 id/prompt/answer）
│   └── test.csv                  # 测试集（34 条）
├── baseline/
│   ├── train.py                  # LoRA 训练脚本
│   └── run_baseline.sbatch       # SBATCH 作业脚本（训练 + 自动打包）
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

当前 sbatch 脚本为调试配置（1× H100，10 步，max_length=512），跑完约 2 分钟。
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

调试通过后，修改 `baseline/run_baseline.sbatch` 为正式配置：

```bash
# 修改以下 SBATCH 参数：
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G

# 修改训练参数：
python baseline/train.py \
    --output_dir   output/baseline-lora \
    --max_steps    500 \
    --batch_size   1 \
    --grad_acc     8 \
    --max_length   2048 \
    --lora_rank    32 \
    --lora_alpha   64
```

然后重新提交：

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

| 参数 | 调试值 | 正式值 |
|------|--------|--------|
| GPU | 1× H100 | 2× H100 |
| 时间限制 | 1h | 4h |
| 内存 | 80G | 120G |
| max_steps | 10 | 500 |
| batch_size | 1 | 1 |
| grad_acc | **1**（等效 batch 1） | **8**（等效 batch 8） |
| max_length | **512**（长样本被截断） | **2048** |
| 学习率 | 2e-4 | 2e-4，cosine decay |
| warmup_steps | 50 | 50 |
| 优化器 | AdamW Fused | AdamW Fused |
| 精度 | BF16 | BF16 |

**调试 vs 正式的核心差异**（除 step 数以外）：

- `grad_acc`：调试为 1，正式为 8——正式训练的等效 batch size 是调试的 8 倍，梯度估计更稳定
- `max_length`：调试为 512，超长样本会被截断导致 answer 残缺；正式为 2048，能完整覆盖绝大多数样本
- GPU 数量：调试单卡，正式双卡（`device_map="balanced"` 自动均分）

**数据覆盖情况**（训练集约 69k 条，batch_size=1）：

| 配置 | 实际消费样本数 | 占全集比例 |
|------|-------------|-----------|
| 调试 10 步 | 10 条 | ~0.01% |
| 正式 500 步 | 500 条 | ~0.7% |

完整跑一个 epoch 需约 8625 步（grad_acc=8）。当前 500 步仅为快速验证，**并非完整训练**。

### LoRA 配置

| 参数 | 值 |
|------|----|
| rank | 32 |
| alpha | 64 |
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
<think></think>\boxed{answer}<|im_end|>
← 只有这部分 token 参与 loss 计算 →
```

Loss 类型为标准 **Cross-Entropy next-token prediction**，目标是让模型在给定 prompt 后直接输出正确的 `\boxed{answer}`。其中 `<think></think>` 是 `enable_thinking=False` 时 chat template 自动插入的空思考块。

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
| kaggle | 2.0.0 | 提交工具（需 2.0.0+ 以支持 KGAT_ token） |

安装顺序：**先装 `mamba-ssm`，再装其余包**（保证 torch 版本由 mamba-ssm 依赖决定，避免 ABI 不兼容）。

### 已知兼容性问题与修复

| 问题 | 修复方式 |
|------|----------|
| `mamba-ssm` CUDA 扩展与 torch 2.9+ ABI 不兼容 | 先安装 mamba-ssm，让 pip 自动选择兼容的 torch 版本 |
| `mamba-ssm` 导入 `GreedySearchDecoderOnlyOutput`（transformers 5.x 已删除） | `train.py` 顶部注入 dummy 占位类 |
| `TrainingArguments` 不再支持 `no_cuda` 参数 | 已删除该参数 |
| `warmup_ratio` 在 transformers 5.3 中废弃 | 改用 `warmup_steps` |
