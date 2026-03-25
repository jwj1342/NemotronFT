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

当前 sbatch 脚本为正式训练配置（1× H100，1 epoch，max_length=1024），跑完约 30 分钟。
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
| 时间限制 | 2h |
| 内存 | 80G |
| 数据采样 | 分层采样：6 类别各 200 条，共 1200 条 |
| max_steps | -1（完整 1 epoch，约 150 步） |
| batch_size | 1，grad_acc=8（等效 batch 8） |
| max_length | 1024 |
| 学习率 | 2e-4，cosine decay |
| warmup | `warmup_ratio=0.05`（总步数的 5%，约 8 步） |
| 优化器 | AdamW Fused |
| 精度 | BF16 |
| NEFTune | noise_alpha=5.0 |

**数据采样策略**：按任务类别（`bit_ops` / `cipher` / `numeral` / `gravity` / `symbol` / `unit_conv`）分层抽取，每类各 200 条，消除类别不均衡问题。

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
| kaggle | 2.0.0 | 提交工具（需 2.0.0+ 以支持 KGAT_ token） |

安装顺序：**先装 `mamba-ssm`，再装其余包**（保证 torch 版本由 mamba-ssm 依赖决定，避免 ABI 不兼容）。

### 已知兼容性问题与修复

| 问题 | 修复方式 |
|------|----------|
| `mamba-ssm` CUDA 扩展与 torch 2.9+ ABI 不兼容 | 先安装 mamba-ssm，让 pip 自动选择兼容的 torch 版本 |
| `mamba-ssm` 导入 `GreedySearchDecoderOnlyOutput`（transformers 5.x 已删除） | `train.py` 顶部注入 dummy 占位类 |
| `TrainingArguments` 不再支持 `no_cuda` 参数 | 已删除该参数 |
| `warmup_steps` 固定值难以适配不同步数 | 改用 `warmup_ratio=0.05`（自适应） |
