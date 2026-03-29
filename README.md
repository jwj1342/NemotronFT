# NVIDIA Nemotron-3-Nano-30B LoRA Fine-tuning

Kaggle 比赛：[NVIDIA Nemotron 3 Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)

对 `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` 进行 LoRA 微调，提交符合比赛格式的 adapter。

---

## 项目结构

```
NemotronFT/
├── model/                   # 基础模型权重（~60GB BF16）
├── data/
│   ├── train.csv            # 原始训练集（~69k 条）
│   ├── test.csv             # 测试集（34 条）
│   ├── by_category/         # 按 7 类别拆分的 CSV
│   └── cot/                 # 合成 CoT 训练数据
├── baseline/
│   ├── train.py             # LoRA 训练脚本（含 eval callback）
│   ├── eval.py              # 评测模块（batch generate + 官方 metric）
│   └── run_baseline.sbatch  # SBATCH 作业脚本
├── cot_synthesis/           # CoT 合成数据生成脚本
├── output/                  # 训练产物（adapter 权重 + submission.zip）
├── scripts/                 # 工具脚本（下载模型/环境安装/打包/提交）
└── logs/                    # SBATCH 日志
```

---

## 快速开始

```bash
# 1. 下载模型
bash scripts/download_model.sh hf_YOUR_TOKEN

# 2. 训练
sbatch baseline/run_baseline.sbatch

# 3. 打包（训练完会自动打包，也可手动）
bash scripts/build_submission.sh output/baseline-lora output/

# 4. 提交
export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxx
bash scripts/submit.sh output/submission.zip "your message"
```

---

## 训练配置

| 参数 | 值 |
|------|-----|
| GPU | 1× H100 |
| 数据 | 7 类别各 400 条，共 2800 条（分层采样） |
| Epoch | 1（~175 步） |
| Batch | 4 × grad_acc 4 = 等效 16 |
| 序列长度 | 512 |
| 学习率 | 2e-4，cosine，warmup 5% |
| 精度 | BF16 |
| NEFTune | noise_alpha=5.0 |

### LoRA

| 参数 | 值 |
|------|-----|
| rank / alpha | 32 / 16 |
| dropout | 0.05 |
| target_modules | `in_proj`, `out_proj`, `up_proj`, `down_proj` |
| 可训练参数 | ~880M / 32.5B（2.71%） |

### 数据格式

训练采用 ChatML 格式，仅对 assistant 回复计算 loss（answer-only，无 CoT）。

---

## 比赛提交规范

提交 `submission.zip`，需包含 `adapter_config.json`（rank ≤ 32）和 `adapter_model.safetensors`。

评测参数：max_lora_rank=32, max_tokens=7680, temperature=0.0, max_model_len=8192。

---

## 依赖

核心依赖：torch 2.5.1, transformers 5.3.0, peft 0.18.1, mamba-ssm 2.2.4, tensorboard 2.20.0

> **注意**：先装 `mamba-ssm`，再装其余包，避免 torch ABI 不兼容。
