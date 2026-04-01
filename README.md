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
│       ├── bit_ops/              # 1000 条 bit_ops CoT 样本
│       ├── cipher/               # 1000 条 cipher CoT 样本
│       └── equation_symbolic/    # 1000 条 equation_symbolic CoT 样本
├── baseline/                     # Baseline answer-only 训练（第一阶段，详见 baseline/README.md）
│   ├── train.py                  # LoRA 训练脚本（含 eval callback）
│   ├── eval.py                   # 评测工具库（官方 metric，被 cot_training 复用）
│   ├── run_baseline.sbatch       # SBATCH 作业脚本（1× H100, 4h）
│   └── README.md                 # Baseline 完整文档
├── cot_training/                 # CoT 混合训练（第二阶段，详见 cot_training/README.md）
│   ├── train.py                  # 训练脚本（regular SFT + CoT，per-token weighted loss）
│   ├── dataset.py                # MixedDataset：数据加载、tokenize、loss_weights 归一化
│   ├── weighted_trainer.py       # WeightedTrainer（decomposed loss logging）
│   ├── eval_vllm.py              # vLLM 评测脚本（Kaggle 对齐，~30x faster）
│   ├── run_cot_training.sbatch   # 训练作业脚本（2× H100, 24h）
│   ├── eval_vllm.sbatch          # vLLM 评测作业脚本（1× H100）
│   └── README.md                 # 混合训练完整文档
├── cot_synthesis/                # CoT 合成数据生成脚本（详见 cot_synthesis/README.md）
│   ├── bit_ops.py                # 位操作任务 CoT 生成器
│   ├── cipher.py                 # 密码任务 CoT 生成器
│   ├── equation_symbolic.py      # 符号方程任务 CoT 生成器
│   └── README.md                 # 生成脚本使用说明
├── output/                       # 训练产物输出目录
│   ├── baseline-lora/            # 第一阶段 baseline adapter 权重
│   ├── cot-lora/                 # 第二阶段 CoT adapter 权重
│   └── submission.zip            # 比赛提交文件
├── scripts/
│   ├── download_model.sh         # 从 HuggingFace 下载模型
│   ├── setup_training_env.sh     # 安装训练依赖到 venv（幂等）
│   ├── build_submission.sh       # 将 adapter 打包为 submission.zip
│   └── submit.sh                 # 上传 submission.zip 到 Kaggle
├── logs/                         # 所有 SBATCH 作业的 .out / .err 日志
├── official-doc/                 # 比赛官方 notebook（metric、submission demo）
├── resource/                     # 社区参考 notebook（其他选手的训练方案，仅供参考）
├── venv/                         # 训练用虚拟环境（torch 2.5.1 + mamba-ssm）
└── venv-vllm/                    # 推理用虚拟环境（vLLM 0.17.1 + torch 2.10.0）
```

---

## 快速开始

```bash
# 1. 下载模型（~60GB）
bash scripts/download_model.sh hf_YOUR_TOKEN

# 2. Baseline 训练（1× H100, ~26min）
sbatch baseline/run_baseline.sbatch

# 3. CoT 混合训练（2× H100, ~24h）
sbatch cot_training/run_cot_training.sbatch

# 4. vLLM 评测（1× H100, Kaggle 对齐）
sbatch cot_training/eval_vllm.sbatch

# 5. 打包提交
bash scripts/build_submission.sh output/cot-lora/checkpoint-XXX output/
bash scripts/submit.sh output/submission.zip "description"
```

**注意**：每天最多 5 次 Kaggle 提交。kaggle 库需要 2.0.0+ 才支持 `KGAT_` 格式 token。

---

## 第一阶段：Baseline

Answer-only LoRA 微调：7 类 × 400 条，1 epoch，max_length=512。模型学习直接输出 `\boxed{answer}`，不含思考过程。最佳 checkpoint-100 Kaggle 得分 **0.64**。

详见 **[baseline/README.md](baseline/README.md)**。

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

## 双 venv 架构

由于 vLLM 需要 torch 2.10+，而训练依赖 mamba-ssm 绑定 torch 2.5.1，两者不兼容，因此使用两个独立虚拟环境：

| venv | 用途 | 核心包 |
|------|------|--------|
| `venv/` | 训练 | torch 2.5.1, transformers 5.3.0, mamba-ssm 2.2.4, peft 0.18.1 |
| `venv-vllm/` | 推理（eval） | vLLM 0.17.1, torch 2.10.0, peft 0.18.1 |

训练 venv 安装顺序：**先装 `mamba-ssm`，再装其余包**（保证 torch 版本由 mamba-ssm 依赖决定，避免 ABI 不兼容）。

推理 venv 需要在激活前 `module load opencv/4.13.0`（vLLM 依赖）。

---

## 训练结果记录

| Job ID | 阶段 | 配置摘要 | 训练时长 | Kaggle 分数 |
|--------|------|----------|----------|-------------|
| 10934369 | Baseline | 6类×200, max_len=1024, batch=1 | ~49 min | 0.61 |
| 10961113 | Baseline | 6类×200, max_len=512, batch=4, NEFTune | ~26 min | 0.62 |
| **11088934** | **Baseline** | **7类×400, batch eval(bs=8), 4h** | **3h20m** | **0.64** |
| 11224xxx | CoT 混合 v1 | 7类×500 + CoT, 2×H100, 150/313步（超时） | 超时(8h) | **0.67** |

---

## 第二阶段：CoT 混合训练

从 baseline 最佳 checkpoint 出发，混合 7 类普通样本 + 3 类合成 CoT 样本进行继续训练，使用 per-token weighted loss 和样本级归一化。训练不含 in-training eval，改用训练后 vLLM 批量评测所有 checkpoint。

详细的数据格式、loss 设计、vLLM eval pipeline 等参见 **[cot_training/README.md](cot_training/README.md)**。

```bash
# 训练
sbatch cot_training/run_cot_training.sbatch

# 训练完成后评测所有 checkpoint（vLLM，每类 20 样本，7680 tokens）
sbatch cot_training/eval_vllm.sbatch

# 手动打包任意 checkpoint
bash scripts/build_submission.sh output/cot-lora/checkpoint-XXX output/
```

---

## 下一步计划

1. **完整 1 epoch 训练**：从头训练 ~313 步（24h job），每 20 步保存 checkpoint，训练后 vLLM 批量 eval
2. **扩展 CoT 覆盖范围**：为 bit_ops、equation_symbolic 改进 CoT 数据质量（当前 eval 仍为 0%）
3. **排行榜差距**：当前 0.67 vs 榜首 0.82，差距 0.15
