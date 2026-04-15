# distillation_synthesis

使用 teacher 模型（Stepfun step-3.5-flash）对原始训练集进行 CoT 蒸馏标注，生成可用于 SFT 的训练数据。

---

## 目录结构

```
distillation_synthesis/
├── distillation.py                # 核心：异步调用 teacher 生成 CoT，输出带 teacher_cot 的 CSV
├── export_distilled_dataset.py    # 将蒸馏 CSV 按 category 拆分为 distilled.jsonl
├── evaluation.py                  # 调用 teacher 对数据集进行评测
├── sample_train.py                # 从 by_category/ 按类均匀采样
├── llm.py                         # LLM 封装（StepLLM / SiliconflowLLM / ArkPlanLLM）
├── prompt.py                      # 蒸馏用系统/用户 prompt 模板
└── utils.py                       # prompt_format 工具函数

data/by_category/                  # 按类拆分的原始训练集（7 类）
├── bit_ops.csv
├── cipher.csv
├── equation_numeric.csv
├── equation_symbolic.csv
├── gravity.csv
├── numeral.csv
└── unit_conv.csv

data/distillation/                 # 蒸馏输出（按类拆分，训练 pipeline 直接读取）
├── bit_ops/distilled.jsonl
├── cipher/distilled.jsonl
├── ...
└── unit_conv/distilled.jsonl
```

---

## 数据流

```
data/by_category/*.csv
    └─ sample_train.py ──→ data/sampled-train.csv  (含 category 列)
                               └─ distillation.py ──→ data/sampled-train-distilled.csv
                                                           └─ export_distilled_dataset.py
                                                                  ──→ data/distillation/<category>/distilled.jsonl
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install openai python-dotenv tqdm
```

### 2. 配置 API Key

在 `distillation_synthesis/.env` 中写入（或导出为环境变量）：

```bash
STEP_API_KEY=your_step_api_key
```

### 3. 采样训练集

从 `data/by_category/` 按类均匀采样，每类最多 200 条，生成含 `category` 列的 CSV：

```bash
python distillation_synthesis/sample_train.py \
  --input-dir data/by_category \
  --output data/sampled-train.csv \
  --samples-per-category 200
```

### 4. 运行蒸馏

调用 teacher 模型为每道题生成 CoT，结果写入 CSV：

```bash
python distillation_synthesis/distillation.py \
  --input data/sampled-train.csv \
  --output data/sampled-train-distilled.csv \
  --concurrency 5
```

支持断点续跑（默认开启 `--resume`）。输出 CSV 包含以下列：

| 列名 | ���明 |
|------|------|
| `category` | 任务类别（来自 by_category 文件名） |
| `teacher_cot` | teacher 生成的逐步推理过程 |
| `teacher_response` | teacher 原始响应（JSON 字符串） |
| `teacher_finish_reason` | 结束原因（stop / length 等） |
| `teacher_prompt/completion/total_tokens` | token 用量 |
| `status` | `success` 或 `error` |
| `error` | 失败时的错误信息 |

### 5. 导出训练数据

将蒸馏 CSV 按 category 拆分为 `distilled.jsonl`，直接供 `distillation_training/` 读取：

```bash
python distillation_synthesis/export_distilled_dataset.py \
  --input data/sampled-train-distilled.csv \
  --output-dir data/distillation
```

输出 JSONL 每行格式：
```json
{"id": "...", "prompt": "...", "cot_answer": "reasoning...\n\n\\boxed{answer}", "answer": "..."}
```

---

## 其他脚本

### evaluation.py — 评测

使用 teacher 模型对数据集进行推理评测，输出 accuracy 统计：

```bash
python distillation_synthesis/evaluation.py \
  --input data/by_category/bit_ops.csv \
  --output data/eval_results.csv \
  --provider step \
  --concurrency 5 \
  --max-tokens 254000
```

---

## LLM 支持

`llm.py` 中封装了多个 teacher 模型，通过 `BaseLLM` 统一接口：

| 类名 | 模型 | API |
|------|------|-----|
| `StepLLM` | step-3.5-flash-2603 | Stepfun（`STEP_API_KEY`） |
| `SiliconflowLLM` | MiniMax-M2.5 | Siliconflow（`SILICONFLOW_API_KEY`） |
| `ArkPlanLLM` | ark-code-latest | Volces（`ARKPLAN_API_KEY`） |

切换 teacher 模型只需在 `distillation.py` 中替换 `StepLLM()` 为对应的类。
