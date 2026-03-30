# 快速开始

## 1. 安装依赖

在项目根目录执行：

```bash
uv sync
```

如果你是直接在当前 `distillation/` 目录里运行，也可以单独安装这几个依赖：

```bash
pip install openai python-dotenv tqdm
```

## 2. 配置 API Key

在 `distillation/.env` 中写入：

```bash
STEP_API_KEY=your_step_api_key
```

## 3. 准备采样后的训练集

从 `data/train.csv` 生成 `data/sample1200-train.csv`：

```bash
python3 sample_train.py
```

这个脚本会按 `label` 每类随机采样 `200` 条，总计 `1200` 条数据。

## 4. 运行蒸馏

```bash
python3 distillation.py \
  --input data/sample1200-train.csv \
  --output data/sample1200-train-distilled.csv \
  --concurrency 5
```

默认行为如下：

- 读取 `data/sample1200-train.csv`
- 输出到 `data/sample1200-train-distilled.csv`
- 默认显示进度条
- 默认开启断点续跑
- 自动增加 `teacher_response`、`teacher_finish_reason`、`teacher_prompt_tokens`、`teacher_completion_tokens`、`teacher_total_tokens`、`teacher_cot`、`status`、`error` 等列
- 不会新增单独的 `reasoning` 存储列；蒸馏结果仍然只写 `teacher_cot`
- 模型返回 JSON 时只提取 `cot` 字段作为蒸馏结果，返回中的 `answer` 不参与写出和校验

## 5. 常用参数

查看完整参数说明：

```bash
python3 distillation.py --help
```

自定义参数示例：

```bash
python3 distillation.py \
  --input data/sample1200-train.csv \
  --output data/sample1200-train-distilled.csv \
  --concurrency 50 \
  --max-tokens 254000 \
  --resume \
  --temperature 0.1
```

Evaluation

```bash
python3 evaluation.py \
  --input data/evaluation_dataset.csv \
  --output data/evaluation_results.csv \
  --provider step \
  --concurrency 5 \
  --max-tokens 254000 \
  --resume \
  --temperature 0.1
```
