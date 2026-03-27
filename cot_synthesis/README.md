# cot_synthesis — 合成 CoT 训练数据

本目录包含两个脚本，用于为比赛中的推理任务生成带有**思维链（Chain-of-Thought, CoT）**的合成训练样本。每个样本均包含：

- 与比赛格式完全一致的 `prompt`
- 逐步推导的 CoT 解题过程（`reasoning`）
- 最终答案（`answer`）

生成的数据可直接用于 LoRA 微调，替代或补充原始训练集中缺乏推导过程的样本。

---

## 文件说明

### `bit_ops.py` — 位操作任务

生成 **bit_ops** 类别的 CoT 样本。

**任务结构**：给定若干组"输入二进制数 → 输出二进制数"的示例，要求推断隐藏的变换规则，并对新输入预测输出。

**支持的原子操作**：
- 逻辑左移 / 右移（`shift_left` / `shift_right`）
- 循环左移 / 右移（`rotate_left` / `rotate_right`）
- 按位取反（`bitwise_not`）
- 按位与 / 或 / 异或（`bitwise_and` / `bitwise_or` / `bitwise_xor`）
- 高低 4 位交换（`swap_nibbles`）

**课程难度**：
- Level 1：单个原子操作
- Level 2：两个操作的复合
- Level 3：三个操作的复合

**CoT 风格**：侦探式假设验证——先枚举候选规则，对每条规则用已知示例逐一验证，筛选出全部通过的规则，再应用于新输入。

---

### `equation_symbolic.py` — 符号方程任务

生成 **equation_symbolic** 类别的 CoT 样本。

**任务结构**：方程形如 `XY OP ZW = RESULT`，其中符号到数字的映射和运算符语义均被隐藏，要求从多组示例中同时推断两层隐含规则并求解目标方程。

**两层隐含规则**：
1. **符号→数字映射**：如 `! → 3`，`@ → 7`
2. **运算符语义**（每个运算符独立）：
   - `concat`：字符串拼接，`XY OP ZW = XYZW`
   - `rev_concat`：反向拼接，`XY OP ZW = ZWXY`
   - `add`：数字相加，结果按符号映射反编码
   - `sub`：数字相减
   - `mul`：数字相乘

**课程难度**：
- Level 1：仅字符串操作（concat/rev_concat），无需数值推理
- Level 2：一个数值运算符 + 一个字符串运算符
- Level 3：两个不同语义的数值运算符
- Level 4：三个运算符的混合（最难）

**CoT 风格**：结构化的多步骤推理——符号映射发现 → 运算符语义识别 → 验证 → 求解。

---

## 输出格式

两个脚本均输出到指定目录，每次运行生成两个文件：

| 文件 | 内容 |
|------|------|
| `{task}.csv` | 与原始训练集格式一致的 CSV（`id`, `prompt`, `answer`），可直接用于训练 |
| `{task}_with_cot.jsonl` | 附带完整 CoT 推导过程的 JSONL（额外含 `reasoning` 字段） |

---

## 使用方式

### bit_ops

```bash
python cot_synthesis/bit_ops.py \
    --output_dir  data/cot/bit_ops \
    --num_samples 500 \
    --levels      1 2 3 \
    --seed        42
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | `data/cot/bit_ops` | 输出目录（自动创建） |
| `--num_samples` | `500` | 生成样本总数 |
| `--levels` | `1 2 3` | 课程难度级别（可组合） |
| `--min_examples` | `3` | 每道题最少示例组数 |
| `--max_examples` | `5` | 每道题最多示例组数 |
| `--seed` | `42` | 随机种子 |

### equation_symbolic

```bash
python cot_synthesis/equation_symbolic.py \
    --output_dir    data/cot/equation_symbolic \
    --num_samples   500 \
    --levels        1 2 3 4 \
    --exotic_op_prob 0.3 \
    --seed          42
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | `data/cot/equation_symbolic` | 输出目录（自动创建） |
| `--num_samples` | `500` | 生成样本总数 |
| `--levels` | `1 2 3 4` | 课程难度级别（可组合） |
| `--min_symbols` | `3` | 每道题最少不同符号数 |
| `--max_symbols` | `6` | 每道题最多不同符号数 |
| `--exotic_op_prob` | `0.3` | 使用非标准运算符字符的概率（增加难度） |
| `--seed` | `42` | 随机种子 |

---

## 集成到训练

生成的 `{task}.csv` 与 `data/by_category/` 下的原始 CSV 格式相同，可直接传入 `baseline/train.py` 的数据加载逻辑，或合并到对应类别的原始 CSV 中扩充训练集。

```bash
# 合并到原始数据（可选）
cat data/cot/bit_ops/bit_ops.csv >> data/by_category/bit_ops.csv
cat data/cot/equation_symbolic/equation_symbolic.csv >> data/by_category/equation_symbolic.csv
```
