# cot_synthesis — 合成 CoT 训练数据

本目录包含三个脚本，用于为比赛中的推理任务生成带有**思维链（Chain-of-Thought, CoT）**的合成训练样本。每个样本均包含：

- 与比赛格式完全一致的 `prompt`
- 逐步推导的 CoT 解题过程（`reasoning`）
- 最终答案（`answer`）

生成的数据用于 `cot_training/` 的混合训练，与官方训练集共同构成训练数据。

---

## 数据总览

| 类别 | 脚本 | 已生成样本数 | 难度级别 | CoT 风格 |
|------|------|-------------|---------|---------|
| bit_ops | `bit_ops.py` | 1000 | Level 1-3 | 侦探式假设验证 |
| equation_symbolic | `equation_symbolic.py` | 1000 | Level 1-4 | 结构化多步推理 |
| cipher | `cipher.py` | 1000 | Level 1-3 | 密码分析式推理 |
| **总计** | | **3000** | | |

生成数据存储于 `data/cot/{task}/`，按难度级别拆分为 `level_*.jsonl`。

---

## 脚本详解

### `bit_ops.py` — 位操作任务

**任务结构**：给定若干组"输入二进制数 -> 输出二进制数"的示例，要求推断隐藏的变换规则，并对新输入预测输出。

#### 支持的原子操作（17 种参数化变体）

| 类别 | 操作 | 说明 |
|------|------|------|
| 移位 | `shift_left_1/2/3`, `shift_right_1/2/3` | 逻辑左/右移 1-3 位 |
| 循环 | `rotate_left_1/2/3`, `rotate_right_1/2/3` | 循环左/右移 1-3 位 |
| 取反 | `bitwise_not` | 按位取反 |
| 掩码 | `bitwise_xor`(5 种), `bitwise_and`(3 种), `bitwise_or`(3 种) | 与随机掩码做位运算 |
| 排列 | `reverse_bits`, `swap_nibbles`, `swap_adjacent_pairs`, `swap_adjacent_bits` | 位序重排 |

掩码通过 `_sample_diverse_masks()` 生成，避免 `0x00`、`0xFF` 等平凡情况。多操作组合时使用 `_sample_compatible_ops()` 避免自消操作（如连续两次 NOT）。

#### 课程难度

| Level | 操作数 | 说明 |
|-------|--------|------|
| 1 | 1 | 单个原子操作 |
| 2 | 2 | 两个操作的复合 |
| 3 | 3 | 三个操作的复合 |

#### CoT 推理模板（9 步侦探式假设验证）

```
Step 1: 初始观察 — 展示 input/output 对（spaced binary 格式）
Step 2: Bit Count 分析 — 统计 Hamming weight / parity 变化，启发式剪枝
Step 3-5: 测试错误假设 — 逐一验证 XOR / NOT / rotation 等常见猜测，
         用具体反例说明为何失败
Step 6: 深度分析 — 发现 2-3 步操作链，系统化搜索
Step 7: Code-as-Thought — 将规则表达为可执行 Python 代码
Step 8: 完整验证 — 在所有示例上逐条验证（带 checkmark）
Step 9: 应用规则 — 对查询输入逐步执行操作链，得出答案
```

#### 使用方式

```bash
python cot_synthesis/bit_ops.py \
    --output_dir  data/cot/bit_ops \
    --num_samples 500 \
    --levels      1 2 3 \
    --seed        42
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | `data/cot/bit_ops` | 输出目录 |
| `--num_samples` | `500` | 生成样本总数（均匀分配到各 level） |
| `--levels` | `1 2 3` | 课程难度级别 |
| `--min_examples` | `6` | 每道题最少示例组数 |
| `--max_examples` | `10` | 每道题最多示例组数 |
| `--seed` | `42` | 随机种子 |

---

### `equation_symbolic.py` — 符号方程任务

**任务结构**：方程形如 `XY OP ZW = RESULT`，其中符号到数字的映射和运算符语义均被隐藏，要求从多组示例中同时推断两层隐含规则并求解目标方程。

#### 两层隐含规则

**1. 符号 -> 数字映射**

符号池：`! @ # $ % ^ & * ( ) [ ] { } | < > ? / \ ` ~ ' _ : ;`（排除了逗号、引号、换行等 CSV 冲突字符）。每道题随机选取 6-10 个符号映射到数字 0-9。

**2. 运算符语义**（每道题有 1-3 个运算符，各自独立选取语义）

| 语义 | 操作 | 示例 |
|------|------|------|
| `concat` | 字符串拼接 | `XY OP ZW = XYZW` |
| `rev_concat` | 反向拼接 | `XY OP ZW = ZWXY` |
| `add` | 数字相加 | 映射后相加，结果反编码为符号 |
| `sub` | 数字相减 | 映射后相减 |
| `mul` | 数字相乘 | 映射后相乘 |

运算符字符分为标准（`+ - * /`）和 exotic（`@ # $ % ^ & !`），exotic 概率由 `--exotic_op_prob` 控制。

#### 课程难度

| Level | 运算符数 | 语义组合 | 说明 |
|-------|---------|---------|------|
| 1 | 1-2 | 全部 string（concat/rev_concat） | 无需数值推理 |
| 2 | 2 | 1 numeric + 1 string | 需要推断符号映射 |
| 3 | 2-3 | 2 种不同 numeric | 交叉约束求解 |
| 4 | 3 | 混合（至少 2 种 numeric） | 最难 |

#### CoT 推理模板（结构化多步推理）

为防止模型过拟合到固定模板，推理文本使用**短语库（Phrase Banks）**增加句式多样性：每个推理阶段有 4-5 种不同的开头表述，随机选取。

```
Phase 1: 观察结构 — 识别不同运算符，列出方程示例
Phase 2: 解码 string 运算符 — 测试 concat/rev_concat 假设
Phase 3: 解码 numeric 运算符
  3a. 结构抖动（Structural Jitter）
      30% 概率: 直觉跳过（genius skip）
      40% 概率: 1 次错误猜测后修正
      30% 概率: 2 次错误猜测后修正
  3b. 结果长度启发式 — 区分 add/sub/mul
  3c. 代数约束推导 — 用变量 A/B/C 表示未知数，逐步求解映射
  3d. Code-as-Thought — 代码验证 + 跨方程交叉检查
Phase 4: 规则总结 — 展示最终映射表和运算符定义
Phase 5: 求解查询 — 逐步应用已学规则
```

"结构抖动"机制确保 CoT 中的错误假设路径不千篇一律，防止模型学到"第一次猜测一定是错的"这种伪模式。

#### 使用方式

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
| `--output_dir` | `data/cot/equation_symbolic` | 输出目录 |
| `--num_samples` | `500` | 生成样本总数 |
| `--levels` | `1 2 3 4` | 课程难度级别 |
| `--min_symbols` | `6` | 每道题最少不同符号数 |
| `--max_symbols` | `10` | 每道题最多不同符号数 |
| `--exotic_op_prob` | `0.3` | exotic 运算符字符概率 |
| `--seed` | `42` | 随机种子 |

---

### `cipher.py` — 密文解密任务

**任务结构**：给定若干组"密文 -> 明文"的示例，要求推断字母替换映射规则，并对新的密文进行解密。密文仅对小写字母进行替换，空格和词边界保持不变。

#### 支持的密码类型

| 类型 | 说明 |
|------|------|
| **Caesar 密码** | 所有字母按固定偏移量平移（shift=1-25），如 shift=7 时 a->h, b->i |
| **随机单表替换密码** | 字母表的 derangement（每个字母映射到不同的字母），通过 rejection sampling 生成 |

Derangement 保证没有字母映射到自身（即不存在 `a->a` 的情况），确保密码非平凡。

#### 课程难度

| Level | 密码类型 | 示例数 | 说明 |
|-------|---------|--------|------|
| 1 | Caesar | 4-6 | 最简单，只需发现偏移量 |
| 2 | 随机替换 | 5-6 | 证据充分，需构建完整映射表 |
| 3 | 随机替换 | 3-4 | 证据较少，难度最大 |

#### 词库与短语模板

使用 **Alice's Wonderland** 主题词汇，确保覆盖全部 26 个字母：

| 词类 | 数量 | 示例 |
|------|------|------|
| 名词（主语） | 20 | alice, cat, dragon, hatter, queen, rabbit... |
| 动词 | 20 | chases, creates, discovers, explores, follows... |
| 形容词 | 20 | ancient, clever, curious, golden, magical... |
| 名词（宾语） | 30 | book, castle, forest, garden, key, mirror, palace... |
| 介词 | 7 | in, inside, near, through, under, above, beyond |

**短语模板**按长度分三档：
- **SHORT**（3 词）：SVO（主谓宾）
- **MEDIUM**（3-4 词）：SVO / SVPO / "the adj S V" / "S V the O"
- **LONG**（4-5 词）：SVPO / "the adj S V" / "S V the adj O"

Level 1 使用 SHORT/MEDIUM 查询，Level 3 使用 LONG 查询以增加需要解密的字母数。

#### 字母覆盖验证

生成时确保查询文本中的所有字母都能从示例对中推导出映射（`query_letters.issubset(covered_letters)`）。如果覆盖不足则重新生成整道题。

#### CoT 推理模板

根据密码类型使用不同的推理路径：

**Caesar 密码（`_generate_cot_caesar`，8-9 步）：**
```
1. 初始观察 — 描述示例对
2. 提取字母映射 — 从前 1-2 个示例逐字母对齐
3. 测试错误假设: Atbash 密码 — 用互补映射测试，失败
4. [可选] 错误偏移弯路 — 尝试常见偏移（ROT13、ROT7 等），得到乱码
5. 计算偏移量 — 从字母对中推算实际 shift 值
6. 在第二个示例上验证 — 确认偏移量恒定
7. Code-as-Thought — Python 解密函数 + shift 参数
8. 完整验证 — 在所有示例上验证
9. 逐词解密查询 — 得出答案
```

**随机替换密码（`_generate_cot_substitution`，8-9 步）：**
```
1. 初始观察 — 列出示例
2. 提取字母映射 — 从前 3 个示例逐字母对齐，详细列出
3. [可选 50%] 提前解密弯路 — 用不完整映射尝试解密，出现 '?' 未知字母，
   意识到需要更多示例
4. 测试错误假设: Caesar — 发现 shift 不恒定，排除
5. 构建完整替换表 — 汇总所有已提取的 enc->plain 映射
6. 一致性验证 — 确认同一密文字母在多个示例中映射一致
7. Code-as-Thought — Python 映射字典 + 解密函数
8. 完整验证 — 在所有示例上验证
9. 逐词解密查询 — 得出答案
```

**弯路生成器（Self-Correction Paths）：**
- `_generate_wrong_shift_detour()`：尝试错误偏移（如 ROT13），展示解密结果为乱码
- `_generate_premature_decrypt_detour()`：用不完整映射提前解密，遇到未知字母后回退

这些弯路增加了 CoT 的多样性，让模型学会自我纠错而非一步到位。

#### CoT 深度分布

每个样本的推理深度随机分配：
- **direct**：无推理（仅答案）
- **light**：简要映射提取（20-30 行）
- **full**：完整密码分析（200-300 行）

分配概率由 `_assign_depth(level, rng)` 根据 level 动态调整。

#### 使用方式

```bash
python cot_synthesis/cipher.py \
    --output_dir  data/cot/cipher \
    --num_samples 500 \
    --levels      1 2 3 \
    --seed        42
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | `data/cot/cipher` | 输出目录 |
| `--num_samples` | `500` | 生成样本总数 |
| `--levels` | `1 2 3` | 课程难度级别 |
| `--min_examples` | `3` | 每道题最少示例组数 |
| `--max_examples` | `6` | 每道题最多示例组数 |
| `--seed` | `42` | 随机种子 |

---

## 输出格式

三个脚本均输出到指定目录，生成以下文件：

| 文件 | 内容 |
|------|------|
| `{task}_cot.jsonl` | 完整数据（含 `prompt`, `answer`, `cot_answer`, `reasoning`, `level` 等字段） |
| `{task}_cot.csv` | CoT 格式 CSV（`id`, `prompt`, `answer`），answer 含推理过程 |
| `{task}_direct.csv` | 与原始训练集格式一致的 CSV（`id`, `prompt`, `answer`），仅直接答案 |
| `level_{n}.jsonl` | 按难度级别拆分的 JSONL（`cot_training/` 训练时加载此格式） |
| `stats.json` | 生成统计信息（各 level 分布、操作类型等） |

cipher.py 额外生成 `cipher_think.csv`（`enable_thinking` 格式）。

---

## 集成到训练

生成的 `level_*.jsonl` 文件被 `cot_training/dataset.py` 的 `MixedDataset` 直接加载：

```python
# dataset.py 中的加载逻辑
for jsonl_file in sorted(Path(cot_data_dir / category).glob("level_*.jsonl")):
    for line in jsonl_file:
        sample = json.loads(line)
        # 使用 sample["prompt"], sample["cot_answer"], sample["answer"]
```

无需手动合并到 `data/by_category/`。混合训练会自动将 CoT 样本与普通样本 shuffle 混合。

---

## 设计原则

### 错误假设 + 自我纠正

所有 CoT 模板都包含**错误假设路径**：先尝试一个看似合理但错误的猜测，用示例反驳后再发现正确规则。这教会模型：
- 推理不是线性的，试错是正常的
- 每一步需要用证据验证
- 发现错误后如何调整方向

### Code-as-Thought

每种任务的 CoT 都包含 Python 代码块（代码即思考），将发现的规则形式化为可执行代码并验证。这利用了 NemotronH 模型的代码理解能力，帮助模型将抽象规则转化为精确的操作步骤。

### 句式多样性

`equation_symbolic.py` 和 `cipher.py` 使用**短语库（Phrase Banks）**为每个推理阶段提供多种不同的开头表述，避免模型过拟合到固定的语言模板。`bit_ops.py` 通过 spaced binary 格式和 Hamming weight 分析提供结构多样性。

### 覆盖全部字母 / 避免平凡情况

- cipher：derangement 保证无自映射字母，字母覆盖验证保证可解性
- bit_ops：掩码避免 `0x00`/`0xFF`，兼容操作选择避免自消
- equation_symbolic：排除 CSV 冲突字符，运算符与符号不重叠
