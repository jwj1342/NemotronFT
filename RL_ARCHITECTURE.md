# RL 训练架构指南

> 给 RL 新手。这份文档是 `opd` 分支工作的"全景说明书"——解释我们在做什么、为什么这么做、里面的数学、以及调试时该关注什么。配套阅读：`ROADMAP.md`（战略层）、`ITERATION_REPORT.md`（实测结果与为什么 RL 在 Huikang adapter 上没跑起来）。
>
> **注意（2026-06-02 更新）**：本指南描述的是 RL *设计与计划*。GRPO 在 Huikang adapter 上的实际结果是策略熵塌缩、无梯度信号 —— 详见 `ITERATION_REPORT.md`。**RL 当前已搁置**：现行主线是 `ROADMAP.md` §3 路线 A 的**答案过滤自训练（RFT/STaR）**——它是 RL 的离线稳定版本（同样用 verifier 信号，但不依赖在线 rollout 多样性，因此不受熵塌缩阻断）。本文档的价值在于：若 RFT 复用 rollout/采样基础设施，这里的 vLLM rollout、FSDP+PEFT 训练环、JSONL 解耦设计仍是参考。要真正复活在线 RL，gatekeeper 是改用 **base 模型**（非 adapter）测 rollout 多样性。

---

## 0. TL;DR

我们要做的事，一句话：**在 Kaggle Nemotron 推理挑战赛上把 LB 从 0.85 推到 0.88+**。

走的路径：
1. 在 Huikang 0.85 LoRA adapter 基础上
2. 用 **RL（GRPO 算法）** 单独训 cryptarithm 和 equation_numeric_guess 两类弱任务的 task-specific adapter
3. 用 **OPD（On-Policy Distillation）** 把它们 + anchor 类别的 trace 一起蒸馏到一个统一的 LoRA 上
4. 在 **多 node L40s 集群（HYBRID_SHARD）** 上跑，目标 1 周完成 RL 阶段

为什么不是 SFT？因为 SFT 学的是 teacher 的能力上限——GPT-4 在 cryptarithm 上也只有 25%。RL 直接用 verifier 当 reward，能脱离 teacher 上限。

---

## 1. 强化学习极简入门（给完全没接触过 RL 的人）

把 LLM 当作一个**策略（policy）**，符号写作 `πθ`。它的工作很简单：

```
输入：一段 prompt x
输出：一段文本 y（一个 token 接一个 token）
```

- `θ`：模型参数（在我们这里只有 LoRA 权重会更新，base 32B 冻结）
- `πθ(y|x)`：表示"给定 x 后产生 y 的概率"

RL 想做的事就一件：**找一组 θ，使期望 reward 最大**：

```
maximize  E[ R(y, x) ]    其中  x ~ data,  y ~ πθ(·|x)
```

`R(y, x)` 是 reward 函数。在我们的项目里：

```python
R(y, x) = 1  if verify(extract_final_answer(y), ground_truth(x))
       = 0  otherwise
```

**模型答对题目得 1 分，答错 0 分。这就是全部 reward**。没有人工标注，没有偏好模型，`verify()` 自己说了算（基于精确字符串比对 + 数值 1% 容差）。

### 1.1 为什么 RL 在 verifier 任务上比 SFT 强

| | SFT | RL |
|---|---|---|
| 数据 | teacher 给的正确答案 | 模型自己生成 + verify 打分 |
| 上限 | `min(teacher 能力, 模型容量)` | **模型容量**（不受 teacher 限制）|
| 信号方向 | "学这个就对了" | "你试一试，对了我告诉你" |

cryptarithm 这种"GPT-4 也只会 25%"的任务，SFT 学到 25% 就到顶了；RL 因为只看答案对不对，理论上能逼近模型的真实推理上限。

DeepSeek-R1、AlphaProof、AlphaGeometry 都是靠这个思路突破"teacher 也不会"的任务的。

### 1.2 几个必须熟悉的术语

- **Trajectory**：一次完整生成的 token 序列（一条答题路径）
- **Rollout**：模型采样生成（一次或多次 trajectory）
- **Policy gradient**：通过 `∇θ log πθ(y|x)` 调整参数的方向
- **Advantage**：一条 trajectory 相对 baseline 的"好多少 / 差多少"——决定 gradient 的强度和方向
- **Reference policy (π_ref)**：冻结的 baseline（我们用 Huikang 0.85 adapter）
- **KL divergence**：两个分布的距离，用来约束 πθ 不要漂得离 π_ref 太远

---

## 2. GRPO 算法（我们用的核心算法）

GRPO = **Group Relative Policy Optimization**。DeepSeek-R1 用的就是它。

### 2.1 一句话直觉

> 对同一道题，让模型采样 **G 条** 不同的答题路径。看哪些对、哪些错，**对的路径加强，错的路径削弱**。

这就是 GRPO 的全部精神。和 PPO 相比，GRPO **不需要训练一个 critic 网络**来估计"答案的好坏"——它直接用 group 内的相对排序来做这件事。

### 2.2 算法流程

```
For each prompt x in batch:
    Sample G trajectories y_1, ..., y_G  from πθ(·|x)  with T=1.0  (探索性采样)
    Compute rewards r_1, ..., r_G            (each is 0 or 1)

    # Group-relative advantage（核心步骤）
    μ = mean(r_1, ..., r_G)
    σ = std(r_1, ..., r_G) + 1e-8
    For i = 1..G:
        A_i = (r_i - μ) / σ

    # Policy gradient with KL anchor
    Loss = -Σ_i [ A_i · log πθ(y_i | x) ]  +  β · KL( πθ || π_ref )

Update θ.
```

直觉理解（举例）：
- 一个 group 内 G=8 条 trajectory，**2 条对 6 条错**，`μ=0.25, σ≈0.43`
- 对的两条 advantage = `(1 - 0.25) / 0.43 ≈ +1.7`
- 错的六条 advantage = `(0 - 0.25) / 0.43 ≈ -0.58`
- gradient 把对的两条概率推高 1.7 倍强度，把错的六条推低 0.58 倍强度

### 2.3 完整数学公式

```
                   ┌                              ┐
J_GRPO(θ) = E_x  E│ A_i · log πθ(y_i | x)         │ -  β · KL(πθ || π_ref)
                   └                              ┘
                  y_i

其中:
    A_i = (r_i - μ_group) / (σ_group + ε)
    μ_group = (1/G) · Σ_i r_i
    σ_group = sqrt( (1/G) · Σ_i (r_i - μ_group)² )
    r_i = R(y_i, x) = verify(extract_final_answer(y_i), gt(x)) ∈ {0, 1}
```

各符号含义：
- `θ`：当前训练中的模型参数（**只有 LoRA 部分会更新**，base 冻结）
- `π_ref`：参考策略 = Huikang baseline（冻结）。防止 θ 飘太远丢失通用能力
- `β`：KL penalty 强度，我们用 **0.05**（reward 信号强时可以小，弱时要大一点）
- `KL`：Kullback-Leibler 散度

### 2.4 KL penalty 怎么算（实现细节）

教科书的 KL 定义是 `KL(p||q) = Σ p log(p/q)`，需要在全 vocab 上求和（vocab 131k，太贵且只对每一步建模）。

实践中用 **Schulman k3 estimator**（GRPO/PPO 业界标准）：

```
KL(πθ || π_ref) ≈ E_y~πθ [ exp(Δ) - Δ - 1 ]    其中  Δ = log π_ref - log πθ
```

只需要每个 sampled token 的 log probability，不用枚举整个 vocab。这是一个**无偏且方差小**的估计。

### 2.5 为什么 GRPO 而不是 PPO？

PPO 需要训练一个 value head 估计 `V(x, y_partial)`，用 GAE 算 advantage。有几个问题：

1. value head 占额外显存（在 32B 模型上是 ~16GB/GPU）
2. value head 训练不稳定（特别是 sparse reward 0/1）
3. value head 是 critic，需要单独 backprop，loss landscape 复杂

**GRPO 用 group baseline 代替 critic**——"这 G 条里平均 reward 是多少"就是天然的 baseline。
省一个 critic 网络、显存减半、对 sparse reward 数值稳定。这正好适合我们 32B + 4×L40s 紧张的显存。

---

## 3. On-Policy Distillation（OPD）

### 3.1 为什么需要 OPD

假设我们对 cryptarithm 单独 RL 训了一个 adapter `A_crypto`，再对 equation_guess 单独 RL 训了一个 adapter `A_eqn`。**最后用哪个？**

- 用 `A_crypto` 提交 → `equation_guess` 性能退化（policy 被拉向了 cryptarithm 分布）
- 用 `A_eqn` 提交 → cryptarithm 性能退化
- 想同时上传两个？vLLM / Kaggle grader 只支持**一个 LoRA**。

我们需要**一个 adapter 同时擅长所有 task**，而且不能比 baseline 在强 task 上变差。

### 3.2 OPD 的思路

不直接合并 adapter 权重（太粗暴会互相干扰），而是：

1. 让每个 task-specific RL adapter **自己跑一遍它擅长的 task**，T=0 greedy 生成高质量 trace
2. 让 Huikang baseline 在强 task（anchor 类别）上也跑一遍，生成 anchor trace
3. 把所有 trace 混在一起，做一次 **SFT**，从 Huikang baseline 开始训
4. 得到的统一 adapter 兼具所有 task 的能力

### 3.3 为什么叫 "On-Policy"

普通蒸馏是用 teacher（比如 GPT-4）生成的 CoT 当训练数据 —— teacher 是 **off-policy**，因为它和 student 不是同一个模型，分布不同。SFT 时会有 distribution shift（学生模仿老师的语气、用词，但底层概率结构不一样）。

OPD 用的是**模型自己（或自己的 RL 改进版）生成的 trace**。这些 trace 与 student 同分布（**on-policy**），SFT 时不会有 distribution shift。

一句话直觉：**OPD 是"让 student 把自己的最好版本学回来"**。

### 3.4 我们的混合比例

| 类别 | 来源 | 占比 |
|---|---|---|
| cryptarithm_deduce | RL adapter T=0 trace, ×3 oversample | ~25% |
| cryptarithm_guess | RL adapter T=0 trace, ×3 oversample | ~10% |
| equation_numeric_guess | RL adapter T=0 trace, ×3 oversample | ~10% |
| bit_manipulation | Huikang baseline T=0 anchor | ~15% |
| equation_numeric_deduce | Huikang baseline T=0 anchor | ~10% |
| cipher / gravity / unit / numeral | Huikang baseline T=0 anchor | ~30% 合计 |

RL trace 重采 3 倍是因为：cryptarithm 那 ~600 题 RL 后能产出的正确 trace 可能只有 200-300 条，不重采会被 anchor 数据淹没。

### 3.5 防破坏的双保险

1. **RL 阶段**：β=0.05 的 KL penalty 到 Huikang baseline，policy 不能漂太远
2. **OPD SFT 阶段**：LR 1e-5 + 1 epoch + 每 50 步在 9500 训练集采样 200 题 eval。**任何 anchor 类别掉 >1% 立即 early stop**

---

## 4. 系统架构

### 4.1 流程总图

```
                          ┌─────────────────┐
                          │  train.csv      │
                          │  (9500 题)      │
                          └────────┬────────┘
                                   │ detect_category + 筛选
                                   ▼
                          ┌─────────────────┐
                          │  pool.jsonl     │  per-task prompt 池
                          │  {id, prompt,   │
                          │   gt, category} │
                          └────────┬────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
              ▼                                         │
    ┌─────────────────┐                                 │
    │ Rollout phase   │                                 │
    │ (vLLM,          │                                 │  iter K
    │  venv-vllm)     │                                 │  循环
    │  T=1.0, G=8     │                                 │
    │  load adapter @ │                                 │
    │  policy/iter_K  │                                 │
    └────────┬────────┘                                 │
             │ generate G trajectories per prompt       │
             ▼                                          │
    ┌─────────────────┐                                 │
    │ rollouts/       │                                 │
    │  iter_K.jsonl   │                                 │
    │ {prompt, group: │                                 │
    │   [(text, r,    │                                 │
    │     token_ids)]}│                                 │
    └────────┬────────┘                                 │
             │                                          │
             ▼                                          │
    ┌─────────────────┐                                 │
    │ Training phase  │                                 │
    │ (FSDP+PEFT,     │  forward πθ       → log_new    │
    │  venv)          │  forward (LoRA off) → log_ref  │
    │                 │  compute GRPO loss             │
    │                 │  backward, update LoRA         │
    └────────┬────────┘                                 │
             │ save adapter                             │
             ▼                                          │
    ┌─────────────────┐                                 │
    │ policy/         │─────────────────────────────────┘
    │  iter_K+1/      │
    └────────┬────────┘
             │ after N outer iters
             ▼
    ┌─────────────────┐
    │ Final RL adapter│  output/rl-<task>/
    └────────┬────────┘
             │ (Stage 4 OPD trace generation, T=0)
             ▼
    ┌─────────────────┐
    │ opd_traces.csv  │  混合 trace pool
    └────────┬────────┘
             │ SFT on Huikang baseline (LR=1e-5, 1 epoch)
             ▼
    ┌─────────────────┐
    │ output/opd-final│  最终统一 adapter
    └─────────────────┘
```

### 4.2 单 node 内的 GPU 分配（4×L40s FSDP FULL_SHARD）

```
Training phase (FSDP active, vLLM 已退出):

┌─── GPU 0 (48GB) ────┐   ┌─── GPU 1 (48GB) ────┐
│ Base model shard 0  │   │ Base model shard 1  │
│ ~16GB (frozen)      │   │ ~16GB (frozen)      │
│ LoRA shard ~1GB     │   │ LoRA shard ~1GB     │
│ Optim state ~2GB    │   │ Optim state ~2GB    │
│ Activations ~15GB   │   │ Activations ~15GB   │
│ ────────────────    │   │ ────────────────    │
│ Free ~14GB          │   │ Free ~14GB          │
└─────────────────────┘   └─────────────────────┘
GPU 2, 3: 同上

Rollout phase (vLLM active, FSDP 已退出):

┌─── GPU 0 (48GB) ────┐
│ Base model shard 0  │  (vLLM TP=4 切分)
│ ~16GB               │
│ LoRA adapter cache  │
│ ~1GB                │
│ KV cache buffer     │
│ ~17GB (依 G 和 seq) │
│ ────────────────    │
│ Reserved (mem util  │
│ 0.35 hard cap)      │
└─────────────────────┘

注意: vLLM gpu_memory_utilization = 0.35（不是 eval 默认的 0.85）
      因为同一 sbatch 内训练阶段需要这块 GPU
```

### 4.3 多 node 协作（HYBRID_SHARD）

假设 2 nodes × 4 L40s = 8 GPUs：

```
┌──────────── Node 0 ────────────┐    ┌──────────── Node 1 ────────────┐
│  GPU0  GPU1  GPU2  GPU3        │    │  GPU0  GPU1  GPU2  GPU3        │
│   │    │    │    │              │    │   │    │    │    │              │
│   └────┴────┴────┘              │    │   └────┴────┴────┘              │
│   FULL_SHARD                    │    │   FULL_SHARD                    │
│   (Base split 4-way 在 4 卡上)  │    │   (Base split 4-way 在 4 卡上)  │
│   Local replica A               │    │   Local replica B               │
└────────────────┬─────────────────┘    └────────────────┬────────────────┘
                 │                                       │
                 └────── DP gradient all-reduce ─────────┘
                         (只同步 LoRA 梯度, ~几 MB / iter)
```

**为什么这样设计**：
- L40s 没 NVLink，节点内走 PCIe 4.0 已经是最快的本地通信
- 节点间走 IB 或万兆网，比节点内慢一个数量级
- 全 mesh FSDP 跨 8 卡会让 base model 的 all-gather 经过慢的节点间链路 → 通信打死
- HYBRID_SHARD：把 base model 的传输**锁在节点内**（PCIe），只让小小的 LoRA 梯度跨节点（万兆网够用）
- LoRA 跨节点流量 ~几 MB，普通以太网即可，**完美避开 L40s 集群的网络短板**

**Rollout 阶段**：
- 每个 node 独立跑 vLLM 实例（TP=4 在节点内切分 base model）
- prompt pool 按 `prompt_id % N_nodes` 分片
- N 个 node = ~N 倍 rollout 吞吐量

**Adapter 同步**：
- training phase 写到共享 `/scratch/jwj/NemotronFT/training/rl/runs/<stage>/policy/iter_K+1/`
- 下一轮 rollout 所有 node 从同一路径 load
- Lustre 偶尔 eventually-consistent，写入后 `sync; sleep 5` 再触发下一轮

---

## 5. 关键技术细节

### 5.1 Reference policy via `disable_adapter()`（核心 trick）

Naive 实现：开两份 32B base model（policy + ref），每张 GPU 多占 16GB。**不可行**。

我们的 trick：base model 只加载**一份**，LoRA 是"插件"，可以临时禁用：

```python
# Policy forward (with LoRA active)
log_new = model(input_ids).log_softmax(-1).gather(-1, target_ids)

# Reference forward (with LoRA disabled — base only)
with model.disable_adapter():
    log_ref = model(input_ids).log_softmax(-1).gather(-1, target_ids)

# 同一份 base 权重，仅 LoRA toggled. 省 ~16GB/GPU.
```

PEFT 的 `disable_adapter()` 是 context manager，进入时把 LoRA 模块短路（W → W），退出时恢复（W → W + ΔW_LoRA）。

**FSDP `use_orig_params=True` 下应该正确路由**——这是 G0 smoke test 必查项。如果 PEFT + FSDP 这个组合有 bug，fallback 是真的加载第二份 base（双 base 在 rollout/train 解耦下也能放下，只是费显存）。

### 5.2 vLLM 与 FSDP 显存分时复用

vLLM 和 FSDP 不能同时占同一张 GPU。我们的策略：**在同一个 sbatch allocation 内，sequential 切换**：

```bash
# Phase 1: rollout (vLLM 启动并独占 GPU)
python -m training.rl.rollout ...
# vLLM 进程退出，OS 回收显存

# 等显存清空
while [ $(nvidia-smi ... | awk '{print $1}') -gt 1000 ]; do
    sleep 5
done

# Phase 2: training (FSDP 启动)
accelerate launch training/rl/train_grpo.py ...
# 训练完保存 adapter，FSDP 进程退出

# 下一轮 outer iter
```

为什么这样比 in-process co-location 简单：vLLM 和 PyTorch FSDP 各自的 memory allocator 状态不会互相干扰；崩溃域小；可以独立调试。

**代价**：每轮 outer iter 多出 vLLM init (~30s) + FSDP init (~60s) 的额外开销。在 outer iter 几十分钟的尺度上完全可接受。

### 5.3 Group 归一化在 JSONL 里预算好

GRPO 的 advantage 是"组内相对"——理论上要保证训练时 batch 包含完整 group。但**我们在 rollout JSONL 里已经预先算好了每条 trajectory 的 `advantage`**（在 `reward.py` 的 `group_advantages()` 里）。

```jsonl
{"prompt_id": "abc", "category": "cryptarithm_deduce",
 "group": [
   {"text": "...", "reward": 1, "advantage": 1.7, "token_ids": [...]},
   {"text": "...", "reward": 0, "advantage": -0.58, "token_ids": [...]},
   ...
 ]}
```

这意味着训练时**不需要保持 group 完整**——advantage 已经反映了组内相对关系，batch_size=1 + grad_acc=16 就够，不需要凑 G=8 的整数倍。

---

## 6. 训练流程总览（stage by stage）

```
Stage 0 (smoke, 2h × 4 GPU 单 node)
├── 目标: 验证 pipeline 不崩
├── 关键验证: rollout JSONL 合法、disable_adapter 真有用、GRPO loss 有限
└── 输出: runs/stage0/SMOKE_OK  +  runs/stage0/MULTINODE_OK 哨兵

Stage 1 (RL on equation_numeric_guess, 8h × 4 GPU × N node)
├── 目标: 验证 RL 真的能涨 + anchor 不掉
├── 配置: 12 outer × 50 inner, G=8, β=0.05, T=1.0, top_p=0.95
├── 数据池: 136 题 equation_guess + 596 题 equation_deduce (~732 prompt)
├── 关键 gate: G1 (reward 单调↑, group var > 0.05)
│            G2 (anchor cats within −1.0%)
└── 输出: output/rl-equation-guess/

Stage 3 (RL on cryptarithm combined, 12h × 2-3 chain × 4 GPU × N node)
├── 目标: 攻克最硬的 task
├── 数据池: 659 deduce + 164 guess = 823 prompt
├── 关键 gate: G3 (reward > 0.05 在 iter 2 后; 否则触发 Stage 3.5)
└── 输出: output/rl-cryptarithm/

Stage 3.5 (cold start, conditional, 6h × 4 GPU 单 node)
├── 触发条件: Stage 3 G3 失败
├── 方法: 用 Huikang baseline 在 823 题各采 64 条 T=1.0, 保留 r=1 + len<4K
│       在合并集上 SFT 5 epoch (LR=5e-6)
└── 目标: pass@1 7% → 18-20%, 让 Stage 3 GRPO 有信号可用

Stage 4 (OPD merge, 8h × 4 GPU × N node)
├── 目标: 把 task adapter 蒸馏到一个统一 adapter
├── 数据: 各 task RL adapter T=0 trace + Huikang baseline anchor trace
├── 比例: §3.4 表格
├── 关键 gate: G4 (每 50 步 anchor 不退化 > 1%)
└── 输出: output/opd-final/
```

---

## 7. 数学公式速查表

| 名称 | 公式 | 含义 |
|---|---|---|
| RL 总目标 | `max_θ  E_x[ E_y~πθ[ R(y,x) ] ] - β · KL(πθ‖π_ref)` | 最大化 reward，但不远离 baseline |
| Reward | `R(y,x) = verify(extract(y), gt(x))` | 0 或 1 |
| Group mean | `μ = (1/G) · Σ_i r_i` | 这道题这 G 次的平均得分 |
| Group std | `σ = sqrt((1/G) · Σ_i (r_i - μ)²)` | 这道题这 G 次得分的离散度 |
| Advantage | `A_i = (r_i - μ) / (σ + ε)` | 第 i 条相对 group 的好坏（标准化）|
| GRPO Loss | `L = -Σ_i A_i · log πθ(y_i|x) + β · KL` | 梯度方向：advantage 高的 trajectory 概率推高 |
| KL (Schulman k3) | `KL ≈ E[ exp(Δ) - Δ - 1 ]`, `Δ = log π_ref - log πθ` | 不需要全 vocab 求和，无偏估计 |
| Token-level KL | `KL_token = Σ_t [exp(Δ_t) - Δ_t - 1]` | 每个 token 各算一次再 sum |
| OPD SFT 目标 | `min_θ -Σ_(y_i ~ mix) log πθ(y_i \| x_i)` | 普通 cross-entropy SFT, 数据混合是关键 |

---

## 8. 调试 Checklist（gates 通俗版）

| Gate | 通俗问题 | 失败时怎么办 |
|---|---|---|
| G0 | smoke 跑完 loss 是不是 NaN？disable_adapter 真有用吗？多 node NCCL 是不是走 IB？| 没救 → 找原因；disable_adapter 不行 fallback 加载 ref 模型 |
| G1 | reward 是不是在涨？每 group 是不是有正有负？KL 没炸吧？| reward 没动 → 增大 LR 或减小 β；KL 炸了 → 加大 β |
| G2 | anchor task 是不是没掉 1%？| 掉了 → β 太小 policy 飘了，加大 β 或缩短 outer loop |
| G3 | cryptarithm reward 在 iter 2 后 > 0.05 了吗？| 没 → 触发 Stage 3.5 cold start |
| G4 | OPD SFT 时 anchor 在 eval 中是不是没掉？| 掉了 → 写 sentinel early stop，调混合比例 |

---

## 9. 词汇表（最快查找）

- **Policy (πθ)**：模型本身。参数 θ。给 prompt 输出 token 分布。
- **Rollout**：模型生成（采样）trajectory 的过程
- **Trajectory**：一次完整生成的 token 序列（一条答题路径）
- **Reward**：trajectory 的得分（我们这里 0 或 1）
- **Advantage**：一条 trajectory 相对 baseline 的"好多少 / 差多少"
- **Reference policy (π_ref)**：冻结的 baseline，约束 πθ 不要漂太远
- **KL divergence**：两个分布的距离（Schulman k3 是常用的低方差估计）
- **GRPO**：用 group 内 reward 均值代替 critic 的 PPO 简化版
- **PPO**：经典 policy gradient 算法，需要 critic 网络
- **OPD**：On-Policy Distillation，用模型自己生成的 trace 蒸馏，避免 teacher 分布偏移
- **LoRA**：低秩适配器，冻结基座、只训外挂的小矩阵（r=32 在我们这里）
- **PEFT**：Parameter-Efficient Fine-Tuning，HF 的 LoRA 等技术库
- **FSDP**：PyTorch 原生的模型分片技术。FULL_SHARD = base 切到所有 GPU
- **HYBRID_SHARD**：节点内 FULL_SHARD + 节点间 DP，多 node 场景标配
- **DP**：Data Parallel，每个 worker 拿不同数据但同样的模型权重副本
- **TP**：Tensor Parallel，单个矩阵乘法跨多 GPU 切分
- **Anchor**：训练时要保护、不能退化的 task 类别
- **Verifier**：判分函数（我们的 `verify()`），精确的、零误差的 reward 信号
- **Cold start**：RL 主训练前的一段 SFT，目的是把 pass@1 抬到 RL 有信号的阈值（~20%）
- **Rejection sampling**：用当前 policy 采样大量 trajectory，只保留 r=1 的，再 SFT
- **NCCL**：NVIDIA Collective Communication Library，多 GPU/多 node 通信的底层
- **gpu_memory_utilization** (vLLM)：vLLM 进程能占的显存比例。alternation 模式必须 ≤ 0.4

---

## 10. 进一步阅读

- **DeepSeek-R1 paper** (2025): GRPO 应用到推理任务的标志性工作
- **PPO paper** (Schulman 2017): GRPO 的"前身"，理解 policy gradient 必读
- **HybridFlow paper** (verl 论文): 分布式 RL 调度的设计思路
- **PEFT/LoRA 论文** (Hu 2021): 低秩适配器
- 项目内：
  - `ROADMAP.md`：战略层（哪些 task 该 RL、不该 RL，预期回报）
  - `ITERATION_REPORT.md`：实测结果（本地 + Kaggle 远端），含 RL 阻断分析
  - `training/rl/README.md`：GRPO 代码的模块级地图 + gates
  - `eval/extract.py`：reward 函数本体
  - `training/train.py`：SFT 训练器（我们 fork 出 GRPO trainer）

---

## 11. 我们具体做了哪些事情（迄今为止）

1. **诊断阶段**（对应 v3-v6 历史）：
   - 发现 v3-v5 失败原因：LoRA target_modules 没覆盖 Mamba 的 `in_proj`/`out_proj`、推理 T=1 而非 grader 的 T=0
   - 修复后 v6 自训达到 0.76

2. **强基线复现**：
   - 写了 `training/convert_tinker_adapter.py`，把 Huikang 的 Tinker 格式 adapter SVD 转换到 vLLM 兼容格式
   - 转换后的 adapter LB 达到 0.85，全 9500 训练集自评 87.79%（与 Huikang 报告的 87.8% 一致）

3. **全集诊断分析**：
   - 写了 `eval/eval_full_train.py` + sharded sbatch，在 9500 上完成 per-category 评估
   - 输出在 `ROADMAP.md` Section 1 的成绩表（也已汇总进 `README.md`）
   - 识别三个 RL 候选类：cryptarithm_deduce/guess + equation_numeric_guess

4. **当前阶段（opd 分支正在做）**：
   - 设计 GRPO 训练 pipeline（本文档 §2-§5）
   - 多 node HYBRID_SHARD 部署方案
   - OPD 整合方案（本文档 §3）
   - 实测结论见 `ITERATION_REPORT.md`（GRPO 因策略熵塌缩未能启动）

5. **下一步将做**：
   - 实施 Stage 0 smoke test（验证 disable_adapter、NCCL、JSONL 流水）
   - Stage 1 RL on equation_numeric_guess
   - Stage 3/3.5 RL + cold start on cryptarithm
   - Stage 4 OPD merge SFT
   - Kaggle 提交

---

*文档版本：v1（与 `opd` 分支同步）。如发现描述与代码不一致，以代码为准并请修正本文档。*
