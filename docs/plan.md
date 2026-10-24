# AI Infra 学习路线（推理优化主线 · 10 周）

> 起点：数据库内核（执行器 + 优化器 / Spark SQL）→ 现做模型推理调优（引擎层、并行策略层，nsys/profiler 视角）
> 目标：成为"能把优化器方法论带进推理编译/调度层"的人，而非手写 kernel 的人
> 起始日：2026-10-19（周一）　结束日：2026-12-27

---

## 一、边界：做什么、不做什么

**主线（投入 80%）**

- 推理优化的编译层与调度层：`torch.compile` / Inductor 融合决策、vLLM / SGLang 调度器
- 代价模型视角的评审：融合策略、并行计划枚举、运行时自适应
- 训练与 agent 保持"够用"深度，不投护城河

**薄壳（投入 20%，只求读得懂）**

- Triton 能写 fused kernel、能自己 benchmark 验证
- 读得懂别人的 kernel（含 `torch.compile` 吐出的 Triton、flash-attention 的 `csrc/`）
- 对每个算子有成本直觉：融合省的是 HBM 往返、非连续访问慢在哪

**明确不做（留给专职底层）**

- 裸 CUDA 手写按教程刷题 —— 不进排期，撞到具体墙时查那一小块
- CUTLASS / CuTe 深水区 —— 彻底删除
- 手写 NCCL 通信 kernel —— 只读实现，不自己写
- PyTorch 建模/训练框架广度（`nn.Module` 搭建、DataLoader、分布式训练用户侧）—— 需要时一个周末补，不排进主线

---

## 二、环境与硬件

| 位置 | 角色 | 装什么 | 用途 |
|---|---|---|---|
| **Mac** | 日常工位（周一到周五） | miniconda + CPU 版 torch；OrbStack 起 Ubuntu 容器装 `triton`；Nsight Systems GUI（Mac 原生）；VS Code + Remote-SSH；克隆 pytorch / vllm / sglang 仓库 | 读源码、写 kernel、`TRITON_INTERPRET=1` 验正确性、打开租卡机采的 `.nsys-rep` |
| **租卡云实例** | 集中测试场（每 1.5–2 周一次） | conda + torch(cu12x) + triton + vllm/sglang；cuda-toolkit + ncu / nsys CLI | 一切真实执行、benchmark、profile |

**选型结论**

- ✅ 选 `ecs.gn8v.*`（GPU 计算型，直通整卡，1×96GB = H20，sm_90）
- ❌ 不选 `sgn8ia`（vGPU 切片）：ncu 硬件 counter 被封、时分共享导致计时不可信、2–16GB 塞不下 7B
- ⚠️ H20 算力被砍（BF16 ≈148 TFLOPS）、带宽满血（≈4TB/s HBM3），roofline 拐点 ≈37 FLOP/Byte（A100 约 162）→ 很多 GEMM 变 memory-bound。**tiling 参数与"提速 N%"的结论不外推到其他卡**，笔记标注 `sm_90 + 低算力带宽比`
- 💡 若页面其他 GPU 分类里能看到 A100（gn7e），W1–W8 的会话用 A100 更划算（结论可外推），只在碰 FP8 时用 H20

**ncu 没有 Mac GUI**，在租卡机上用 CLI 出报告；nsys 有 Mac GUI，采完拉回本地看。

---

## 三、租卡会话排期（共 6 场，约 19 GPU 小时）

| 会话 | 时间 | 规格 | 时长 | 粗估 | 只做这些事 |
|---|---|---|---|---|---|
| **S1** | W1 末（10/25 前后） | `gn8v.4xlarge` | 3h | ¥109 | torch.compile 前后对比 + nsys 数 kernel launch 变化；**环境装好后存自定义镜像** |
| **S2** | W4 末（11/15 前后） | `gn8v.4xlarge` | 4h | ¥146 | serve 7B 模型，打 log 观察 SGLang 调度行为；nsys 抓 decode 阶段 batch 动态；权重下到数据盘 |
| **S2.5**（可选） | W4 末 | `gn8v-2x.8xlarge` 2×96G | 2h | ¥143 | 多卡 NCCL all-reduce 观察 —— 把"TP 的 all-reduce 在 kernel 层什么形状"看懂 |
| **S3** | W6 末（11/29 前后） | `gn8v.4xlarge` | 3h | ¥109 | fused RMSNorm benchmark + autotune（`num_warps`/block）+ ncu 对比融合前后 memory throughput |
| **S4** | W8 末（12/13 前后） | `gn8v.4xlarge` | 5h | ¥182 | matmul tutorial 推到 Tensor Core 版（BF16）+ FlashAttention forward 复现 + ncu 读 stall |
| **S5** | W10 中（12/20 前后） | `gn8v.4xlarge` | 4h | ¥146 | INT8 W8A8 Triton kernel 实跑 + SGLang 量化开关端到端 A/B + 一个量化 kernel 的 SOL |

单卡合计 ≈ **¥690**（含 S2.5 则 ≈¥833）。

**省钱两条**：优先查**抢占式/spot**（GPU spot 常 1–2 折，会话都是"跑几小时拿数据就走"，被回收不心疼）；用完**立刻释放实例**，不要只停止。

---

## 四、10 周计划表

| 周次 | 日期 | 主题 | Mac 侧（读 + 写 + interpret） | 租卡侧（对应会话） | 里程碑产出 |
|---|---|---|---|---|---|
| **W1** | 10/19–10/25 | torch.compile 全景 | 跑通最小例子；`TORCH_LOGS=output_code` dump 编译产物；理解 Dynamo→AOTAutograd→Inductor→CUDAGraphs 四层；记 graph break / guard recompile 两个坑 | S1 | 一页"Inductor 融合决策评审"草稿 |
| **W2** | 10/26–11/01 | Inductor 源码 | 读 `torch/_inductor` 的 fusion pass 与 pattern matching；对照 Catalyst 规则优化写映射笔记；把"我觉得幼稚的决策"攒成清单 | — | **幼稚清单 v1**（后续选题库） |
| **W3** | 11/02–11/08 | vLLM 调度器 | 读 scheduler、continuous batching、preemption（recompute vs swap）、chunked prefill；对照 AQE 运行时自适应记映射 | — | 调度数据流标注图（草稿） |
| **W4** | 11/09–11/15 | 调度实测 | 准备 serve 脚本与观察点；顺带扫 block manager / radix cache（非主场，认识即可） | S2（+S2.5） | **一张 SGLang 调度数据流标注图** |
| **W5** | 11/16–11/22 | Triton 入门 | 学 API 与 block 编程模型；写 fused RMSNorm；Mac 容器里 `TRITON_INTERPRET=1` 对数值 | — | 数值正确的 fused RMSNorm |
| **W6** | 11/23–11/29 | 融合验证 | 写 benchmark 脚本与内存流量论证草稿 | S3 | **fused RMSNorm + "省 HBM 往返"论证笔记** |
| **W7** | 11/30–12/06 | matmul | 读 triton matmul tutorial 源码结构；interpret 验 naive matmul 数值；BF16 路径（H20 支持，无历史包袱） | — | naive matmul 数值正确 |
| **W8** | 12/07–12/13 | FlashAttention | 读 FA2 forward 参考实现；准备复现代码 | S4 | **matmul ≥ cuBLAS 70% + "看不懂的地方"清单** |
| **W9** | 12/14–12/20 | 量化薄壳 | 读 vLLM/SGLang 的 INT8/FP8 kernel 代码（只读不跑）；写笔记《查询优化器方法论能给推理调度补什么》 | — | 方法论笔记初稿 |
| **W10** | 12/21–12/27 | 量化实测 + 收口 | 整理全部清单与笔记 | S5 | **技术分享提纲**（选题 = 幼稚清单升级版） |

---

## 五、每周节奏

- **周一到周五（Mac）**：读源码 + 写代码 + interpret 验数值。产出是"代码和笔记"。
- **里程碑末（租卡）**：把攒够的一批"待测问题"集中测完。产出是"数字"。
- **每次会话后（Mac）**：分析 nsys/ncu 报告，把结论写回笔记。

铁律：**interpret 保证上卡时数值已经是对的，租来的小时只用来测性能。** 绝不在计时里 debug Python 报错。

---

## 六、租卡工程纪律

1. **镜像化**：S1 装好的 conda/依赖存进平台自定义镜像，之后开机即就绪，不在计时里 `pip install`。
2. **数据盘存模型**：HF 权重与数据集放持久数据盘（系统盘随实例销毁）。阿里云内网到 ModelScope 快，7B 十几分钟，重下也可接受。
3. **代码永远在 Mac 改**：git 推私有仓库，会话里 `pull`；产物（profile、报告）当次会话 `scp` 回 Mac。
4. **别信"节省模式停机"**：它虽不收 GPU 费用，但释放资源配额，下次可能因库存不足起不来。正确姿势 = 镜像 + 数据盘保留，实例用完即释放。
5. **S1 第一件事是验证 ncu 可用**：`ncu --version` + 一个最小 kernel 跑 `ncu --set full`，确认指标不是 `<not supported>`。万一不行要在 W1 就知道。

---

## 七、背景映射表（复习用，不是学习清单）

| 数据库内核里的你 | 推理栈对应物 | 陌生度 |
|---|---|---|
| Catalyst 规则优化 pass（IR 树 pattern 改写） | Inductor fusion pass 的 pattern matching | 几乎零 |
| Tungsten whole-stage codegen | Triton kernel 生成本身 | 低，差目标 ISA |
| RDD 惰性求值 / DAG 构建切分 | torch compute graph → compile → 执行 | 几乎零 |
| Shuffle / exchange / 分区倾斜 | 集合通信、TP 度选择、MoE expert load balancing | 概念同源 |
| AQE 运行时自适应重优化 | continuous batching 调度、chunked prefill、preempt/recompute | 概念同源 |
| Buffer pool / 分页管理 | PagedAttention 的 KV cache block manager | 主场外，认识即可 |
| 代价模型 / join order 枚举 | 算子成本模型、并行策略搜索 | ⭐ 空白区 = 你的机会 |

**行业信号**：LLM 推理系统的"优化器侧"落后查询优化十年以上 —— Inductor 融合是纯 greedy、无代价模型参与；vLLM/SGLang 调度是写死的启发式，没有 plan enumeration 和 cost-driven 的 TP/PP/DP 自动选型。这是你的旧方法论在新赛道的位置。

---

## 八、资源清单

**读**

- 《Programming Massively Parallel Processors》前几章（按需查，不通读）
- Triton 官方 tutorial（matmul / fused softmax / FlashAttention）
- PyTorch 源码 `torch/_inductor/`
- vLLM / SGLang 源码：`scheduler`、`attention/backends`、`csrc/quantization`
- NVIDIA 技术博客：roofline、memory coalescing、Tensor Core

**工具**

- `TORCH_LOGS=output_code` —— dump 编译产物
- `TRITON_INTERPRET=1` —— Mac 上纯 Python 逐元素跑 kernel，验正确性
- `nsys profile` —— 时间线、kernel launch 次数、通信 overlap
- `ncu --set full` / `--set roofline` —— 单 kernel：SOL、算术强度、warp stall 原因

**在线兜底**

- LeetGPU 这类浏览器平台：不想开卡时练 kernel 手感，零 setup 成本
