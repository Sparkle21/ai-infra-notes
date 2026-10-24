# W2-01 · torch/_inductor 地形图

> 读源码先有地图。下面每个条目：**它管什么 → 为什么值得读 → 你要找的证据**。
> 仓库：`~/src/pytorch`（以你 clone 的 commit 为准，底部记 hash）。

## 一、一次 compile 调用的旅程（先背下这张图）

```
torch.compile(fn)(x)
   │
   ├─ torch/_dynamo/convert_frame.py      捕获字节码 → FX graph（graph break 在这里决定）
   │
   ├─ torch/fx/                           graph 数据结构（node/op/args；对照 IR 树）
   │
   ├─ torch/_functorch/ + AOTAutograd     训练时拆 forward/backward；推理基本透传
   │
   ├─ torch/_inductor/compile_fx.py       ★ Inductor 入口：graph 落进来
   │     ├─ graph.py                      ★ Lower to IR：FX node →Buffer/Kernel 对象
   │     ├─ fx_passes/                    ★ pattern matching 与融合决策（本周主战场）
   │     │     ├─ post_grad.py            ★ 消费者归类（compute / reduction / extern）
   │     │     └─ pattern_matcher.py      ★ 注册表：几百条 ReplacePattern
   │     ├─ ir.py                         节点类型（ExternKernel/Pointwise/Reduction...）
   │     ├─ codegen/                      ★ IR → Triton/C++（triton.py / cpp.py）
   │     ├─ cuda_buildbot / autotuner     编译产物调度与 autotune（caching.py 也在旁）
   │     └─ cudagraph_trees.py            CUDAGraphs 录制
```

## 二、逐点侦察清单（每条都要留下证据）

| # | 位置 | 管什么 | 我找到的证据（文件:函数/行） |
|---|---|---|---|
| 1 | `compile_fx.py` | compile_fx 入口：pass 流水线的组装顺序 | |
| 2 | `fx_passes/post_grad.py` | **融合决策**：group_fusion / 按消费者分类决定谁能进一个 kernel | |
| 3 | `fx_passes/pattern_matcher.py` | pattern 注册表与匹配引擎（__init__ 里几百条 add_pattern） | |
| 4 | `ir.py` | 节点分类学：什么算 pointwise / reduction / extern（extern=不可融合，直接调 cuBLAS 等） | |
| 5 | `codegen/triton.py` | 生成 Triton kernel 的模板；`num_warps` 等参数从哪来 | |
| 6 | `config.py` | 所有开关（fallback 门槛、阈值……）——读它能找到一堆「写死的启发式」 | |

## 三、Catalyst 映射笔记（本周要写的对照）

| Catalyst（你熟） | Inductor（新） | 差异一句话 |
|---|---|---|
| Optimizer 规则链（batched，顺序敏感） | fx_passes 流水线（pre/post grad 分组） | |
| TreeNode pattern 改写 | pattern_matcher 的子图替换 | |
| 物理计划阶段（算子选择） | graph.py 落到 Buffer/Kernel IR + codegen | |
| 没有东西对应 | autotune（Catalyst 没有 per-plan autotune） | |

（TODO(you)：每行补一句「差异一句话」，写完这张表就够 W2 的映射笔记了。）

## 四、读源码的三条经验（踩过再回看）

1. 别从 `compile_fx.py` 顶层一路硬读——先跑 `02_drill` 拿到一个**具体行为**，再反向找做决策的那行代码。
2. `grep -rn "def group_fusion" torch/_inductor/` 比翻文件快十倍。
3. 看到 `torch._inductor.config.xxx` 就抄下来——config 名本身就是文档。

---
## 记录

- clone 的 pytorch commit：
- 读完 Top3 大的意外发现：
