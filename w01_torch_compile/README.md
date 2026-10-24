# W1 · torch.compile 全景（10/19–10/25）

> 计划：跑通最小例子；`TORCH_LOGS=output_code` dump 编译产物；理解 **Dynamo→AOTAutograd→Inductor→CUDAGraphs** 四层；记 graph break / guard recompile 两个坑。
> 里程碑：一页「Inductor 融合决策评审」草稿 → [notes_W1.md](notes_W1.md)
> 环境：Mac（CPU torch 即可，本周末上 S1 会话拿 GPU 数字）

## 文件与学习顺序

| 序 | 文件 | 学什么 | 验收 |
|---|---|---|---|
| 01 | [01_stack_overview.py](01_stack_overview.py) | 四层栈各自做什么、产物长什么样 | 打印出 FX graph / Inductor 代码各一份 |
| 02 | [02_graph_break_probe.py](02_graph_break_probe.py) | graph break：什么触发、掉回 eager 的代价 | 制造并捕获 ≥2 处 break，记录恢复行为 |
| 03 | [03_guard_recompile.py](03_guard_recompile.py) | guard：什么条件变化触发 recompile | 观察编译缓存命中/重编译计数变化 |
| 04 | [04_compile_vs_eager.py](04_compile_vs_eager.py) | eager vs compiled 计时 harness（S1 上 GPU） | CPU 空跑逻辑通过；GPU 数字 S1 回填 |
| — | [notes_W1.md](notes_W1.md) | 里程碑底稿：融合决策评审 | 周末写完草稿 |

## 本周背景映射（来自 plan.md 第七节）

| 数据库里的你 | 本周对应物 |
|---|---|
| Catalyst 规则优化 pass | Inductor fusion pass（W2 深入） |
| RDD 惰性求值 / DAG 构建切分 | dynamo 捕获的 FX graph → compile → 执行 |
| Tungsten whole-stage codegen | Inductor 生成的 Triton/C++ 代码 |

## 提醒

- `torch._dynamo` 的 API 随版本变动，凡注释标「版本敏感」处，先小跑验证再依赖。
- Mac 上 `torch.compile` 默认走 CPU codegen（C++/OpenMP），没有 Triton——**这是预期**，GPU 版本在 S1 看。
