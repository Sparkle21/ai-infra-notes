# W6 · 融合验证（11/23–11/29）

> 计划：写 benchmark 脚本与内存流量论证草稿。
> 里程碑：**fused RMSNorm + 「省 HBM 往返」论证笔记** → [notes_W6.md](notes_W6.md)
> 环境：Mac 上空跑验证逻辑（fused 路径可用容器 interpret 验）；真实数字在 S3 会话（H20）。

## 文件与分工

| 序 | 文件 | 干什么 | Mac 验收 | S3 拿什么 |
|---|---|---|---|---|
| 01 | [01_bench_core.py](01_bench_core.py) | 三口径 benchmark：eager / decomposed / fused | 空跑（eager+decomposed 两口径可跑通；fused 无 triton 时优雅降级） | 三口径 × 三形状的 ms 与 GB/s |
| 02 | [02_autotune.py](02_autotune.py) | num_warps/BLOCK 扫描（autotune 范式） | 导入级自检 | best config per shape |
| 03 | [03_hbm_traffic.py](03_hbm_traffic.py) | 流量账本：解析式算 HBM 趟数与理论时间 | 跑通并与手算对上 | 与实测时间对账 |
| — | [notes_W6.md](notes_W6.md) | 论证笔记（本周里程碑） | 骨架填好 | 证据回填 |

## 实验表（S3 前必须填「预期」，S3 后填「实测」）

| # | 实验 | 变量 | 预期 | 实测 |
|---|---|---|---|---|
| 1 | 三口径对比 @ [16, 4096] | 算子组织 | | |
| 2 | 三口径对比 @ [8192, 4096] | 行数（并行度） | | |
| 3 | 三口径对比 @ [8192, 11008] | 行宽（单行工作量） | | |
| 4 | autotune 扫 num_warps | 并行粒度 | | |
| 5 | ncu 看 fused vs eager 的 Memory Throughput % | 硬件口径 | | |

## 本周方法论要求（笔记的评分标准）

1. **每个数字都有口径**：设备、dtype、形状、迭代数、是否 synchronize——缺一项这个数字就不能进笔记。
2. **论证是账本不是形容词**：「fused 快」不算论证；「decomposed 搬 3×2×N×M×4B，fused 搬 1×2×N×M×4B，
   在 4TB/s 带宽下理论时间差 Δt，实测差 Δt'，两者比值 r」才是。
3. **解释不了的差距留在笔记里**——那是 W7/W8 的燃料（比如 L2 命中让实测差距小于账本差距）。
