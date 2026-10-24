# W7 · matmul 阶梯（11/30–12/06）

> 计划：读 triton matmul tutorial 源码结构；interpret 验 naive matmul 数值；BF16 路径（H20 支持，无历史包袱）。
> 里程碑：naive matmul 数值正确
> 环境：容器 interpret 验数值；性能数字 S4 会话拿。

## 三级阶梯（本周就干这三步）

| 级 | 文件 | kernel 形态 | 学什么 |
|---|---|---|---|
| L1 | [01_naive_matmul.py](01_naive_matmul.py) | 一个 program 算输出一个 tile，不循环（小形状专用） | tl.dot 的语义、累加器（acc）、`allow_tf32` |
| L2 | [02_tiled_matmul.py](02_tiled_matmul.py) | 经典 K 维循环分块 | tiling 的本质：**用 SRAM 当中转仓**，重用 tile |
| L3 | [03_bf16_tensor_core.py](03_bf16_tensor_core.py) | 输入转 BF16，走 Tensor Core | dtype 对性能的决定性影响；数值容差怎么定 |

| — | [notes_W7.md](notes_W7.md) | | 笔记：roofline 第一次亲手算 |

## roofline 一分钟版（H20 参数，本周笔记要亲手算一遍）

- 算力上限 ≈148 TFLOPS（BF16），带宽 ≈4TB/s → 拐点 = 148e12 / 4e12 = **37 FLOP/Byte**
- matmul 每 FLOP 搬的字节：2*M*N*K FLOP / ((M*K + K*N + M*N)*bytes) ——形状越大算术强度越高，
  大 GEMM 是 compute-bound；RMSNorm 之类 ≈1 FLOP/Byte，是 memory-bound。
- **同一张卡上两种 bound 的优化思路完全不同**——这就是为什么先写 RMSNorm 再写 matmul。

## 本周验收

- [ ] 01/02 interpret 数值 PASS（含非整除形状）
- [ ] 03 的 BF16 数值在合适容差内 PASS（BF16 有效位少，rtol 要放宽——具体多少自己实验定）
- [ ] notes_W7 的 roofline 算完（S4 前写好「预期」，会话后对账）

## 对照阅读（各 20 分钟）

- Triton 官方 matmul tutorial（`03-matrix-multiplication.ipynb`）——你的 02 与它的差异就是你的学分量
- cuBLAS 的分级策略（搜索 cublas heuristic）——「工业库怎么选 kernel」的初印象
