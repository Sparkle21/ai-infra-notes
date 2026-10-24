# notes_W7 · matmul 与 roofline 笔记

## 一、roofline 亲手账（S4 前完成「预期」，S4 后完成「实测」）

设备参数（S1 抄回来的）：SM 数 = ____，BF16 峰值 = ____ TFLOPS，HBM 带宽 = ____ GB/s

| 形状 (M,N,K) | FLOP | 搬运字节（fp32，LR式简化） | 算术强度 FLOP/B | 预测 bound | 实测 TFLOPS | 实测 GB/s |
|---|---|---|---|---|---|---|
| 1024³ | | | | | | |
| 4096×4096×11008（FFN 真实形状） | | | | | | |
| [8192,11008] RMSNorm（W6 数据搬来对照） | ~0（≈1FLOP/元素） | | ≈1 | memory | — | |

> 结论句式：同一张卡上，GEMM 坐在 roofline 斜坡右上（compute-bound），
> RMSNorm 坐在左下（memory-bound）——优化语言完全不同。

## 二、tiling 的本质（我的话，≤5 句）

（提示方向：SRAM 容量小但快；tile 进 SRAM 一次、消费多次；K 循环让「读」被
「算」摊薄；BLOCK 是摊薄比例的旋钮；组内排序决定 L2 能不能帮忙。）

## 三、naive → tiled → bf16 各自的台阶（S4 数字回填）

| 级 | TFLOPS | 相对上一级的台阶来源 |
|---|---|---|
| L1 naive fp32 | | — |
| L2 tiled fp32 | | tiling：读被摊薄 |
| L3 bf16 (Tensor Core) | | 峰值×2 + TC 指令 |
| cuBLAS bf16（参照） | | 工业库的全部 trick |

## 四、看不懂的地方（转进 W8 的清单机制）

1.
2.

## 五、幼稚清单回写

- 「融合/tiling 决策与形状无关」这条被我这周的 roofline 账证实/证伪了吗：
