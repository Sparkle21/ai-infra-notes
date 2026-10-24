# W8 · FlashAttention forward 复现（12/07–12/13）

> 计划：读 FA2 forward 参考实现；准备复现代码。
> 里程碑：**matmul ≥ cuBLAS 70% + 「看不懂的地方」清单** → [notes_W8.md](notes_W8.md)
> 环境：容器 interpret 验数值（小形状）；性能与 ncu 在 S4 会话。

## FA2 到底在干什么（30 秒版）

标准注意力的瓶颈不是计算，是那个 **N×N 的中间矩阵**：softmax(QK^T) 必须整体落地显存
（O(N²) 显存 + 3 次读写）。
FA 的思路：**softmax 可以流式算**（running max + running sum，见 01 的推导），
于是 K/V 按块流过 SRAM，中间矩阵永远不落显存——O(N) 显存，读写也大减。
代价：外积的关联重排（rescale），这就是 kernel 里那几行「魔法」。

## 文件与学习顺序

| 序 | 文件 | 学什么 | 验收 |
|---|---|---|---|
| 01 | [01_fa2_forward.py](01_fa2_forward.py) | online softmax 推导 + kernel 复现 | 非 causal 与 causal 数值都 PASS |
| 02 | [02_fa_benchmark.py](02_fa_benchmark.py) | mine vs SDPA，序列长度扫描 | 容器空跑；S4 拿真数字 |
| — | [notes_W8.md](notes_W8.md) | 里程碑笔记 + 看不懂清单 | 里程碑两件套齐 |

## 阅读路线（先读再写）

1. Triton 官方 `06-fused-attention.ipynb` 的前 1/3（跳过 backward）
2. FA2 论文只读 Sec.3 的 Algorithm 1（一页伪代码，对上 01 的 kernel 逐行）
3. flash-attention 仓库 `csrc/flash_attn/src/flash_fwd_kernel.h`——**不求读懂**，
   目标是攒「看不懂清单」：CUDA 版多出来的东西（tensor descriptor、pipeline、
   warp specialization）每一项都是一张卡片。

## 本周验收

- [ ] 01 两种模式数值 PASS（含与 SDPA 对照）
- [ ] 能不看资料在白板上写 online softmax 的两条递推
- [ ] notes_W8 的看不懂清单 ≥5 条（这是下周方法论笔记的原料）
