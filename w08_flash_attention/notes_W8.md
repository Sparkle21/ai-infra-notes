# notes_W8 · 里程碑周笔记：matmul 达标判定 + 看不懂清单

## 一、里程碑 A：matmul ≥ cuBLAS 70%（W7 的收口）

- 我的最优 BLOCK 组合（M,N,K / BLOCK_M,N,K / TFLOPS）：
- cuBLAS bf16 同形状 TFLOPS：
- 比值 = ___% （≥70% 达标；未达标的差距归因：①BLOCK 组合 ②没做 grouped ordering
  ③没做流水（num_stages）——ncu 的 stall 指标指向哪个？）
- 结论句：我这版 kernel 与 cuBLAS 的差距 = ___ %，其中可解释部分 ___ %。

## 二、里程碑 B：看不懂清单（FA 复现攒出来的）

> 格式：现象/代码位置 + 我卡在哪 + 我的猜测（哪怕是错的）。
> 这份清单是 W9 方法论笔记与 W10 分享的原料——「知道自己不知道什么」比懂了什么值钱。

| # | 位置 | 看不懂的点 | 我的猜测 |
|---|---|---|---|
| 1 | 官方 06 notebook 的 STAGE 两段循环 | 为什么分两段就能省 mask？ | |
| 2 | flash_attn csrc 的 `MainloopPipeline` | 「pipeline」在 kernel 里具体指什么 | |
| 3 | csrc 里 tensor descriptor / TMA 字样 | sm_90 专属的新搬运方式？ | |
| 4 | warp specialization（producer/consumer warp） | 谁 produce 谁 consume | |
| 5 | FA2 相对 FA1 的「减少非 matmul FLOP」 | 我实现的版本里哪些行是非 matmul | |
| 6 | | | |

## 三、FA 数值与性能小结（02 的数字）

- causal / 非 causal 数值状态：
- mine/SDPA 随 N 的变化：
- O(N²) 显存账（N=4096，B=1,H=8）：标准 ___ GB vs FA ~___ MB。

## 四、调度层与 kernel 层的「省」（跨周连线）

- W3 模拟器：调度层的省 = 不空转 slot（时间维度）
- W8 FA：kernel 层的省 = 不算被 mask 的块（空间维度）+ 不落中间矩阵（带宽维度）
- 我的类比：两者都是「__」（一句话，留给分享提纲当标题）

## 五、幼稚清单回写

- 本周证据支撑/推翻的条目：
