# W9-01 · 量化概念补课（30 分钟）

> 目标：建立一张能装下所有量化名词的表格。读不懂的 kernel 代码 90% 是名词问题。

## 一、一句话总纲

量化的本质：**用更少的位存/算数字，用 scale 把信息搬回来**。
所有方案都是这个句式的变体：`y = dequant(quant(x))`，差别只在「几位、scale 算多准、在哪一步还原」。

## 二、名词归类表

| 术语 | 含义 | 一句话点评 |
|---|---|---|
| W8A8（INT8） | 权重和激活都是 8bit 整数 | scale 质量决定精度；kernel 是 int8 dot + fp32 累加 |
| weight-only（W8A16） | 只把权重量化成 8bit，激活保持 16bit | 显存省一半、带宽省一半；算力不变（dequant 后按 bf16 算） |
| FP8（E4M3/E5M2） | 8bit 浮点（指数位不同） | H20/Ada 之后硬件原生支持；比 INT8 少一次 scale 校准纠结 |
| per-tensor scale | 整个张量一个 scale | 最省事、精度最差 |
| per-channel / per-group | 每个输出通道（或每组）一个 scale | 精度好；kernel 要按组广播 scale（你 W8 的 acc 广播经验直接复用） |
| static / dynamic 校准 | scale 离线定 / 运行时按 batch 定 | dynamic 多一次激活统计 pass |
| GPTQ / AWQ | 权重量化的「怎么算 scale」算法 | 对 kernel 透明——kernel 只关心 scale 长什么样 |
| smoothquant | 把激活的离群值「搬」给权重 | 一种让 W8A8 可行的预处理 |

## 三、kernel 视角：量化 kernel 到底在算什么

W8A8 linear 的 forward（就是你要在 W10 手写的那个）：

```
y = x @ W^T + b            # 原始世界
↓
y = (x_int8 * s_x) @ (W_int8 * s_w)^T + b
 = s_x * s_w * (x_int8 @ W_int8^T) + b    # scale 都是标量（per-tensor 时）→ 提出来！
```

- per-tensor：scale 是标量，可以**提出来在 epilogue 乘**——kernel 主体就是纯 INT8 GEMM
  （和 W7 的 matmul 只差 dtype！这就是为什么 W10 的 kernel 你已经会写了）。
- per-channel/per-group：scale 是向量，必须在 dot 之后按列（或按组）乘——广播的位置就是 kernel 设计的位置。
- 「量化收益从哪来」的三个账本（S5 验证对象）：
  1. 显存：权重体积 ÷2（bf16→int8）
  2. 带宽：weight-only 的 decode 阶段收益最大（每 token 都要搬全部权重）
  3. 算力：W8A8 的 INT8 峰值是 BF16 的 2 倍（理论；实际吃到几成看 kernel）

## 四、与 W7/W8 的知识对接（你已经会的部分）

| 已有知识 | 在量化里的对应 |
|---|---|
| tl.dot 的 dtype 规矩（W7-01） | int8 输入走 INT8 Tensor Core——同一个规矩 |
| acc 用 fp32 累加（W7） | INT8 dot 同样 fp32 累加（防溢出 + 精度） |
| epilogue 的标量/向量广播（W5/W8） | scale 乘法就是 epilogue——广播位置=粒度 |
| roofline 账本（W6/W7） | 量化三收益的账本——S5 的 A/B 就是在验这三个账 |

## 五、自测三题（答不上来就回去翻）

1. weight-only 为什么省带宽不省算力？
2. per-channel scale 下，为什么 scale 不能在 dot 前提出来？
3. dynamic 校准相比 static 多花了什么？（一次对激活的统计 pass——它是不是也可以「online」掉？
   这个问题连到方法论笔记：**量化校准和 AQE 的 runtime stats 是同构的**。）
