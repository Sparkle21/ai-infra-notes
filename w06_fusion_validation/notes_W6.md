# notes_W6 · 「融合省的是 HBM 往返」论证笔记

> 本周里程碑。结构：命题 → 账本 → 实测 → 对账 → 修正结论。
> 评分标准见本目录 README（口径齐全、账本不是形容词、解释不了的差距保留）。

## 一、命题

fused RMSNorm 相对逐算子（decomposed）的加速，主要来自**减少 HBM 往返**，
而非减少计算量（两者的 FLOP 几乎相同——每元素都是一次平方、一次乘、一次加级别的活）。

## 二、账本（03_hbm_traffic.py 的口径）

| 版本 | HBM 趟数 | 说明 |
|---|---|---|
| decomposed | 7（读4写3） | pow→mean→rsqrt·mul→mul·w，中间结果都要落 HBM |
| fused | 2（读1写1） | 数据只在 kernel 入口/出口过 HBM |
| eager（黑盒） | ?（待 S3 反推） | 框架实现可能内部融合，账本未知——**这正是有趣的部分** |

理论加速比上限 = 7/2 = 3.5x（纯带宽视角，忽略 launch 与并行度差异）。

## 三、实测（S3 回填；口径：H20 sm_90 / FP32 / iters=200 / synchronize 划界）

| shape | eager ms | decomp ms | fused ms | fused GB/s | 实测加速比 |
|---|---|---|---|---|---|
| [16, 4096] | | | | | |
| [8192, 4096] | | | | | |
| [8192, 11008] | | | | | |

## 四、对账（03 的 --measured 输出贴这里）

- decomposed 有效带宽 / 理论占比：
- fused 有效带宽 / 理论占比：
- 实测加速比 vs 理论 3.5x 的偏差：

## 五、修正后的结论

（写法：先说账本哪几行假设站住了、哪几行被实测打脸、为什么；
再给出修正后的账本；最后一句人话总结——给不懂 GPU 的同事也能听懂的版本。）

## 六、ncu 证据（S3 的 `--set full` 报告）

- fused kernel 的 Memory Throughput %：
- eager kernel 的 Memory Throughput %：
- warp stall 主因：

## 七、遗留问题（W7/W8 的燃料）

1.
2.

## 八、幼稚清单回写

- 哪条被本周证据验证/推翻了（去 w02/notes_W2_naive_list.md 标记）：
