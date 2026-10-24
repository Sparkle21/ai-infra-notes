# notes_W1 · Inductor 融合决策评审（草稿）

> 本周里程碑产物。所有「证据」都要能指回 01–04 的观察记录或 S1 会话记录。
> 写法提醒：这是**评审**，不是教程——以「它做了什么决策、我认为该打几分、依据是什么」的结构写。

## 一、评审对象：Inductor 的融合决策在干什么

（写法提示：一段话讲清 fusion pass 的输入输出——FX graph 进、若干个 fused subgraph 出。
数据库人话：规则驱动的 pattern 改写，没有代价模型投票。W2 读源码后再补准。）

## 二、证据清单

| # | 现象 | 出处 | 我的解读 |
|---|---|---|---|
| E1 | 例：bn/relu/mul 合成一个 kernel | 01_stack_overview | |
| E2 | print 导致 break，断点两侧各自成图 | 02_graph_break_probe | |
| E3 | shape/dtype 变化触发 recompile | 03_guard_recompile | |
| E4 | S1: eager vs compiled kernel 数与提速比 | 04 + S1 会话记录 | |

## 三、我的评分（每项给 做得好/幼稚 + 一句依据）

- 融合粒度选择：
- 对 graph break 的容错策略（断点两侧继续编译小图）：
- guard 机制（shape 特化 vs dynamic）：
- 缺失的东西（对照 Catalyst 有而它没有的）：

## 四、留给 W2 的问题（幼稚清单的种子）

1. （例）哪些 pattern 明明可融但没融？去源码里找 pattern 表来对照。
2. （例）融合决策里有没有任何一处读到 tensor 大小来决定策略？
3.

## 五、一句话结论

（例：「纯规则、零代价模型的 greedy 融合，对算子链长尾有效，对形状相关的取舍视而不见——
这印证了 plan.md 第七节的行业信号，也定下了 W2 源码阅读的靶子。」）
