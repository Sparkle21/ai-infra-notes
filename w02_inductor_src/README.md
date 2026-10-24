# W2 · Inductor 源码阅读（11/02–11/08）

> 计划：读 `torch/_inductor` 的 fusion pass 与 pattern matching；对照 Catalyst 规则优化写映射笔记；把「我觉得幼稚的决策」攒成清单。
> 里程碑：**幼稚清单 v1**（后续选题库）→ [notes_W2_naive_list.md](notes_W2_naive_list.md)
> 环境：Mac + 本地 clone 的 pytorch 仓库（只读，不跑 GPU）

## 准备：clone 仓库

```bash
git clone --depth 1 https://github.com/pytorch/pytorch.git ~/src/pytorch
# 浅 clone 够读源码用；不参与编译。
```

## 文件与学习顺序

| 序 | 文件 | 学什么 | 验收 |
|---|---|---|---|
| 01 | [01_source_map.md](01_source_map.md) | `torch/_inductor` 的地形图：哪个文件管什么 | 地图上 5 个关键位置填上「我找到的证据行」 |
| 02 | [02_drill_fusion_pass.py](02_drill_fusion_pass.py) | 融合决策在哪发生、按什么顺序 | 用 2 个小实验钉死一个行为差异 |
| 03 | [03_drill_pattern_matcher.py](03_drill_pattern_matcher.py) | pattern matching 的注册表长什么样 | 摘录 3 条 pattern 并写数据库类比 |
| — | [notes_W2_naive_list.md](notes_W2_naive_list.md) | **幼稚清单 v1** | ≥10 条，每条带证据 |

## 本周读法纪律

- 只读 `torch/_inductor/`（+ 少量 `torch/fx/`），不进 `torch/csrc/` 深水区。
- 每读一个文件先看它的 `__init__` / 顶层 docstring，再找入口函数。
- 版本以你 clone 的时间点为准；笔记里记 commit 短 hash。
- **幼稚清单的门槛**：每条必须是「具体决策 + 证据（文件/行为）+ 如果是我会怎么做」，情绪化的"好蠢"不算数。
