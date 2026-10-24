#!/usr/bin/env python3
"""
W2-02 · 融合决策钻探：行为实验 → 反向定位源码
=================================================================
先补一句背景（W1 的延续）：
    W1 看到了「编译后变快」，W2 要回答「Inductor 是怎么决定哪些算子
    合并成一个 kernel 的」——这个决策叫融合决策。
    研究思路是科学方法：先设计实验看行为（黑盒），再拿着现象去
    源码里找对应的那几行（白盒）。

目标：
    融合发生在哪？决策长什么样？本文件做两组「行为对照实验」，
    然后带着现象去源码里找那行代码（对照 01_source_map 的第 2 条）。

运行（Mac，CPU 即可）：
    python 02_drill_fusion_pass.py
    TORCH_LOGS=output_code python 02_drill_fusion_pass.py  # 看 kernel 划分

名词白话：
    - pointwise（逐元素）：每个位置的数只和自己算，如「全部×2」「负数归零」。
      不需要邻居的信息 → 最适合融合；
    - reduction（归约）：把一串数压成少数几个数，如 sum、max。
      需要跨位置汇总 → 融合方式与 pointwise 不同；
    - extern（外部库调用）：Inductor 自己不写 matmul，直接调用 NVIDIA
      现成的高性能库 cuBLAS（专做矩阵乘法的官方库）。它是天然的融合边界。

实验设计（数据库人话，可选）：
    实验 A 测「消费者类型决定分组」——同一个中间结果，
    被 pointwise 消费 vs 被 reduction 消费，融合结果不同；
    实验 B 测「extern kernel 是融合边界」——一旦进了 cuBLAS 调用，
    它前后的元素运算各自分组。
"""
from __future__ import annotations

import torch


# ══ 实验 A：消费者类型如何决定融合分组 ══════════════════════════════════
def case_pointwise(x: torch.Tensor) -> torch.Tensor:
    h = x * 2.0          # pointwise
    h = torch.relu(h)    # pointwise（同类消费者 → 预期融成一个 kernel）
    return h + 1.0


def case_reduction(x: torch.Tensor) -> torch.Tensor:
    h = x * 2.0          # pointwise
    h = h.sum(dim=-1)    # reduction（预期：sum 单独成 kernel 或与 producer 融合，
    return h + 1.0       #  但 add 不一定能跟 sum 同组——去 output_code 里验证）
    # 白话预期：乘法是「逐位置」，sum 是「跨位置压缩」——两类活节奏不同，
    # Inductor 对它们的分组规则也不同。到底怎么分组，去生成代码里数函数个数。


# ══ 实验 B：extern（cuBLAS matmul）是融合边界吗 ═════════════════════════
def case_with_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    h = torch.relu(x)    # pointwise
    h = h @ w            # extern：预期 Inductor 不自己写 matmul，直接调 cuBLAS
    return h * 2.0       # pointwise（预期与谁融合？）


def show(name: str, fn) -> None:
    """把一个函数编译并跑通，只打印「跑通了 + 输出摘要」。
    真正要观察的东西在 stderr 的生成代码里（用 TORCH_LOGS=output_code 开启）。"""
    cfn = torch.compile(fn)
    x = torch.randn(256, 256)
    w = torch.randn(256, 256)
    out = cfn(x, w) if fn is case_with_matmul else cfn(x)   # matmul 案例需要两个输入
    print(f"[{name}] 跑通，输出 sum={out.sum().item():.3f}")   # .item() 这里可用：不在被编译的热路径里
    # TODO(you)：用 TORCH_LOGS=output_code 逐个跑，把每个 case 生成的
    # kernel 数量与「哪些算子进了同一个函数」抄到观察记录。
    # 看法：生成代码里每个 def 的函数体对应一个 kernel；
    # 数函数个数 + 看函数体里做了几件事，就是融合结果。
    # 然后回到源码：post_grad.py 里给消费者分类的函数，找到对应分支，
    # 在 01_source_map.md 表格第 2 行填上证据。


def main() -> None:
    torch.manual_seed(0)
    show("A-pointwise链", case_pointwise)
    show("A-reduction混链", case_reduction)
    show("B-matmul边界", case_with_matmul)

    # ══ 进阶钻头（选做）：修改 config 观察决策变化 ══════════════════════
    # torch._inductor.config 里找与 fusion 相关的开关（如 max_autotune、
    # post_grad_... 系列；名字随版本变），改一个开关重跑 case_with_matmul，
    # 观察生成的 kernel 划分是否变化。找到一条「写死的启发式」就记入幼稚清单。
    # 白话：这一步在验证 W2 的核心判断——融合决策是「写死的规则」而不是
    # 「按代价算的」；每找到一条写死规则，都是一条幼稚清单素材。
    print("\n下一步：源码定位。建议 grep：")
    # grep = 在文件里搜文本的命令；-rn = 递归目录 + 显示行号。
    # torch 的源码就在你 pip 安装的目录里，可以直接读。
    print("  grep -rn 'class.*Fusion' torch/_inductor/fx_passes/post_grad.py")
    print("  grep -rn 'is_node_supported' torch/_inductor/fx_passes/")


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录 ═══════════════════════════════
# - A-pointwise 链的 kernel 数与组成：
# - A-reduction 混链的 kernel 数与组成（sum 和 add 各在哪）：
# - B-matmul 边界：relu / matmul / mul 的分组结果：
# - 我在 post_grad.py 定位到的决策代码（文件:函数）：
# - 幼稚清单新增条目（转记到 notes_W2_naive_list.md）：
