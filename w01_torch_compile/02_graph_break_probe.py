#!/usr/bin/env python3
"""
W1-02 · graph break 探针：什么把图打断、打断的代价是什么
=================================================================
先补一句背景（W1-01 的延续）：
    torch.compile 想加速你的代码，第一步是 Dynamo 把整段 Python「画」成一张
    连续的计算图。但有些写法它画不下去，只能中途放弃、退回逐行解释执行——
    这就叫 graph break（图被断开）。
    打个比方：编译器想把整段高速路修成一条无红绿灯快速路，结果路上有
    几个路口必须停下来看情况，快速路只好断成几截。

目标：
    1. 亲手制造 ≥2 类 graph break（常见诱因：print、data-dependent 分支、
       不支持的高层 API、对 tensor 调 .item()/.nonzero() 等）；
    2. 用 TORCH_LOGS=graph_breaks 看断在哪一行、为什么；
    3. 量化「掉回 eager」的代价：断点两边的代码各自失去编译收益。

运行（Mac，CPU 即可）：
    python 02_graph_break_probe.py
    TORCH_LOGS=graph_breaks python 02_graph_break_probe.py

名词白话：
    - eager：逐行解释执行的老模式（W1-01 讲过）；
    - .item()：把 GPU/张量里的一个数取出来变成 Python 普通数字——
      这一步会强制 CPU 等 GPU 算完（同步），代价很大；
    - data-dependent：分支走哪条路取决于数据的值——这个值要运行时才知道。

数据库类比（可选，不懂可跳过）：
    graph break ≈ 整段 stage codegen 中间被迫吐回解释执行的一块——
    你在 Spark 里见过「UDF 边界打断 whole-stage codegen」，就是同一个东西。

验收：
    脚本输出三段 break 的位置与原因，并跑通「修复版」函数（无 break）。
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

# ── 三个「事故现场」，从轻到重 ──────────────────────────────────────────
# 现场一：print（轻度）——很多人以为 print 无害，其实每处都是一次 break。
# 为什么？print 要「立刻拿到结果去显示」，而编译后的代码是一条连续快路，
# 中间插一个 print 就得把张量从快路里拉出来交给 Python，路就断了。
def bad_print(x: torch.Tensor) -> torch.Tensor:
    h = x * 2
    print("debug:", h.shape)      # 破坏点：print 挂在 tensor 表达式中间
    return h + 1


# 现场二：data-dependent 分支（重度）——分支条件依赖 tensor 的值，
# 编译期不可知，dynamo 只能把整个分支结构吐回 eager。
# 白话：编译器面对 if 需要先把两条路都定下来；条件是运行期的数据时
# 它没法预知走哪条 → 干脆两边都不编译，整个 if 吐回老模式。
def bad_data_dependent(x: torch.Tensor) -> torch.Tensor:
    if x.sum() > 0:               # 破坏点：条件是运行时的值（x 里的数加起来到底正不正？）
        return x * 2
    return x * 3


# 现场三：.item()（经典）——把 tensor 拉回 Python 标量，图被拦腰截断。
# 白话：从张量里「取出一个数给 Python 用」，这个数变成运行期的普通变量，
# 编译器无法把它编进图里 → 图在这一行被斩成两段。GPU 上还附带一次
# CPU/GPU 同步等待（CPU 必须停下来等 GPU 算完才能拿到值），双重伤害。
def bad_item(x: torch.Tensor) -> torch.Tensor:
    scale = (x.abs().max() + 1).item()   # 破坏点：GPU→CPU 同步点 + break
    return x * scale


# ── 修复示范：现场一的「正确 debug 姿势」 ───────────────────────────────
def fixed_print(x: torch.Tensor) -> torch.Tensor:
    h = x * 2
    # 办法：要么删掉，要么用 torch._dynamo.disable 装饰的辅助函数隔离。
    # 这里选择直接删——生产代码里 print 本来就该用 logging 且不夹在算子链中间。
    return h + 1


def measure(fn, x: torch.Tensor, iters: int = 50) -> float:
    """粗计时（CPU 上数字只看相对量级，GPU 数字在 S1 拿）。

    返回值单位是微秒/次调用。为什么要先跑 5 次「空转」？——
    第一次调用要触发编译（很慢），先把它消化掉，计时才准。
    """
    with torch.no_grad():
        for _ in range(5):          # 预热：把首次编译、缓存查找等一次性开销消化掉
            fn(x)
        t0 = time.perf_counter()    # perf_counter：高精度计时器，比 time.time() 更适合测短耗时
        for _ in range(iters):
            fn(x)
        # （总耗时 / 次数）× 1e6 = 平均每次多少微秒（µs）
        return (time.perf_counter() - t0) / iters * 1e6  # µs/call


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(256, 256)

    # ══ 逐个「案发」：explain 只报告，不修复 ═══════════════════════════
    # 流程白话：torch.compile(fn) 包一层壳 → 调用一次让它真正编译 →
    # explain() 再单独诊断一次，报告图断成几截、各断在哪。
    for name, fn in [("bad_print", bad_print), ("bad_data_dependent", bad_data_dependent), ("bad_item", bad_item)]:
        cfn = torch.compile(fn)
        with torch.no_grad():
            _ = cfn(x)                     # 第一次调用触发编译（break 也会在这里显形）
        ex = torch._dynamo.explain(fn)(x)  # 版本敏感：确认你的 torch 上该 API 存在
        print(f"[{name}] graph_count={ex.graph_count}  breaks={ex.graph_break_count}")
        # 白话：breaks > 0 = 这个函数的图被断开了（理想是 0）；graph_count = 断成的段数。
        # TODO(you)：配 TORCH_LOGS=graph_breaks 再跑一遍，把三个函数各自的
        # break 行号和 reason 抄到本文件底部观察记录里。
        # 操作：终端里运行  TORCH_LOGS=graph_breaks python 02_graph_break_probe.py
        # 输出会多一大段日志，搜「break」关键字即可定位。

    # ══ 代价量化：break 版 vs 修复版 vs 纯 eager ═══════════════════════
    # 三档对比的读法：
    #   t_eager = 不编译的基础线；t_break = 编了但中间断一次；t_fixed = 编了且无断点。
    #   预期：t_fixed < t_break（修复值得），t_break 未必比 t_eager 慢（见下方思考题 1）。
    t_eager = measure(bad_print, x)                     # 不编译，直接跑（含 print）
    t_break = measure(torch.compile(bad_print), x)      # 编译，但中间有 print 断点
    t_fixed = measure(torch.compile(fixed_print), x)    # 编译，无断点（正确姿势）
    print(f"\n现场一（print）：eager={t_eager:8.1f}µs  compiled(有break)={t_break:8.1f}µs  compiled(修复)={t_fixed:8.1f}µs")
    # TODO(you)：回答——
    #   1) 有 break 的 compiled 相比 eager 还剩多少优势？（x*2 和 x+1 分居断点两侧时，
    #      两段各自还是会被编译成小图——所以优势没有归零，想想为什么。）
    #   2) bad_item 的 break 为什么比 bad_print 更伤？（提示：同步点 + 标量回传后
    #      后续所有运算的 shape/scale 都成了运行期未知。）

    # ══ vLLM 视角（为什么要较真这个）═══════════════════════════════════
    # vLLM 的 worker 循环里如果混进一处 .item()/print，整个 step 的 capture 就废了。
    # 这就是它代码里大量出现 "不要在热路径放 python 分支" 注释的原因。
    print("\n对照任务：在 vllm 仓库 grep 一下 graph_break 相关注释或 "
          "torch._dynamo.disable 的使用位置，记 2 处到 notes_W1.md。")


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录 ═══════════════════════════════
# - bad_print     break 行号/原因：
# - bad_data_dep  break 行号/原因：
# - bad_item      break 行号/原因：
# - 三档计时（µs）：eager / compiled(break) / compiled(fixed)：
# - vLLM 里找到的 dynamo 相关位置：
