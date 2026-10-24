#!/usr/bin/env python3
"""
W5-01 · vector_add：你的第一个 Triton kernel
=============================================================
先补几句背景（Triton 是什么、为什么学它）：
    GPU 上有几万个计算小单元，想快就得「把活切给它们分头干」。
    Triton 让你用 Python 语法写这种「分头干」的代码（不用学 C++/CUDA），
    是 W5–W8 全部手写 kernel 的工具。

    核心思想一句话：你只写「一个工人怎么干自己那一份」，
    Triton 负责把成千上万个工人同时拉起来——每个工人叫一个 program。

    今天用「两个数组相加」当 Hello World：z[i] = x[i] + y[i]。

名词白话：
    - program：一个「工人」（GPU 上的执行单元）。同一段 kernel 代码，
      每个工人拿到自己的编号，处理自己那一段数据；
    - grid：工人总数（一个元组，如 (32,) 表示拉 32 个工人）；
    - BLOCK_SIZE：每个工人负责的元素个数（切多大的活给一个人）；
    - mask（掩码）：越界保险——数据总数除不尽时，最后一个工人会多拿到
      一些「假下标」，mask 把假下标标出来，防止读写越界；
    - tl.load / tl.store：整块读/写显存（一次搬一整块，不是逐个元素）。

目标：
    亲手摸到三件东西：program_id（我是哪个 program）、
    tl.load/tl.store（整块读写）、mask（边界保护）。
    学完后你应该能回答：「一个 program 处理多少元素？总共起了几个 program？」

运行（OrbStack 容器内）：
    TRITON_INTERPRET=1 python 01_vector_add.py
    （TRITON_INTERPRET=1 = 用纯 Python 模拟 GPU 执行——Mac 上没 GPU 也能验证算得对不对）

对照（数据库类比，可选，最后一周用得上）：
    grid 划分 ≈ 数据分区的切法；BLOCK_SIZE ≈ 每个分区的页大小；
    mask ≈ 分区边界补齐——切分不整除时保证正确性的那层胶。
"""
from __future__ import annotations

import os

# 铁律：Mac/容器上一律 interpret（先保证数值对，性能留到租卡时测）。
# setdefault：如果你已在环境里设过这个变量，就不覆盖你的选择。
os.environ.setdefault("TRITON_INTERPRET", "1")

# noqa: E402 = 「我知道 import 不在文件顶，别报警」——因为必须先设环境变量再 import
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402   # tl = Triton 语言库：program_id/load/store 都在这


@triton.jit   # 这个装饰器把普通 Python 函数标记为「Triton kernel」——里面的代码会被翻译成 GPU 代码
def vector_add_kernel(
    x_ptr,          # 输入 a 的首地址（指针：指向显存里数组开始的位置）
    y_ptr,          # 输入 b
    out_ptr,        # 输出
    n_elements,     # 总元素数（int）
    BLOCK_SIZE: tl.constexpr,   # 编译期常量：每个 program 处理多少元素
    # constexpr 白话：它的值在「翻译时」就写死进代码（不是运行时变量），
    # 翻译器能针对这个值专门优化——代价是换值就得重新翻译一份。
):
    # ── 第 1 步：我是谁？ ────────────────────────────────────────────────
    pid = tl.program_id(0)              # 本 program 在 0 维 grid 里的编号（0,1,2,...）
    # ── 第 2 步：我负责哪段下标？ ────────────────────────────────────────
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 白话：我的编号 × 每人块大小 = 我的起始位置；arange 生成 [0..BLOCK) 的连续偏移。
    # 例：pid=2、BLOCK=1024 → offs = [2048, 2049, ..., 3071]，我专管这一段。
    # ── 第 3 步：⚠️ 边界保护（n 不整除 BLOCK 时防越界）──────────────────
    mask = offs < n_elements
    # 白话：n=4013、BLOCK=1024 时，第 4 个 program 的 offs 里有 83 个 >= n；
    # mask 把它们标 False，读写时跳过——不写这行，越界就是未定义行为。
    # ── 第 4 步：整块读写 ────────────────────────────────────────────────
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)   # 一次读一整块；假下标位置填 other=0.0
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    # TODO(you)：把下面这行补成真正的加法并写回
    #   out = x + y                                # 整块逐元素相加（一行搞定，不用 for）
    #   tl.store(out_ptr + offs, out, mask=mask)   # 写回：假下标位置不写
    # 写完删掉 raise 行，跑 main 对数。
    raise NotImplementedError("第 4 步：补完加法与 store 后删掉本行")


def triton_vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Python 侧包装：准备输出、算出要拉几个工人、启动 kernel。"""
    out = torch.empty_like(x)   # 开一块同样形状的空内存装结果
    n = x.numel()               # numel = number of elements，总元素数
    # grid = 总 program 数：元素数除以 BLOCK，向上取整
    BLOCK = 1024
    # cdiv = ceil division（向上取整除法）：4013/1024=3.92 → 4 个 program
    grid = (triton.cdiv(n, BLOCK),)
    # 启动语法：kernel[grid](参数...)——拉起 grid 里所有 program 跑 kernel
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)
    return out


def main() -> None:
    torch.manual_seed(0)
    # 三种形状：整除 / 不整除 / 极小（网格只有 1 个 program）
    for n in (1024 * 4, 1000 * 4 + 7, 13):
        x = torch.randn(n)
        y = torch.randn(n)
        ref = x + y
        out = triton_vector_add(x, y)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
        print(f"n={n:7d} ✓（grid 大小 = {triton.cdiv(n, 1024)}）")

    # ══ 练习（写完上面再动这些，都会改 harness 本身）══════════════════════
    # TODO(you)：练习 1 —— 把 BLOCK_SIZE 改成 64 和 65536 各跑一遍。
    #   interpret 模式没有性能差异，但回答：grid 大小分别变成多少？
    #   （GPU 上这就是「每个 program 干多少活」的旋钮，W6 autotune 的对象。）
    # TODO(you)：练习 2 —— 写一个 2D 版本：x 形状 [H, W]，program 按「行块」分工，
    #   用 tl.program_id(0) 选行、tl.arange 选列。写完对 torch 的 x+y 验数值。
    #   （这直接就是 02_fused_softmax 的骨架——相当于预习。）


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录 ═══════════════════════════════
# - grid 大小随 BLOCK 的变化（练习 1）：
# - 2D 版我踩的坑（练习 2）：
# - 一句话：program / BLOCK / mask 各自是干嘛的：
