#!/usr/bin/env python3
"""
W5-02 · fused softmax：第一次在 kernel 里做 reduction
=============================================================
先补几句背景（softmax 是什么、为什么要自己写）：
    softmax 把一串数变成「和为 1 的概率」：每个数取 e 的指数，再除以总和。
    它是注意力机制的核心部件（W8 FlashAttention 的主角），
    也是学「归约类 kernel」的最佳入门：算完一整行才能得到结果。

    「归约（reduction）」= 把一串数压缩成少数几个数（如 max、sum）。
    它和 W5-01 的逐元素加法有本质区别：加法各算各的，归约要「全行汇总」。
    所以分工方式也变了：一个工人包一整行。

名词白话：
    - HBM：显存的大仓库（几百 GB 但速度慢）；SM：GPU 内部的工作台
      （速度极快但空间小）。数据必须从 HBM 搬到 SM 才能算——
      搬运速度往往比计算速度更卡脖子（这就是「带宽瓶颈」）；
    - decomposed（拆开版）：用现成 torch 算子分三步算（exp→sum→div）；
    - fused（融合版）：一个 kernel 里一口气算完——中间结果不落回 HBM；
    - 数值稳定：e^1000 会溢出变成 inf；先减去行内最大值再 exp，
      就不会溢出——数学上结果不变（softmax 平移不变）。

目标：
    1. 一个 program 负责一整行——「行级归约」模式（RMSNorm/softmax/layernorm 全家桶同款）；
    2. 数值稳定版 softmax：先减 max 再 exp（为什么？观察记录里回答）；
    3. 理解「融合」到底省了什么：对比 decomposed 三步与 fused 一步的 HBM 流量。

运行（容器内）：
    TRITON_INTERPRET=1 python 02_fused_softmax.py

融合的意义（W6 论证的预演，本周只建立直觉）：
    decomposed（exp→sum→div 三个 torch 算子）：
        x 从 HBM 读进 SM 算 exp 写回 HBM；再读回算 sum 写回；再读回算 div 写回。
        行宽 N 的数据在 HBM 与 SM 之间来回 3 趟。
    fused（一个 kernel）：
        读 1 趟、算完写 1 趟。省下的全是「中间结果回访 HBM」的带宽与延迟。
        ——这就是 plan.md 里「融合省的是 HBM 往返」的具象。
"""
from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    x_stride,                # 行与行之间的跨度（元素数）——行主序下 = N
    # stride 白话：第 2 行的开头在第 1 行开头后的第 x_stride 个元素处。
    # 知道跨度才能用「行号 × 跨度」跳到任意行。
    n_cols,                  # 每行元素数
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)                # 一个 program 包一行：我的编号就是行号
    cols = tl.arange(0, BLOCK_SIZE)       # 本行内的列偏移 [0, 1, ..., BLOCK)
    # ⚠️ 列方向 mask：行宽不足 BLOCK 的部分不能参与 load（other=0 只保 load 不炸，
    #    后续 reduction 的分母还会把它算进去——所以 mask 要贯穿到 exp 那一步）
    mask = cols < n_cols

    # TODO(you)：按下面四步把 kernel 补完（每步一行注释已给出语义）：
    # 1) 行首地址 = x_ptr + row * x_stride；加载本行：x = tl.load(..., mask=mask, other=-float('inf'))
    #    写法：x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=-float('inf'))
    #    （other=-inf 的用意：被屏蔽位置的 exp 结果应为 0，-inf 经 exp 恰好变 0——
    #     但注意 -inf 参与 max 是安全的，参与 sum 前已被 exp 化 0）
    # 2) 行最大值：row_max = tl.max(x, axis=0)   # axis=0：把整个块压成一个数
    # 3) 数值稳定 exp：num = tl.exp(x - row_max)；denom = tl.sum(num, axis=0)
    #    （被屏蔽处 exp(-inf - max)=exp(-inf)=0 ✓ 分母不受污染）
    # 4) 写回：tl.store(out_ptr + row * x_stride + cols, num / denom, mask=mask)
    raise NotImplementedError("按 TODO 四步补完，然后删掉本行")


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """启动器：一行一个 program，总 program 数 = 行数。"""
    assert x.ndim == 2 and x.is_contiguous()   # 只支持二维连续内存（断言：条件不满足直接报错）
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n_cols)     # reduction 的 BLOCK 取 2 的幂
    # 为什么必须 2 的幂：tl.arange 的长度限制（硬件向量化的要求）；
    # 代价：行宽 513 → BLOCK=1024，近一半算力浪费在假下标上——
    # 这就是「行很宽时需要分块循环归约」的原因（进阶话题，见尾部 TODO 2）。
    softmax_kernel[(n_rows,)](x, out, x.stride(0), n_cols, BLOCK_SIZE=BLOCK)
    return out


def decomposed_softmax(x: torch.Tensor) -> torch.Tensor:
    """三算子版（eager 世界）：给 W6 留的对照口径。"""
    e = torch.exp(x)
    return e / e.sum(dim=1, keepdim=True)


def main() -> None:
    torch.manual_seed(0)
    for shape in [(4, 128), (3, 129), (8, 1024)]:
        # 三种形状故意选的：整除 / 不整除（mask 生效）/ 宽行
        x = torch.randn(*shape) * 3.0          # 乘 3 放大数值，逼出稳定性问题
        ref = torch.softmax(x, dim=1)          # 官方实现当标准答案

        out_f = triton_softmax(x)              # 我的融合版
        torch.testing.assert_close(out_f, ref, rtol=1e-5, atol=1e-5)
        out_d = decomposed_softmax(x)          # 三算子拆开版
        torch.testing.assert_close(out_d, ref, rtol=1e-5, atol=1e-5)
        print(f"shape={shape}  fused ✓  decomposed ✓")   # 三方对数全过才算数

    # 极端数值稳定性：大值行（不做 max-subtraction 的 naive 版会 inf）
    # 白话：e^1000 远超 float32 上限（约 e^88）→ 直接溢出报 inf；
    # 先减 max 后最大指数只到 e^2 —— 这就是「先减 max」的必要性实证。
    x = torch.tensor([[1000.0, 1001.0, 1002.0]])
    torch.testing.assert_close(triton_softmax(x), torch.softmax(x, dim=1), rtol=1e-5, atol=1e-5)
    print("大值行数值稳定 ✓")

    # TODO(you)：回答（观察记录）：
    #  1) 如果第 1 步的 other 给 0.0 而不是 -inf，哪一步会算错？（提示：sum 分母）
    #  2) BLOCK 必须 ≥ 行宽吗？行宽 5000 时 next_power_of_2 是多少，浪费多少？
    #     （预告：处理超宽行要用「循环分块 reduction」——RMSNorm 在 LLM 里行宽
    #      最多 128k，但那已是 W6 之后的进阶话题。）
    #  3) decomposed 版对 HBM 的读写趟数是几？fused 是几？（为 W6 做账）


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录 ═══════════════════════════════
# - other=0.0 会在哪一步算错：
# - HBM 趟数账（decomposed vs fused）：
# - 行级 reduction 模式的一句话总结：
