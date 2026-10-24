#!/usr/bin/env python3
"""
W7-01 · naive matmul：第一次用 tl.dot
=============================================================
先补几句背景（为什么从 matmul 开始、tl.dot 是什么）：
    矩阵乘法（matmul/GEMM）是深度学习里最重的计算——LLM 里几乎每个
    大层（QKV 投影、FFN）都是它。W5 的归约/逐元素是「搬运瓶颈」，
    matmul 是「计算瓶颈」，两类 kernel 的优化思路完全不同。

    自己写 matmul 的核心工具是 tl.dot（Triton 的矩阵乘指令）：
    它能把小矩阵块的乘加直接映射到 GPU 的专用矩阵计算单元
    （Tensor Core——算矩阵乘比普通算术单元快一个量级）。

名词白话：
    - tile（块）：把大矩阵切成的小方块。大矩阵直接算放不下，
      必须切成 [BLOCK_M × BLOCK_K]、[BLOCK_K × BLOCK_N] 的小块搬进工作台；
    - 累加器（acc/accumulator）：存「乘累加到一半」结果的寄存器块。
      C = A@B 要沿 K 维累加很多次部分积，acc 就是那个记账本；
    - stride（跨度）：同一维度上相邻元素隔多远（W5-02 讲过，这里是二维版）；
    - 2D grid：本文件 grid 有两个维度（行块数 × 列块数），
      program_id(0) 是行块号，program_id(1) 是列块号。

任务：C = A @ B，A[M,K] @ B[K,N] → C[M,N]。

naive 版的形态：一个 program 负责输出**一个 tile**（BLOCK_M × BLOCK_N），
把 K 维一次性整块读进来（所以只适用于 K ≤ BLOCK_K 的小形状——这是刻意的，
02 会把它推广成循环）。先把 tl.dot 的三条规矩学会：

    1. tl.dot(a, b, acc) 的 a/b 是 2D block（[BLOCK_M, BLOCK_K] × [BLOCK_K, BLOCK_N]），
       返回 [BLOCK_M, BLOCK_N]；三边都得 ≥16（Tensor Core 的形状约束）；
       白话：硬件一次至少吃 16×16×16 的矩阵块，小于 16 不接单。
    2. acc 是累加器：tl.dot(a, b, acc) 等价 acc += a @ b——K 循环分块时靠它累积；
    3. dot 的输入 dtype 决定走哪条硬件路径：fp32(+allow_tf32) / bf16 / int8——
       W10 的量化 kernel 就是换 dtype 这一件事。

运行（容器）：
    TRITON_INTERPRET=1 python 01_naive_matmul.py
"""
from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def naive_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,   # a 的行/列跨度：元素 a[m,k] 在 a_ptr + m*stride_am + k*stride_ak
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 二维分工：我是第几行块、第几列块 → 我负责输出里的哪个小方块
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)     # 本 tile 的行下标（一串 BLOCK_M 个）
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)     # 本 tile 的列下标
    rk = tl.arange(0, BLOCK_K)                       # K 维一次性读完（naive 的含义）

    # ⚠️ mask：三组下标各自可能越界（M/N/K 不整除 BLOCK 时）
    mask_m = rm < M
    mask_n = rn < N
    mask_k = rk < K

    # TODO(you)：六步补完：
    # 1) a 的 2D 下标与 mask：offs_a = rm[:, None] * stride_am + rk[None, :] * stride_ak
    #    mask_a = mask_m[:, None] & mask_k[None, :]；tl.load(a_ptr + offs_a, mask=mask_a, other=0.0)
    #    白话：rm[:,None] 把行号竖起来（列向量）、rk[None,:] 把 k 号横过去（行向量），
    #    两者相加 = 一个 [BLOCK_M, BLOCK_K] 的下标矩阵——每个格子里是 a 中一个元素的位置。
    #    这套「竖向量+横向量=下标矩阵」是 Triton 二维寻址的标准姿势，后面反复用。
    # 2) b 同理：offs_b = rk[:, None] * stride_bk + rn[None, :] * stride_bn
    #    mask_b = mask_k[:, None] & mask_n[None, :]；b = tl.load(b_ptr + offs_b, mask=mask_b, other=0.0)
    # 3) acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # 累加器必须 fp32！
    #    白话：乘加一直在累，中间若用 16 位存会慢慢丢精度——所以累加器固定用 32 位。
    # 4) acc = tl.dot(a, b, acc)     # ← 三规矩同时上身（acc += a @ b）
    # 5) offs_c = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    # 6) tl.store(c_ptr + offs_c, acc, mask=mask_m[:, None] & mask_n[None, :])
    raise NotImplementedError("六步补完后删掉本行")


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """启动器：二维 grid（行块数 × 列块数），每格一个 program。"""
    assert a.ndim == 2 and b.ndim == 2 and a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K2, N = b.shape
    assert K == K2        # 矩阵乘的合法条件：a 的列数 = b 的行数
    c = torch.empty(M, N, device=a.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32       # 三边都 ≥16（dot 约束）
    # grid 两维：第 0 维管输出切成几条行带，第 1 维管切成几条列带
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    naive_matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


def main() -> None:
    torch.manual_seed(0)
    # 形状刻意带「不整除」与「K 大于 BLOCK」之外的边界：
    # 第三个用例 K=48 > BLOCK_K=32 —— naive 版**读不完整条 K**，预期会错！
    # 它的使命就是让你亲眼看到「为什么需要 02 的 K 循环」。跑完看断言怎么失败的。
    cases = [
        (64, 64, 32),      # 整除 ✓
        (48, 80, 32),      # M/N 不整除 ✓（mask 的考场）
        (64, 64, 48),      # K > BLOCK_K ✗（预期失败——本文件的教具）
    ]
    for i, (M, N, K) in enumerate(cases):
        a = torch.randn(M, K)
        b = torch.randn(K, N)
        ref = a @ b
        try:
            out = triton_matmul(a, b)
            torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
            print(f"case{i} M={M} N={N} K={K} ✓")
        except AssertionError:
            print(f"case{i} M={M} N={N} K={K} ✗ 数值不一致（K > BLOCK_K 的教具用例，预期失败）")
        except NotImplementedError:
            print(f"case{i} M={M} N={N} K={K} —— kernel 还没写完")
            return

    # TODO(you)（观察记录）：
    #  1) case2 失败的具体形态：是完全错、还是部分行/列对？（把 diff 打出来看）
    #  2) 为什么 mask_k 的 other=0.0 在 K=32 时无害，在 K=48 时致命？
    #     （0 参与 dot 的累加应该是无害的……真相是「根本没读到后 16 个 k」——
    #      记住这个坑，02 的循环版就是它的解。）
    #  3) 如果把 BLOCK_K 提到 64，case2 会过吗？跑一下验证你的推理。


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录 ═══════════════════════════════
# - case2 的失败形态（diff 长什么样）：
# - BLOCK_K=64 后 case2 的结果：
# - tl.dot 三规矩我自己的复述：
