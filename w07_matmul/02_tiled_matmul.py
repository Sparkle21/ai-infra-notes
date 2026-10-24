#!/usr/bin/env python3
"""
W7-02 · tiled matmul：K 维循环分块（tutorial 的主形态）
=============================================================
先补几句背景（01 留下的悬念）：
    01 的第三个用例（K=48 > BLOCK_K=32）是**故意失败**的：naive 版把 K 维
    一次读完，只能读到前 32 个——剩下 16 个直接丢了，答案自然错。
    修法就是本文件：K 维不要一次读完，用 for 循环每次搬一小块、
    算一小块、把部分积累加进 acc；循环全部转完，acc 才是完整答案。
    这也是所有真实 GEMM kernel 的主形态（K 经常是几千，不可能一次进工作台）。

名词白话：
    - SRAM：每个 SM 自带的「随身工作台」——极快但极小（几十 KB），
      tl.load 搬进来的 tile 就待在这里，tl.dot 从这里取数；
    - L2 cache：全 GPU 共享的「中转仓」——比 HBM（大仓库）快、
      比 SRAM（工作台）大，各 SM 都能来拿，适合存大家共用的数据。

tiling 到底在省什么（本周最重要的一段直觉）：
    不分块时算 C 的一个 tile 要从 HBM 读：BLOCK_M*K + BLOCK_N*K 个元素；
    分块后每个 k 块的 a-tile/b-tile 在 SM 的 SRAM 里被**本 program 独占重用**，
    数据从 HBM 进 SRAM 一次，被 dot 消费多次的潜力靠 BLOCK 之间的组织实现
    （跨 program 的重用靠 L2——组内 tile 的排布顺序决定 L2 命中率，
    这就是 tutorial 里 grouped ordering 的动机，进阶题见文件尾）。

    本质：把「访存 × 计算」的比例从 O(K) 降到 O(K/BLOCK_K)×BLOCK……
    ——别背公式，观察记录里用自己的话推一遍。

运行（容器）：
    TRITON_INTERPRET=1 python 02_tiled_matmul.py
"""
from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def tiled_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 二维分工同 01：我是第几行块、第几列块
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)     # 本 tile 的行下标
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)     # 本 tile 的列下标
    rk = tl.arange(0, BLOCK_K)                       # 圈内相对 K 下标（0..BLOCK_K-1）
    mask_m = rm < M                                  # 行越界保护
    mask_n = rn < N                                  # 列越界保护

    # 累加器先清零：K 循环每一圈往里加一笔部分积，转完所有圈才是完整答案
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ══ K 维循环：每次搬运一个 [BLOCK_M, BLOCK_K] 和 [BLOCK_K, BLOCK_N] ═══
    # 白话：k0 是本圈 K 维的起点，一圈前进 BLOCK_K，直到把整条 K 走完。
    # 例：K=48、BLOCK_K=32 → k0 取 0 和 32 两圈；第二圈只有 16 个真元素，
    # 剩下 16 个靠 mask_k 填 0（0 乘任何数 = 0，不污染累加结果）。
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + rk              # 本圈的绝对 K 下标 = 圈起点 + 圈内偏移
        mask_k = k_idx < K            # ⚠️ 最后一圈可能不满块（每圈都要重算）
        # TODO(you)：四步（大部分是 01 的搬运，多了「k_idx 偏移」）：
        # 1) offs_a = rm[:, None] * stride_am + k_idx[None, :] * stride_ak
        #    a = tl.load(a_ptr + offs_a, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        #    白话：与 01 唯一的差别是 rk 换成 k_idx——第 1 圈下标是 0..31，
        #    第 2 圈自动变成 32..63，「循环挪人」就完成了 K 维的推进。
        # 2) offs_b = k_idx[:, None] * stride_bk + rn[None, :] * stride_bn
        #    b = tl.load(b_ptr + offs_b, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        # 3) acc = tl.dot(a, b, acc)          # 累加器跨循环圈累积（fp32）
        #    白话：acc 像记账本，每圈记一笔部分积；圈圈相加 = 完整乘加。
        # 4) 循环结束后（注意缩进：回到循环**外**）：store 同 01 第 6 步——
        #    offs_c = rm[:, None] * stride_cm + rn[None, :] * stride_cn
        #    tl.store(c_ptr + offs_c, acc, mask=mask_m[:, None] & mask_n[None, :])
        #    为什么 store 放循环外？输出 C 的每个格子要在**整条 K 走完后**
        #    才有最终值——中途写出去的是没算完的半成品。
    raise NotImplementedError("循环体补完后删掉本行（store 放循环外）")


def triton_matmul(a: torch.Tensor, b: torch.Tensor,
                  BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 32) -> torch.Tensor:
    """启动器：三个 BLOCK 想调大调小就在这里换（进阶 1 的实验开关）。"""
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    tiled_matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


def main() -> None:
    torch.manual_seed(0)
    cases = [
        (64, 64, 32),        # 01 的老朋友（单圈循环）
        (48, 80, 32),        # M/N 不整除
        (64, 64, 48),        # K 不整除 BLOCK_K（01 的教具用例——循环版必须过！）
        (128, 96, 256),      # K 跨多个块（多圈循环的考场）
        (333, 217, 155),     # 全不整除（终极边界）
    ]
    for i, (M, N, K) in enumerate(cases):
        a = torch.randn(M, K)
        b = torch.randn(K, N)
        out = triton_matmul(a, b)
        torch.testing.assert_close(out, a @ b, rtol=1e-2, atol=1e-2)
        print(f"case{i} M={M} N={N} K={K} ✓")

    # ══ 进阶（S4 前选做，直接关系「≥ cuBLAS 70%」能不能达成）══════════════
    # TODO(you)：进阶 1 —— BLOCK 大小敏感性：把 BLOCK_M/N/K 在
    #   {32,64,128}×{32,64,128}×{16,32,64} 里挑 4 组跑 GPU（S4），
    #   interpret 模式只验正确性。记录哪组在 H20 上最快 → tiling 参数不外推的第一手证据。
    # TODO(you)：进阶 2 —— grouped ordering：tutorial 的「GROUP_M」技巧——
    #   pid_m/pid_n 不再按行主序映射，而是按小组分组，让相邻 program 共享 a-tile
    #   提高 L2 命中。读懂 tutorial 这 10 行，把它抄进本文件并注明每行含义
    #   （S4 的性能台阶之一；解释不清就写进「看不懂清单」）。


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录 ═══════════════════════════════
# - K=48 用例在循环版过了吗（01→02 的意义所在）：
# - tiling 省了什么（我自己的话，≤3 句）：
# - BLOCK 组合初测（S4 回填）：哪组最快 / 和 cuBLAS 的比值：
# - grouped ordering 我读懂了吗 / 没懂的行：
