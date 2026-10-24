#!/usr/bin/env python3
"""
W7-03 · BF16 与 Tensor Core：dtype 是最大的性能旋钮
=============================================================
先补几句背景（为什么 dtype 是「最大」的旋钮）：
    同一个 kernel、同一块卡，只把数据类型从 fp32 换成 bf16，
    速度能差好几倍——这不是玄学，是硬件层面的原因：
    fp32 走普通算术单元（一次一个乘加）；bf16 输入走 Tensor Core
    （一次算一整块矩阵乘）。「选 dtype」实际上是在选**走哪条硬件车道**。

名词白话：
    - fp32 / bf16：浮点数的存储格式。fp32 = 32 位（1 符号 + 8 指数 + 23 尾数），
      精度高、占地大；bf16 = 16 位（1+8+7）——**指数位与 fp32 一样多
      （数值范围不缩水），尾数砍到 7 位（精度约 2–3 位十进制）**，
      牺牲精度换速度与省显存；
    - FMA / TF32：FMA = 融合乘加（普通单元的干活方式）；TF32 = 把 fp32
      截成 19 位喂给 Tensor Core 的折中档位（A100 起支持）；
    - TFLOPS：每秒万亿次浮点运算——GPU 算力的计分单位。

本文件的三级阶梯（S4 会话在 GPU 上跑，容器里先验数值）：
    L1  fp32 naive dot（02 的 kernel，BLOCK 调小）        → 慢
    L2  fp32 tl.dot（02 原样）                            → 走 FP32 FMA / TF32 路径
    L3  bf16 输入 + fp32 acc                              → Tensor Core 路径

数值口径（重要——BF16 不是 fp32 的近似，是另一种舍入）：
    BF16：8 位指数 + 7 位尾数 → 有效数字约 2–3 位十进制。
    白话：bf16 不是「把 fp32 算得更粗」，而是「存进去之前就把尾数砍短了」——
    误差发生在**进门那一刻**，不是计算过程中慢慢攒出来的。
    参考答案的做法：把 a/b 转 bf16 后，再**升回 fp32 做参考乘加**
    （这样对比的只是「输入量化误差」，不含累加误差——分因变量：
    一次只变一个因素、另一个先固定住，才知道误差该赖谁）。
    容差需要自己定：先跑 rtol=1e-2，观察最大相对误差，再决定收录的容差并写进笔记。

运行（容器）：
    TRITON_INTERPRET=1 python 03_bf16_tensor_core.py
    S4：python 03_bf16_tensor_core.py --device cuda --bench
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import time  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def bf16_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 与 02 的 tiled 版结构完全一致——差异只有两处：
    #   1) load 出来后转 bf16：a = a.to(tl.bfloat16)
    #      （生产写法是整个 tensor 直接存 bf16，这里演示「在 kernel 里转」的两种姿势都见过）
    #   2) acc 仍是 fp32——Tensor Core 的原生契约就是 bf16 进、fp32 累加。
    #      白话：**存得粗可以、算得粗不行**——乘加要累 K 次，中间结果
    #      若只有 16 位精度会越累越歪；硬件也正因此把 TC 设计成 bf16 进 fp32 出。
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    mask_m = rm < M
    mask_n = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + rk
        mask_k = k_idx < K
        # TODO(you)：按 02 写好的模式补完，唯一新增动作是两处 .to(tl.bfloat16)：
        # a = tl.load(...)   →  a = a.to(tl.bfloat16)     # 进门时砍尾数
        # b = tl.load(...)   →  b = b.to(tl.bfloat16)
        # acc = tl.dot(a, b, acc)                          # TC：bf16 进 fp32 出
        # 循环外 store 同 02；写回时 Triton 自动把 acc 转成 c 的 dtype（fp32）。
    raise NotImplementedError("补完后删本行；store 放循环外，写回前 acc 可保持 fp32（c 是 fp32）")


def run_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    bf16_matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return c


def bench(fn, iters: int, device: str) -> float:
    """计时：先预热 5 次再抹表（W1 学过的坑——首调用含编译/缓存成本，不算数）。"""
    for _ in range(5):
        fn()          # 预热：把编译、缓存分配这些一次性成本先花掉
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--bench", action="store_true", help="S4：跑性能阶梯")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    torch.manual_seed(0)
    M = N = K = 1024
    a = torch.randn(M, K, device=args.device)
    b = torch.randn(K, N, device=args.device)

    # ══ 数值（分因变量：只测输入量化误差）═════════════════════════════════
    a_q = a.to(torch.bfloat16).to(torch.float32)    # 先量化再升回 = 「输入已是 bf16」的世界
    b_q = b.to(torch.bfloat16).to(torch.float32)    # 同上
    ref = a_q @ b_q
    out = run_bf16(a, b)
    diff = (out - ref).abs().max().item()
    rel = diff / ref.abs().max().item()
    print(f"max abs diff = {diff:.4e}   max rel err ≈ {rel:.2e}")
    # TODO(you)：据此定容差并断言（初判 1e-2 量级；记录你实测到的量级）：
    # torch.testing.assert_close(out, ref, rtol=?, atol=?)
    # 定容差的依据写进观察记录——「我为什么敢用这个数」。

    if args.bench:
        # ══ 性能阶梯（S4）══════════════════════════════════════════════════
        # 总浮点运算量：每个输出元素要 K 次「乘+加」（= 2 flops），
        # 共 M*N 个输出元素 → 2*M*N*K。这是「算力吃到几成」的公比分母。
        flops = 2 * M * N * K
        rows = []
        rows.append(("torch.matmul fp32", bench(lambda: torch.matmul(a, b), args.iters, args.device)))
        ab, bb = a.to(torch.bfloat16), b.to(torch.bfloat16)   # 真 bf16 输入（与 mine 同赛道）
        rows.append(("torch.matmul bf16", bench(lambda: torch.matmul(ab, bb), args.iters, args.device)))
        rows.append(("mine bf16", bench(lambda: run_bf16(a, b), args.iters, args.device)))
        print(f"\nshape={M}x{K}x{N}  (TFLOPS = 2MNK/t)")
        for name, ms in rows:
            print(f"  {name:20s} {ms:8.3f} ms   {flops / (ms / 1e3) / 1e12:6.1f} TFLOPS")
        # 里程碑判定：mine bf16 的 TFLOPS ≥ torch.matmul bf16 的 70%？
        # 没到的话：先查 BLOCK 组合（02 进阶 1），再查是否真的走了 TC（ncu 里看
        # tensor pipe 利用率——S4 清单第 3 步）。
        # H20 BF16 理论 ≈148 TFLOPS：你吃到了几成？


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录 ═══════════════════════════════
# - 实测最大相对误差 / 我定的容差与依据：
# - S4 阶梯数字（fp32 matmul / bf16 matmul / mine）：
# - mine 相对 cuBLAS 的百分比（里程碑：≥70%）：
# - ncu 证据：走没走 Tensor Core：
