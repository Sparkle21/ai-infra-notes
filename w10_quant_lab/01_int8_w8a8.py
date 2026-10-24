#!/usr/bin/env python3
"""
W10-01 · INT8 W8A8 matmul：把 W7 的积木拼起来
=============================================================
先补几句背景（量化是什么、为什么值得）：
    模型推理的两大成本是「算得慢」和「存得多」，两者都跟数字的位宽
    挂钩：fp32 每个数 4 字节，int8 每个数 1 字节。量化 = 把浮点数
    压成小整数（外带一个缩放系数），一次压缩换来：显存÷4、
    搬运÷4、int8 算力峰值还比 bf16 高一倍。代价是精度损失——
    所以每个量化方案都要过「数值关 + 性能关」两道测试。

名词白话：
    - W8A8：权重和激活都量化到 int8（Weights 8-bit, Activations 8-bit）；
      只压权重的叫 W8A16，难度低一档；
    - 激活（activation）：层与层之间流动的中间数据——就是这里的 x；
    - per-tensor scale：整个张量共用一个缩放系数（最简单的档位；
      更细的 per-channel/per-group 是进阶，W9 笔记展开）；
    - epilogue：kernel 主循环（乘加）之后的收尾步骤——反量化缩放、
      加 bias 等轻活都放这里，一次算完不多跑一趟；
    - clamp（夹住）：把数限制在给定范围内——这里是 [-127,127] 防溢出
      （-128 留着不用是对称量化的惯例，保证 0 可精确表示）。

任务：
    y = x @ W^T        x: [M, K] 激活，W: [N, K] 权重（row-major 语义同 nn.Linear）

    INT8 W8A8（per-tensor scale）的算法：
      1) 量化：  x_i8 = round(x / s_x).clamp(-127, 127)     s_x = 127 / max|x|
                 w_i8 = round(w / s_w).clamp(-127, 127)
                 白话：除以 scale 把整个张量「塞进」[-127,127] 的整数格子——
                 最大的数正好落在 127，其它数按比例缩小。
      2) kernel：acc = dot(x_i8, w_i8^T)  —— int8 输入、fp32 累加（W7-01 规矩）
      3) epilogue：y = acc * (s_x * s_w)  —— scale 是标量，最后一步乘（W9-01 的推导）
                 白话：量化时除了 s_x、s_w，乘回来时合并成一次乘法还原量纲——
                 两个除法在整数世界外面合并，几乎白送。

    你会发现：这个 kernel = W7 的 tiled matmul（dtype 换 int8）+ W5 的 epilogue 广播。
    没有新知识，只有组合——这就是设计意图（期末考试性质）。

运行（容器）：
    TRITON_INTERPRET=1 python 01_int8_w8a8.py
    S5：python 01_int8_w8a8.py --device cuda --bench
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import time  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


# ══ 量化 / 反量化（host 侧，W9-01 的公式落地）════════════════════════════
def quantize_per_tensor(t: torch.Tensor) -> tuple[torch.Tensor, float]:
    """返回 (int8 tensor, scale)。scale = 127 / max|t|。"""
    # TODO(you)：三行——amax、scale、round+clamp（torch.round / torch.clamp）。
    # 写法提示：
    #   amax = t.abs().amax()                       # 全张量最大绝对值（一个数）
    #   scale = 127.0 / amax                         # 最大值将映射到 127
    #   t_i8 = torch.round(t / scale).clamp(-127, 127).to(torch.int8)
    #   return t_i8, scale
    raise NotImplementedError


def dequantize(t_i8: torch.Tensor, scale: float) -> torch.Tensor:
    # 反量化 = 整数升回浮点再乘回 scale。只在「量化世界的参考答案」里用；
    # 真推理不单独跑这一步——kernel epilogue 里一步做完。
    return t_i8.float() * scale


# ══ INT8 kernel：W7-02 的变奏（dtype 与 epilogue 不同）════════════════════
@triton.jit
def int8_matmul_kernel(
    x_ptr, w_ptr, y_ptr,
    M, N, K,
    sx, sw,                            # 两个 per-tensor scale
    stride_xm, stride_xk,
    stride_wn, stride_wk,              # 注意：W 是 [N, K]（nn.Linear 语义）
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 二维分工同 W7：我是第几行块、第几列块
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)     # 本 tile 的行下标
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)     # 本 tile 的列下标（输出维 N）
    rk = tl.arange(0, BLOCK_K)                       # 圈内相对 K 下标
    mask_m = rm < M                                  # 行越界保护
    mask_n = rn < N                                  # 列越界保护

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + rk
        mask_k = k_idx < K
        # TODO(you)：四步（与 W7-02 逐行同构，差异已注释）：
        # 1) x = tl.load(...)   # [BLOCK_M, BLOCK_K]；other=0——0 参与乘加无害
        #    offs_x = rm[:, None] * stride_xm + k_idx[None, :] * stride_xk
        #    x = tl.load(x_ptr + offs_x, mask=mask_m[:, None] & mask_k[None, :], other=0)
        # 2) w = tl.load(...)   # [BLOCK_N, BLOCK_K]！W 的行是输出维（转置读）
        #    → offs_w = rn[:, None] * stride_wn + k_idx[None, :] * stride_wk
        #    w = tl.load(w_ptr + offs_w, mask=mask_n[:, None] & mask_k[None, :], other=0)
        # 3) acc = tl.dot(x, tl.trans(w), acc)     # ← W7 没有的新 API：tl.trans
        #    （或者按 [BLOCK_K, BLOCK_N] 读 w：offs_w = k_idx[:, None]*stride_wk + rn[None, :]*stride_wn，
        #     然后 acc = tl.dot(x, w, acc)——两种都行，体会「读的布局决定 dot 的姿势」）
        #    白话：dot 不需要手动转 dtype——int8 进、fp32 累加是 TC 原生契约（同 W7）。
        # 4) 循环外 epilogue：acc * (sx * sw) 是标量乘——直接写进 store 的表达式里
        #    offs_y = rm[:, None] * stride_ym + rn[None, :] * stride_yn
        #    tl.store(y_ptr + offs_y, acc * (sx * sw),
        #             mask=mask_m[:, None] & mask_n[None, :])
    raise NotImplementedError("四步补完后删掉本行")


def int8_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y = x @ w^T 的 INT8 全流程（量化 → int8 kernel → epilogue 反量化）。返回 fp32 结果。"""
    x_i8, sx = quantize_per_tensor(x)
    w_i8, sw = quantize_per_tensor(w)
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty(M, N, dtype=torch.float32, device=x.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    int8_matmul_kernel[grid](
        x_i8, w_i8, y, M, N, K, sx, sw,
        x_i8.stride(0), x_i8.stride(1),
        w_i8.stride(0), w_i8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return y


def bench(fn, iters: int, device: str) -> float:
    """计时：预热 5 次 + cuda.synchronize 前后包夹（W1 三坑的标准对策）。"""
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
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    torch.manual_seed(0)
    M, K, N = 512, 1024, 1024
    x = torch.randn(M, K, device=args.device)
    w = torch.randn(N, K, device=args.device)

    # ══ 数值：与「量化语义的参考实现」比（分因变量，同 W7-03 的方法论）════
    x_i8, sx = quantize_per_tensor(x)
    w_i8, sw = quantize_per_tensor(w)
    ref = dequantize(x_i8, sx) @ dequantize(w_i8, sw).T   # 量化后世界的精确答案
    # 白话：参考答案也「先量化再算」——这样对比的就只剩 kernel 本身的对错，
    # 不把量化固有误差算进 kernel 的账（分因变量，同 W7-03 的方法论）。
    out = int8_linear(x, w)
    # TODO(you)：观察最大相对误差，自定容差断言（INT8 量化误差比 BF16 大得多，
    # rtol 1e-1 量级都是正常范围——「我为什么敢用这个数」写进观察记录）：
    diff = (out - ref).abs().max().item()
    print(f"max abs diff vs 量化参考 = {diff:.4e}")
    # torch.testing.assert_close(out, ref, rtol=?, atol=?)

    if args.bench:
        # 总运算量同 W7-03 口径；注意 int8 那行打印单位是 TOPS
        # （整数运算也算 operations——和浮点的 TFLOPS 是两个计分板）
        flops = 2 * M * N * K
        t_i8 = bench(lambda: int8_linear(x, w), args.iters, args.device)
        xb, wb = x.to(torch.bfloat16), w.to(torch.bfloat16)
        t_bf = bench(lambda: torch.nn.functional.linear(xb, wb), args.iters, args.device)
        print(f"\nmine int8 : {t_i8:8.3f} ms  {flops / (t_i8 / 1e3) / 1e12:6.1f} TOPS")
        print(f"bf16 linear: {t_bf:8.3f} ms  {flops / (t_bf / 1e3) / 1e12:6.1f} TFLOPS")
        # TODO(you)（S5）：INT8 理论峰值是 BF16 的 2 倍——实测比值？
        # 不到 2x 的差距去哪了？（quant 的 host 开销混进来了吗？——用 --kernel-only
        # 口径再测一遍，把量化开销与 kernel 开销分开算。）
        # ncu 的 SOL 段落（S5 清单第 3 步）：int8 kernel 是算力受限还是带宽受限？
        # 用 H20 roofline 拐点 37 FLOP/Byte 判断。


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录 ═══════════════════════════════
# - 量化误差量级 / 我定的容差与理由：
# - tl.trans vs 转置读 两种写法我选了哪种、为什么：
# - S5：int8 vs bf16 的 TOPS/TFLOPS 比值：
# - S5：ncu SOL 结论（受限于什么）：
