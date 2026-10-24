#!/usr/bin/env python3
"""
W6-02 · autotune：BLOCK/num_warps 扫描（S3 会话用）
=============================================================
先补几句背景（autotune 是什么、为什么需要）：
    W5 的 kernel 里 BLOCK_SIZE 是写死的（1024）。但「每人干多少活最快」
    取决于形状和硬件——没有银弹。工业界的解法：让机器自己试。

    @triton.autotune 就是「自动试配置」开关：你给一份候选清单
    （如 BLOCK=1024/2048/4096 × 不同线程组数），它对每个新形状
    把候选全跑一遍、掐表、记住最快的那个，之后直接复用。

名词白话：
    - num_warps：一个 program 里线程组的数量（32 线条/组）——
      组越多并行越粗，但也更耗资源，同样要试；
    - tuning（调参）：把候选配置逐个实测选最优的过程；
    - key：触发重新调参的「形状指纹」——key 里列的参数变了
      （比如列数从 4096 变 11008），才重新试一轮。

目标：
    学会工业界标准姿势——@triton.autotune 装饰器：
      - 你给一组候选配置（BLOCK、num_warps、num_stages）；
      - Triton 对每个形状自动实测一遍、缓存最优；
    并理解它的两个坑：
      坑1 tuning 有编译+试跑成本（首调用慢——S3 里别把这算进 benchmark）；
      坑2 「最优」只对当前形状负责（H20 的结论不外推，plan.md 已警告）。

运行：
    Mac/容器（只验证装饰器语法与接线，数字无意义）：
        TRITON_INTERPRET=1 python 02_autotune.py --device cpu --iters 2
    S3：
        python 02_autotune.py --device cuda --iters 200

注意：
    autotune 在 interpret 模式下行为退化（不真计时，直接选第一个配置）——
    这没关系，本文件的 Mac 验收标准是「不报错地走完流程」。
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
import triton
import triton.language as tl


# ══ 带自动调参的 kernel：拷贝自 w05/03（独立成册，避免跨文件加载两份）════
@triton.autotune(
    configs=[
        # 每条 Config = 一个候选：字典里是 kernel 的 constexpr 参数值，
        # 后面跟的 num_warps 是启动参数。tuning = 逐条试跑选最快。
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        # TODO(you)（S3 前扩候选）：
        #   再加 3–4 条：BLOCK ∈ {512, 8192} × num_warps ∈ {2, 16} 的组合，
        #   并思考：11008 列的形状，BLOCK=4096 意味着什么？（hint: next_power_of_2(11008)=16384，
        #   低于行宽的 BLOCK 会漏列！autotune 的 key 里为什么必须放 n_cols？）
    ],
    key=["n_cols"],        # ← 形状变了必须重新 tune（这个 key 漏掉 = 拿旧配置跑新形状，错误答案）
    # 白话：key 就是「什么变了才重选」。列宽一变，最优 BLOCK 大概率也变。
)
@triton.jit
def rmsnorm_autotune_kernel(
    x_ptr, w_ptr, out_ptr, stride, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # TODO(you)：把 w05/03 里你写好的 kernel 主体搬进来（原样搬运，别改逻辑）。
    # 提示：搬运后注意 BLOCK_SIZE 现在可能 < 行宽——这意味着需要「循环分块」版
    # （W5 进阶题）。如果只搬单块版，请把候选里的 BLOCK 全部 ≥ 16384 或按形状约束。
    raise NotImplementedError("搬入 kernel 主体后删除本行")


def run(device: str, iters: int) -> None:
    import time

    for n_rows, n_cols in [(8192, 4096), (8192, 11008)]:
        x = torch.randn(n_rows, n_cols, device=device)
        w = torch.randn(n_cols, device=device)
        out = torch.empty_like(x)

        # 第一次调用 = tuning（贵，不计入）
        # 白话：autotune 是懒的——第一次见到这个 key（n_cols）才试跑所有候选，
        # 这一步包含多次编译+计时，很慢很正常；选完就缓存，后续调用直接用。
        rmsnorm_autotune_kernel[(n_rows,)](x, w, out, x.stride(0), n_cols, 1e-6)

        best = rmsnorm_autotune_kernel.best_config   # 问它：最后选了哪个配置？
        print(f"shape=[{n_rows},{n_cols}]  best_config = {best}")

        # 稳态计时
        for _ in range(5):
            rmsnorm_autotune_kernel[(n_rows,)](x, w, out, x.stride(0), n_cols, 1e-6)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            rmsnorm_autotune_kernel[(n_rows,)](x, w, out, x.stride(0), n_cols, 1e-6)
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
        print(f"           稳态均次 = {ms:.3f} ms")
        # TODO(you)（S3）：记录每个形状的 best_config 与固定 BLOCK（w05 版）的差距——
        # autotune 值几个百分点？这个答案决定你对「写死配置」的态度（幼稚清单素材）。


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    if args.device == "cpu":
        import os

        os.environ.setdefault("TRITON_INTERPRET", "1")
    run(args.device, args.iters)


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录（S3 回填）══════════════════════
# - 各形状 best_config：
# - autotune 相对固定配置的提升幅度：
# - key 漏写 n_cols 会发生什么（设计推演，不用真试）：
