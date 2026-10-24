#!/usr/bin/env python3
"""
W1-04 · eager vs compiled 计时 harness（S1 租卡会话的主脚本）
=================================================================
先补一句背景：
    这个文件回答一个问题——「编译版到底比不编译快多少？」
    但计时这件事陷阱很多，量错了会得出荒唐结论。本文件是一个
    「诚实计时」的示范，也是之后所有 benchmark（S1–S5 会话）的模板。

目标：
    一个诚实的计时 harness：Mac 上空跑验证逻辑；S1 上拿 GPU 数字。
    诚实的意思（三个坑及其对策，记熟了受益全程）：
      - 坑 1：首次调用的编译时间很慢（秒级），混进去会把稳态性能测得一塌
        糊涂 → 对策：warmup（预热）与正式计时分开，先白跑几次再开始掐表；
      - 坑 2：GPU 是异步执行的——你喊「开始算」后它只是记下任务就立刻返回，
        不等算完。不等待就掐表会测个寂寞（时间≈0）→ 对策：
        torch.cuda.synchronize() 强制停下来等 GPU 全部干完，再读表；
      - 坑 3：只看总时间不知道「时间花在哪」 → 对策：两套口径都报
        （端到端 perf_counter 秒表 + profiler 逐算子分解）。

运行：
    Mac（逻辑空跑）：     python 04_compile_vs_eager.py --device cpu --iters 20
    S1 租卡：           python 04_compile_vs_eager.py --device cuda --iters 200
    S1 抓时间线：        nsys profile -o s1_compiled python 04_compile_vs_eager.py --device cuda --mode compiled
    （nsys profile = 用 NVIDIA 的分析器录下全程时间线，生成可回看的报告文件）

名词白话：
    - harness：测试脚手架——把「被测对象之外的固定流程」搭好，保证每次测的条件一致；
    - 冷启动/稳态：第一次调用的耗时（含编译）/ 编译完后每次调用的真实耗时；
    - GEMM：矩阵乘法（General Matrix Multiply），深度学习里最重的计算，
      之后 W7 整周都在写它。

验收：
    cpu 空跑：两端数值一致、无异常退出；GPU 数字回填观察记录。
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn


class BlockStack(nn.Module):
    """7B 量级模型的「形状骨架」：不做真权重，形状是对的。

    为什么不加载真模型？——本文件测的是「编译带来的结构变化」，不是模型精度，
    用随机权重搭一个同样形状的骨架就够，Mac 也跑得动。

    decoder layer（LLM 的基本重复单元，一堆同样的层叠起来）的算子骨架：
      RMSNorm → QKV proj → attn(softmax) → o_proj → RMSNorm → MLP(门控)
    每层五段，每段都是可融合元素运算 + 两个 GEMM。
    算子白话：RMSNorm = 把数值拉回正常范围的归一化（W5 亲手写）；
             QKV proj / o_proj / MLP = 都是矩阵乘法（GEMM）；
             门控 MLP = 两条分支相乘再投回原维度（Llama/SwiGLU 风格）；
             attn 里刻意保留了 softmax——它是 W8 FlashAttention 的主角。
    """

    def __init__(self, d: int = 1024, ff: int = 4096, layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.RMSNorm(d),
                        "qkv": nn.Linear(d, 3 * d, bias=False),
                        "o": nn.Linear(d, d, bias=False),
                        "norm2": nn.RMSNorm(d),
                        "gate": nn.Linear(d, ff, bias=False),
                        "up": nn.Linear(d, ff, bias=False),
                        "down": nn.Linear(ff, d, bias=False),
                    }
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 一个 decoder layer 内部的数据流，跟着注释走一遍：
        for lyr in self.layers:
            h = lyr["norm1"](x)                 # 先归一化
            qkv = lyr["qkv"](h)                 # 一次 GEMM 同时算出 Q/K/V 三个矩阵
            q, k, v = qkv.chunk(3, dim=-1)      # 沿最后一维切三段 → 分别是 Q、K、V
            # 注意力简化为单头 softmax(QK^T)V——够用来观察融合与 launch 数
            # 白话：Q 和 K 相似度打分 → softmax 变成「和为 1 的权重」→ 加权平均 V
            scores = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)  # 打分；除以 √d 防止数值过大
            attn = torch.softmax(scores, dim=-1)                         # 归一化成权重
            x = x + lyr["o"](attn @ v)         # 加权平均后投影回去；x + ... = 残差连接（保留原信息）
            h = lyr["norm2"](x)                # 第二次归一化
            # 门控 MLP：gate 分支过一个「平滑开关」函数 silu 后与 up 分支相乘，再投回原维度
            x = x + lyr["down"](torch.nn.functional.silu(lyr["gate"](h)) * lyr["up"](h))
        return x


def bench(model: nn.Module, x: torch.Tensor, iters: int, device: str) -> tuple[float, float]:
    """返回 (首次耗时含编译, 稳态均次耗时 ms)。

    结构口诀：冷启动一次 → 预热几次 → 正式计时 iters 次。
    每一段的起点/终点都夹 synchronize（GPU 时），保证掐的是「真干完」的时间。
    """
    with torch.no_grad():
        # 冷启动：第一次调用。对编译版来说，这一步内部会触发编译，所以特别慢——
        # 但这个慢只发生一次，不能让它污染后面的稳态测量。
        if device == "cuda":
            torch.cuda.synchronize()   # 先等之前的所有 GPU 任务干完，表才从零开始
        t0 = time.perf_counter()
        model(x)
        if device == "cuda":
            torch.cuda.synchronize()   # 等 GPU 干完再停表（异步！不 sync 测到的是假的）
        cold = (time.perf_counter() - t0) * 1e3   # ×1e3：秒 → 毫秒
        # 预热 + 稳态
        for _ in range(5):             # 预热：把缓存、内存分配等一次性开销消化掉
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()       # 正式计时开始
        for _ in range(iters):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()   # 计时结束前必须等 GPU 全部干完
        warm = (time.perf_counter() - t0) / iters * 1e3  # 总耗时÷次数 = 均次，秒→毫秒
    return cold, warm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--mode", default="both", choices=["eager", "compiled", "both"])
    args = ap.parse_args()

    torch.manual_seed(0)
    device = args.device
    model = BlockStack().eval().to(device)   # eval() = 切到推理模式；.to(device) = 搬到 cpu 或 gpu
    x = torch.randn(64, 512, 1024, device=device)   # 64 seq × 512 token × d=1024
    # 白话：64 条序列一批进来，每条 512 个 token，每个 token 用 1024 维向量表示。
    # 这个形状模拟的是「prefill 阶段」——一大段 prompt 一次性并行算完。

    with torch.no_grad():
        ref = model(x)

    results: dict[str, tuple[float, float]] = {}
    if args.mode in ("eager", "both"):
        results["eager"] = bench(model, x, args.iters, device)
    if args.mode in ("compiled", "both"):
        results["compiled"] = bench(torch.compile(model), x, args.iters, device)
        with torch.no_grad():
            out = torch.compile(model)(x)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)  # 编译版数值校验
        # 白话：无论快多少，「算出来的数」必须和不编译时一致（误差容忍范围内）。
        # 快而错 = 白快。这个「先对数再谈性能」的流程，之后每一周都会重复。

    print(f"\ndevice={device}  shape={tuple(x.shape)}")
    for name, (cold, warm) in results.items():
        print(f"  {name:9s}  冷启动(含编译)={cold:9.1f}ms   稳态均次={warm:7.3f}ms")
    if len(results) == 2:
        e, c = results["eager"][1], results["compiled"][1]
        print(f"  提速比 = {e / c:.2f}x")   # 不编译耗时 ÷ 编译后耗时 = 快了几倍
    # TODO(you)（S1 后回答）：
    #   1) 冷启动里编译占了多少？把它除以稳态均次——多少次调用才「回本」？
    #   2) nsys 时间线上 eager vs compiled 的 kernel 数差多少倍？
    #   3) H20 上 get_device_properties 打出的 SM 数/带宽是多少（记给 W6）？


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录 ═══════════════════════════════
# - CPU 空跑结论（逻辑是否通过）：
# - GPU：eager 稳态 / compiled 稳态 / 提速比 / 编译回本次数：
# - kernel 数（eager → compiled）：
# - H20 属性（SM 数 / 带宽）：
