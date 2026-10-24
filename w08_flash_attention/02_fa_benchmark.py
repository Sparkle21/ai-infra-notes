#!/usr/bin/env python3
"""
W8-02 · FA forward benchmark：mine vs SDPA（S4 会话主脚本之一）
=============================================================
先补几句背景（这个文件在干什么）：
    W8-01 写完、数值对了之后，自然的下一个问题是：我的手写版距离
    工业级实现有多远？本文件就是这场比赛的裁判：同样的输入，
    各自计时、看比值。输不可怕——可怕的是不知道输在哪
    （文件尾 TODO 就是逼你把「输在哪」说清楚）。

名词白话：
    - SDPA：PyTorch 官方的 scaled_dot_product_attention——你调它，
      它在后台自动挑一个现成的快 kernel（很可能就是 FlashAttention）；
    - dispatch：按输入情况自动选后端实现的路由机制；
    - forward-only：只算前向（推理够用）；训练还要 backward，难一档；
    - O(N²) vs O(N)：标准注意力的中间分数矩阵随 N 平方膨胀（显存爆），
      FA 边扫边归并、中间只留 O(N) 的小账本——「FA 到底赢在哪」的骨架。

对比对象：
    mine  ：w08/01 你写完的 fa_forward（Triton，forward-only）
    SDPA  ：torch.nn.functional.scaled_dot_product_attention
            （PyTorch 官方 dispatch，可能走 FlashAttention/内存高效后端）
维度：序列长度扫描（512 / 2048 / 4096）——FA 的优势随 N 增大而扩大，
      因为标准注意力 O(N²) 显存 vs FA 的 O(N)。

运行：
    容器空跑（只验证 wiring，tiny 形状）：
        TRITON_INTERPRET=1 python 02_fa_benchmark.py --smoke
    S4：
        python 02_fa_benchmark.py --device cuda --iters 50
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def load_mine():
    """动态加载 01 的实现。

    为什么不用 import？文件名以数字开头（01_fa2_forward）不是合法的
    Python 标识符，`import 01_fa2_forward` 语法直接报错——所以走
    importlib 三步组合拳（W6-01 讲过）：spec 建模块壳 → module_from_spec
    造模块对象 → exec_module 把文件内容执行进去。
    """
    path = Path(__file__).resolve().parent / "01_fa2_forward.py"
    spec = importlib.util.spec_from_file_location("w08_fa01", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fa_forward


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
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--smoke", action="store_true", help="tiny 形状 wiring 自检")
    args = ap.parse_args()

    mine = load_mine()
    B, H, d = 1, 8, 64
    causal = True      # 与 LLM 推理一致的口径（生成只能看过去）
    seq_lens = [64] if args.smoke else [512, 2048, 4096]   # smoke=只验接线的迷你档

    print(f"device={args.device}  causal={causal}  B={B} H={H} d={d}\n")
    print(f"{'N':>6} | {'mine ms':>9} | {'SDPA ms':>9} | {'mine/SDPA':>9}")
    print("-" * 45)
    for N in seq_lens:
        q = torch.randn(B, H, N, d, device=args.device)
        k = torch.randn(B, H, N, d, device=args.device)
        v = torch.randn(B, H, N, d, device=args.device)

        # 数值先行（铁律：性能对比前数值必须已对）
        torch.testing.assert_close(mine(q, k, v, causal), F.scaled_dot_product_attention(q, k, v, is_causal=causal), rtol=1e-3, atol=1e-3)

        t_mine = bench(lambda: mine(q, k, v, causal), args.iters, args.device)
        t_sdpa = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal), args.iters, args.device)
        print(f"{N:>6} | {t_mine:9.3f} | {t_sdpa:9.3f} | {t_mine / t_sdpa:8.2f}x")

    # TODO(you)（S4 后回答，写 notes_W8.md）：
    #  1) mine/SDPA 的比值随 N 怎么变？（预期：N 大时差距缩小——你的 tiling
    #     摊薄效应变大；但仍输——输在哪些没做的 trick？对照看不懂清单。）
    #  2) nsys 时间线：mine 的 kernel 数与 SDPA 相同吗（1 个 vs 可能的 split-k）？
    #    （nsys = Nsight Systems，NVIDIA 官方「行车记录仪」——把 GPU 上每个
    #     kernel 何时启动、跑多久录成时间线，S4 会用到。split-k = 把 K 维
    #     拆给多个 program 并行算再归并的技巧，官方实现可能用。）
    #  3) 序列 4096 时显存：标准注意力如果手写（O(N²) 中间矩阵）需要多少 GB？
    #     算一笔账写进笔记（B*H*N*N*4B）——FA 省显存的量化表达。


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录（S4 回填）═══════════════════════
# - 三档 N 的 mine/SDPA 比值：
# - nsys：kernel 形态差异：
# - O(N²) 显存账（N=4096）：
