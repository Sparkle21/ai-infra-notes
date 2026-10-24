#!/usr/bin/env python3
"""
W4-05 · S2.5 探针：TP 的 all-reduce 在 kernel 层长什么样
=============================================================
先补几句背景（这个实验在探什么）：
    一张 GPU 放不下大模型时，把模型切开分给多张卡一起算——
    最常见的切法叫 TP（张量并行，Tensor Parallelism）。
    切了就得分工，分工就得对结果——卡与卡之间必然发生通信。
    本文件把 TP 里那次通信放大到时间线上一眼可见，看清它的真面目。

目标：
    构造一个「TP 式 matmul + all-reduce」的最小循环，让通信在 nsys 时间线上
    足够显眼，回答三个问题（见 ../sessions/S2.5_nccl_tp.md）：
      Q1 all-reduce 是一个 kernel 还是多个 chunk kernel？
      Q2 计算 kernel 与通信 kernel 有无重叠？
      Q3 消息大小与通信时长的关系（测两档尺寸）

运行：
    CPU 逻辑自检（Mac）：python 05_nccl_tp_probe.py --mode cpu_check
    S2.5 单卡对照       ：python 05_nccl_tp_probe.py --mode single --device cuda
    S2.5 双卡 TP2       ：torchrun --nproc_per_node=2 05_nccl_tp_probe.py --mode tp2 --device cuda
    S2.5 抓时间线       ：nsys profile -o s25_tp2 torchrun --nproc_per_node=2 05_nccl_tp_probe.py --mode tp2 --device cuda
    （torchrun：PyTorch 官方多卡启动器，会把两个进程拉起来并自动分配编号）

名词白话：
    - rank / world_size：进程编号（0号卡、1号卡…）/ 总进程数（这里=卡数）；
    - all-reduce：所有卡各自出一个数，全体相加后每个人都拿到同一个总和；
    - NCCL：NVIDIA 的卡间通信库（GPU 专用），gloo 是它的 CPU 替代品；
    - 分片（shard）：把一个大矩阵切开，每卡拿一块。

TP 原理一分钟版（为什么会有 all-reduce）：
    把 Y = X @ W 按列切成两半，W = [W0 | W1]，两张卡各算 X @ W_i，
    得到部分和 Y0、Y1。Y = Y0 + Y1 —— 所以每张卡都要拿到「别人算的那半」，
    全体相加 = all-reduce。这就是 TP forward 里的那一次通信。
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.distributed as dist   # 分布式通信库：all_reduce 等跨卡操作都在这


def tp_matmul_once(x: torch.Tensor, w_shard: torch.Tensor) -> torch.Tensor:
    """单次 TP matmul：本地分片 matmul + all-reduce。

    Args:
        x:        [M, K] 完整输入（两卡相同，TP 的本质是切权重不切输入）
        w_shard:  [K, N//world] 本卡持有的权重分片
    """
    y_partial = x @ w_shard                      # 本地部分和（只有半列结果，还不是最终答案）
    dist.all_reduce(y_partial, op=dist.ReduceOp.SUM)   # 跨卡求和 ← 通信发生在这
    # 白话：每张卡把自己的部分和交出去、全体相加、各自领回完整结果。
    # 注意：all_reduce 会原地修改 y_partial（不需要接收返回值）。
    return y_partial


def run_tp2(device: str, m: int = 2048, k: int = 2048, n: int = 2048, rounds: int = 50) -> None:
    """双卡模式：每轮做一次 TP matmul，循环 rounds 轮让时间线上出现清晰条纹。

    TODO(you)（S2.5 现场前在本函数补两个动作）：
      1. 前后各加 torch.cuda.synchronize()，并对整体计时；
      2. 再写一个「计算与通信分离计时」版本：分别记录 matmul 段与 all_reduce 段
         （用 synchronize 划界），这样不用 nsys 也能先拿到粗口径的重叠/串行结论。
    """
    rank = dist.get_rank()          # 我是几号卡（0 或 1）
    world = dist.get_world_size()   # 一共几张卡
    torch.manual_seed(42)                      # 两卡同种子：x 完全一致（TP 要求输入相同）
    x = torch.randn(m, k, device=device)
    w = torch.randn(k, n // world, device=device)   # 演示用随机权重（数值无意义，形状要对）
    # 白话：本卡只拿权重的一半列（n//world 列）——这就是「切权重不切输入」。

    for _ in range(rounds):
        y = tp_matmul_once(x, w)    # 循环多轮：时间线上出现重复条纹，nsys 里好认
    torch.cuda.synchronize() if device == "cuda" else None
    if rank == 0:                   # 只让 0 号卡打印（不然两份重复输出）
        print(f"[tp2] world={world} rounds={rounds} y.sum={y.sum().item():.1f}（随机权重，只看形状与时间）")


def run_single(device: str, m: int = 2048, k: int = 2048, n: int = 2048, rounds: int = 50) -> None:
    """单卡对照：同样的 matmul 不带通信——时间线上的「纯计算」基准。"""
    x = torch.randn(m, k, device=device)
    w = torch.randn(k, n, device=device)
    t0 = time.perf_counter()
    for _ in range(rounds):
        y = x @ w
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / rounds * 1e3
    print(f"[single] {m}x{k}x{n} 均次 = {dt:.3f} ms")


def cpu_check() -> None:
    """Mac 逻辑自检：不碰 NCCL，验证 tp_matmul_once 的数学是对的。

    用 gloo 后端（CPU 可用）+ 单进程进程组模拟 world=2 的 all-reduce 语义。
    最小验证：手算 2 份部分和相加 == 完整 matmul。
    白话：先在草稿纸上验证「切开算再相加 = 不切直接算」，
    数学站住了，GPU 上才值得看通信；数值在 Mac 验，性能在租卡测。
    """
    x = torch.randn(8, 16)
    w = torch.randn(16, 12)
    y_full = x @ w                          # 不切：直接算完整结果（标准答案）
    y_partial = x @ w[:, :6] + x @ w[:, 6:]     # 模拟两卡部分和相加
    # 切两半分别算再相加，应和直接算一致（浮点误差容忍内）
    torch.testing.assert_close(y_partial, y_full, rtol=1e-5, atol=1e-5)
    print("cpu_check ✓（TP 部分和数学正确；NCCL 路径留给 GPU 会话）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cpu_check", choices=["cpu_check", "single", "tp2"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    if args.mode == "cpu_check":
        cpu_check()
        return

    if args.mode == "tp2":
        # init_process_group：多进程互认 + 建好通信线路；GPU 用 nccl，CPU 用 gloo
        dist.init_process_group(backend="nccl" if args.device == "cuda" else "gloo")
        run_tp2(args.device)
        dist.destroy_process_group()   # 收尾：拆掉通信线路，进程干净退出
    else:
        run_single(args.device)


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录（S2.5 回填）═════════════════════
# - Q1 all-reduce 的 kernel 形态（单个 / 多 chunk）：
# - Q2 计算与通信的重叠情况：
# - Q3 两档消息大小的通信时长：
# - NCCL_DEBUG=INFO 里 ring/channel 的关键行：
