#!/usr/bin/env python3
"""
W6-01 · RMSNorm 三口径 benchmark（S3 会话主脚本之一）
=============================================================
先补几句背景（benchmark 是什么、为什么分三口径）：
    benchmark = 性能测试。要证明「融合值得」，不能只凭感觉，得把三种
    实现放在同一条件下计时对比——同一份数据、同一预热、同一计时法。

    三口径就是同一个 RMSNorm 的三种写法（口径 = 统一的比较标准）：
    一个黑盒官方版、一个故意不融合版、一个你的融合版——
    融合的收益 = decomposed 与 fused 的耗时差。

三口径：
    eager       ：torch.nn.RMSNorm（框架实现，黑盒对照）
    decomposed  ：x.pow(2).mean + rsqrt + mul*w（逐算子，模拟「没融合」的世界）
    fused       ：w05/03 的 Triton kernel（你的实现，动态加载）

名词白话：
    - 有效带宽（GB/s）：每秒实际搬了多少 GB 数据。算式 = 搬运字节数 ÷ 耗时。
      和「理论带宽」（硬件极限，H20 约 4000 GB/s）的比值越接近 1，
      说明 kernel 把硬件利用得越充分；
    - warmup（预热）：正式计时前先白跑几次，把首次调用的一次性开销
      （编译、缓存、内存分配）消化掉；
    - interpret 模式的数字无意义：纯 Python 模拟太慢，只验逻辑、不比快慢。

运行：
    Mac（无 triton 也能跑两口径，验证 harness 逻辑）：
        python 01_bench_core.py --device cpu --iters 5
    容器（interpret，只验 fused 接线对不对，**数字无意义**）：
        TRITON_INTERPRET=1 python 01_bench_core.py --device cpu --iters 3
    S3（真数字）：
        python 01_bench_core.py --device cuda --iters 200 --kernel rmsnorm
"""
from __future__ import annotations

import argparse
import importlib.util   # 动态加载文件用（文件名以数字开头不能普通 import）
import time
from pathlib import Path

import torch

# 三个形状故意选的：小（启动开销主导）/ 中（典型 decode 批）/ 大（FFN 宽行，S3 重点）
SHAPES = [(16, 4096), (8192, 4096), (8192, 11008)]   # S3 的形状清单（plan 对齐）


def load_fused_rmsnorm():
    """动态加载 w05/03 的实现（文件名以数字开头，用 spec 加载）。

    白话：Python 的 import 语法不吃数字开头的文件名（03_fused_rmsnorm.py
    不能写 import 03_fused_rmsnorm），所以用这套三步组合拳手动加载：
    spec（指定路径建一个模块壳）→ module_from_spec（造出模块对象）→
    exec_module（把文件内容执行进去）。效果等同普通 import。

    返回 (callable(x, w, eps)->Tensor) 或 None（环境没 triton 时优雅降级）。
    """
    # Path(__file__).resolve().parent.parent = 本文件的上两级 = ai_infra/ 根
    path = Path(__file__).resolve().parent.parent / "w05_triton_rmsnorm" / "03_fused_rmsnorm.py"
    if not path.exists():
        print(f"[fused] 未找到 {path}")
        return None
    try:
        spec = importlib.util.spec_from_file_location("w05_rmsnorm", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.triton_rmsnorm
    except Exception as exc:  # noqa: BLE001
        # 优雅降级：Mac 上没装 triton 是正常情况——报个信、返回 None，
        # 让 benchmark 继续跑其他两个口径，而不是整个崩掉
        print(f"[fused] 加载失败（Mac 无 triton 属预期）：{type(exc).__name__}: {exc}")
        return None


def bench(fn, x, w, iters: int, device: str) -> float:
    """稳态均次耗时（ms）。warmup 分离 + cuda synchronize 划界（诚实计时）。

    与 W1-04 同一套模板：预热 5 次 → 掐表跑 iters 次 → 总耗时÷次数。
    GPU 上每次掐表前后都 sync（异步执行不等待会测个寂寞）。
    """
    for _ in range(5):
        fn(x, w)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x, w)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def eff_bandwidth_gbs(ms: float, n_rows: int, n_cols: int, passes: int, dtype_bytes: int = 4) -> float:
    """有效带宽 = 搬运字节数 / 时间。passes = 「读+写算几趟」（见 03 的账本定义）。

    白话算例：8192x4096 的 float32（每数 4 字节）读写各一趟（passes=2）：
    字节 = 8192×4096×4×2 ≈ 268MB；若耗时 0.1ms → 带宽 ≈ 2680 GB/s。
    这个数越接近硬件理论值，说明 kernel 越接近「带宽榨干」的理想状态。
    """
    bytes_moved = n_rows * n_cols * dtype_bytes * passes
    return bytes_moved / (ms / 1e3) / 1e9   # ÷耗时（ms→s）÷1e9 → GB/s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--kernel", default="rmsnorm")
    args = ap.parse_args()
    assert args.kernel == "rmsnorm", "本周只有 rmsnorm，一个主角讲透胜过三个跑马"

    device = args.device
    torch.manual_seed(0)

    def decomposed(x, w):
        # 与 w05/03 的 torch_rmsnorm 同款：故意不融合，逐步落回内存（对照组）
        ss = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ss + 1e-6) * w

    fused = load_fused_rmsnorm()

    print(f"device={device}  iters={args.iters}  fused={'✓' if fused else '✗(降级)'}\n")
    print(f"{'shape':>16} | {'eager ms':>9} | {'decomp ms':>9} | {'fused ms':>9} | {'fused GB/s':>10}")
    print("-" * 70)
    for n_rows, n_cols in SHAPES:
        x = torch.randn(n_rows, n_cols, device=device)
        w = torch.randn(n_cols, device=device)

        eager = torch.nn.RMSNorm(n_cols, eps=1e-6).to(device)
        t_e = bench(lambda a, b: eager(a), x, w, args.iters, device)
        t_d = bench(decomposed, x, w, args.iters, device)
        t_f = bench(fused, x, w, args.iters, device) if fused else float("nan")
        # fused 的「读 1 趟 + 写 1 趟」= passes 2（口径详见 03_hbm_traffic.py）
        # 白话：融合版只把数据从显存读进来一次、把结果写回去一次，
        # 中间全程待在 GPU 工作台（SM）里——所以只计这 2 趟。
        bw = eff_bandwidth_gbs(t_f, n_rows, n_cols, passes=2) if fused else float("nan")
        print(f"{f'{n_rows}x{n_cols}':>16} | {t_e:9.3f} | {t_d:9.3f} | {t_f:9.3f} | {bw:10.1f}")

    # TODO(you)（S3 后回答，写进 notes_W6.md）：
    #  1) fused 的 GB/s 到了 H20 理论 4TB/s 的百分之多少？（不足 60% 的话差距去哪了）
    #  2) eager（黑盒）意外地快/慢吗？去 torch 源码看 nn.RMSNorm 在 CUDA 上走什么路径。
    #  3) 三个形状里哪个形状 fused 优势最大？和 W5 笔记里你的预测一致吗？


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录 ═══════════════════════════════
# - Mac 空跑：harness 逻辑是否通过 / fused 是否正常加载：
# - S3 三形状 × 三口径的 ms 表（回填 README 实验表 1–3）：
# - fused GB/s 与理论带宽之比：
# - 与预测的偏差及初步解释：
