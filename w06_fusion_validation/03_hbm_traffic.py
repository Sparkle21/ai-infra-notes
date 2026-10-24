#!/usr/bin/env python3
"""
W6-03 · HBM 流量账本：解析式论证「融合省的是 HBM 往返」
=============================================================
先补几句背景（这个文件在干什么）：
    W6 要回答的问题：融合到底快在哪？能快多少？
    方法不是瞎测，而是先拿纸笔算一笔账（解析式 = 纯公式推算，不跑代码）：
    数据从显存（HBM）搬到计算单元的次数 × 总字节数 ÷ 带宽 = 理论耗时。
    算完理论值，拿去和实测对账——对得上，理解就牢了；对不上，
    差异本身就是新知识（见尾部 TODO）。

    这是性能工程的通用方法：先建模、后实测、再对账。

这是本周论证笔记的计算引擎：不跑任何 kernel，纯算账。

名词白话：
    - HBM：显存大仓库（High Bandwidth Memory）；
    - 「一趟」：整个张量在 HBM 与计算单元之间搬一次（读或写各算一趟）；
    - 理论耗时下限：哪怕计算零耗时，光搬运也得这么久——是性能的天花板。

账本口径（先立规矩，之后所有数字都用这套口径）：
    - RMSNorm 输入 [R, C]，FP32（4 字节），输出同形状。
    - 「一趟」= 读或写整个 [R, C] 张量一次（2*R*C*4 字节算 2 趟的量）。

    decomposed（4 个算子）：
        pow(2)      : 读 x，写 x²            → 2 趟
        mean(-1)    : 读 x²，写 [R,1]（忽略） → 1 趟
        rsqrt+mul   : 读 x、读 rms，写 t      → 2 趟
        mul w       : 读 t、读 w（小，忽略），写 y → 2 趟
        合计 ≈ 7 趟（读 4 写 3）
        白话：每步的中间结果都要落回仓库、下一步再搬出来——
        搬运次数 = 算子数的两倍左右，这就是「不融合的代价」。
    fused：
        读 x 1 趟 + 写 y 1 趟 = 2 趟
        白话：数据进仓一次、出仓一次，中间全程待在计算单元里。
    （框架的 eager 实现可能内部已融合 → 账本按 2 趟算，但要看 S3 实测印证。）

运行：
    python 03_hbm_traffic.py                       # 默认 H20 4TB/s，S3 形状
    python 03_hbm_traffic.py --bw 2000             # 换带宽（如 A100 2TB/s）
    python 03_hbm_traffic.py --measured 0.120 0.035  # S3 后：把实测 ms 传进来对账
"""
from __future__ import annotations

import argparse

# 与 01_bench_core 同一份形状清单（三文件共用同一实验设计）
SHAPES = [(16, 4096), (8192, 4096), (8192, 11008)]
DTYPES = {"fp32": 4, "fp16": 2, "bf16": 2}   # 每元素字节数：fp32 用 4 字节，两种 16 位浮点用 2 字节


def trips_bytes(n_rows: int, n_cols: int, dtype: str, variant: str) -> tuple[int, float]:
    """返回 (HBM 趟数, 搬运字节数)。口径见文件头。"""
    element = DTYPES[dtype]
    n = n_rows * n_cols * element          # 整个张量的字节数（一趟的量）
    if variant == "fused":
        trips = 2                          # 读 x + 写 y
    elif variant == "decomposed":
        trips = 7                          # 见文件头的逐算子账
    else:
        raise ValueError(variant)
    return trips, trips * n                # 趟数 × 每趟字节数 = 总搬运量


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw", type=float, default=4000.0, help="HBM 带宽 GB/s（H20≈4000）")
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--measured", nargs=2, type=float, metavar=("DECOMP_MS", "FUSED_MS"),
                    help="S3 实测 ms，用于对账（只对最后一个形状 8192x11008）")
    args = ap.parse_args()

    print(f"带宽假设 = {args.bw:.0f} GB/s   dtype = {args.dtype}\n")
    print(f"{'shape':>14} | {'decomp 趟':>9} {'fused 趟':>8} | {'decomp 理论ms':>12} {'fused 理论ms':>11} | {'理论加速比':>9}")
    print("-" * 78)
    for n_rows, n_cols in SHAPES:
        t_d, b_d = trips_bytes(n_rows, n_cols, args.dtype, "decomposed")
        t_f, b_f = trips_bytes(n_rows, n_cols, args.dtype, "fused")
        # 理论耗时 = 字节数 ÷ 带宽：GB÷(GB/s)=秒；×1e3 换成 ms。
        # 这是「光搬运就要这么久」的下限（计算本身另算，但对带宽瓶颈型
        # 算子如 RMSNorm，搬运几乎就是全部）。
        ms_d = b_d / args.bw / 1e9 * 1e3
        ms_f = b_f / args.bw / 1e9 * 1e3
        # 加速比 = 趟数比：7 趟 ÷ 2 趟 ≈ 3.5x——理论上融合的收益上限
        print(f"{n_rows:>6}x{n_cols:<7} | {t_d:>9} {t_f:>8} | {ms_d:>12.3f} {ms_f:>11.3f} | {ms_d / ms_f:>8.1f}x")

    if args.measured:
        # 对账模式：拿实测耗时反推「实际带宽利用率」，验证账本假设对不对
        ms_d, ms_f = args.measured
        n_rows, n_cols = SHAPES[-1]
        t_f, b_f = trips_bytes(n_rows, n_cols, args.dtype, "fused")
        t_d, b_d = trips_bytes(n_rows, n_cols, args.dtype, "decomposed")
        # 有效带宽 = 按账本口径应搬的字节 ÷ 实测耗时——实测慢，说明带宽没吃满
        bw_eff_d = b_d / (ms_d / 1e3) / 1e9
        bw_eff_f = b_f / (ms_f / 1e3) / 1e9
        print("\n== 对账（shape=[8192, 11008]）==")
        print(f"  实测加速比           = {ms_d / ms_f:.1f}x")
        print(f"  decomposed 有效带宽  = {bw_eff_d:.0f} GB/s（≈理论 {args.bw:.0f} 的 {bw_eff_d / args.bw * 100:.0f}%）")
        print(f"  fused     有效带宽   = {bw_eff_f:.0f} GB/s（≈理论 {args.bw:.0f} 的 {bw_eff_f / args.bw * 100:.0f}%）")
        # TODO(you)（notes_W6.md 的核心段落就看这三行数字怎么圆）：
        #  - 若 fused 有效带宽 < 80% 理论：差距是 launch 开销？occupancy？行数不够铺满 SM？
        #  - 若实测加速比 < 理论 3.5x：eager 世界里框架的 kernel 实际上没按 7 趟搬？
        #    （去看 torch 源码的 fused path——很可能 pow+mean 是一个 kernel。）
        #  - 结论必须落到「账本哪一行的假设错了」才算完成论证。


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录 ═══════════════════════════════
# - 理论表（默认参数）的加速比：
# - 对账后三个关键数字：
# - 我修正的账本假设（哪一行错了，为什么）：
