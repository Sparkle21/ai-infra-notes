#!/usr/bin/env python3
"""
W3-03 · continuous batching 对比实验：static vs continuous vs +preemption
=============================================================================
先补几句背景（这个实验在比什么）：
    batching = 几个请求挤同一班车（一次 forward 同时算一批）——
    GPU 吞吐量的根本来源。但「怎么凑班、何时发车、到站谁下车」有不同做法，
    性能差别巨大。本实验用同一批「剧本」把三种做法的成绩单拿出来对比。

    直观比喻（公交车模型）：
      A. static batching：坐满才发车，全车人都到终点才折返接下一批——
         车上有人早到了也得干等（「批尾空转」）；
      B. continuous batching：到站就有人下、站台有人就上，车永不停运——
         这就是 vLLM 让业界吞吐翻倍的核心思想；
      C. B + 抢占：车上太挤（显存不够）时把乘客赶下去等下一班。

目标：
    复用 02 的内核，对比三种 batching 策略在同一剧本下的表现：
      A. static batching ：batch 凑齐 MAX_BATCH 才开工，全做完才收下一批
                           （朴素 serving 的形态，也是 vLLM 之前的旧世界）
      B. continuous      ：请求完成即释放 slot，waiting 立即补位
                           （vLLM 的核心思想）
      C. continuous+preempt ：B 基础上打开抢占（02 的 pick_decode_victims）
    输出一张对比表：平均完成步数 / 尾延迟（P90 步数）/ 吞吐（完成数/总步数）。

名词白话：
    - P90 / 尾延迟：把所有请求的耗时从快到慢排队，第 90% 位置那个的耗时。
      平均值好不一定服务好——用户记住的是「最慢的那次」；
    - makespan：从开工到全部干完的总时长；
    - slot：batch 里的一个座位（一个名额）。

运行：
    python 03_continuous_batching_sim.py            # 三策略对比（同一剧本×5 seed）
    python 03_continuous_batching_sim.py --trace    # 额外打印 B 策略时间线

注意：
    本文件 import 02 的类。若你在 02 里改了类名/签名，这里同步改。
"""
from __future__ import annotations

import argparse
import random
import statistics

from importlib import import_module

# 数字开头的文件名不能用普通 import 语法（Python 语法不允许），
# 所以用 import_module 按字符串名字加载——效果等同 import 02_mini_scheduler
m02 = import_module("02_mini_scheduler")   # 按文件名 import（同目录运行）
Request = m02.Request   # 把 02 里的类取过来用（类型别名，只是方便少打 m02.）
Ledger = m02.Ledger


# ══════════════════════ 你要写的三个策略函数 ═════════════════════════════
def run_static(reqs: list[Request], max_batch: int = 4) -> dict[int, int]:
    """A. static batching：凑批 → 整批 decode 到全部完成 → 下一批。

    提示：
      - 按 arrive_step 顺序取 max_batch 个为一批；
      - 批内每个 step 所有活着的请求各 +1 token；
      - 一批**全部完成**才能开下一批（这就是 static 的浪费所在）。
    返回 {rid: 完成步数}。
    """
    # TODO(you)：实现。不使用 Ledger（static 场景假设显存够整批）。
    # 写法提示（约 20 行）：
    #   finished = {}
    #   i = 0
    #   while i < len(reqs):
    #       batch = reqs[i:i+max_batch]              # 按顺序切一批
    #       t = batch[0].arrive_step                  # 从最早到达者的时间开始
    #       alive = {r.rid: r for r in batch}
    #       while alive:                              # 批内循环：直到最后一个完成
    #           t += 1
    #           for r in list(alive.values()):
    #               r.generated += 1
    #               if r.done:
    #                   finished[r.rid] = t
    #                   del alive[r.rid]              # 下车——但车还得等其他座位
    #       i += max_batch
    #   return finished
    raise NotImplementedError


def run_continuous(reqs: list[Request], max_batch: int = 4) -> dict[int, int]:
    """B. continuous batching：完成即补位（无 preemption，账本压力靠 MAX_BATCH 控制）。

    提示：
      - 复用 02 的 simulate()：把 ledger 调大（无资源竞争），
        admit 条件只剩「batch 没满」——直接构造一个大 ledger + 传入策略；
      - 或自己写 30 行主循环也行（写一遍更扎实）。
    返回 {rid: 完成步数}。
    """
    # TODO(you)：实现。
    # 写法提示：主循环与 02 的 simulate 骨架一样，只改两处——
    #   1) Ledger 换成超大额度（Ledger(total=10**9, watermark=0)），显存永远够，
    #      于是抢占永不触发；
    #   2) admit 判断只剩 len(running) < max_batch（座位满就等）。
    #   或者直接 30 行手写主循环——写一遍对「补位」的体会更扎实。
    raise NotImplementedError


def run_continuous_preempt(reqs: list[Request], max_batch: int = 4) -> dict[int, int]:
    """C. continuous + preemption：账本故意收紧（TOTAL_BLOCKS 调小），
    让 02 的抢占逻辑真实发生，观察代价（recompute 造成的额外步数）。

    提示：复用 simulate()，把 m02.TOTAL_BLOCKS 临时改小（如 24）。
    """
    # TODO(you)：实现。
    # 写法提示：m02.TOTAL_BLOCKS 是模块级常量，模拟时临时改小：
    #   old = m02.TOTAL_BLOCKS
    #   m02.TOTAL_BLOCKS = 24        # 收紧账本，逼出抢占
    #   rep = m02.simulate(reqs, verbose=False)
    #   m02.TOTAL_BLOCKS = old       # 用完改回去（不然会污染后续实验）
    #   return rep.finished_at
    #   注意：需先在 02 里写好 admit / pick_decode_victims，这里才能调。
    raise NotImplementedError


# ══════════════════════ 统计 harness（勿改）══════════════════════════════
def summarize(name: str, results: list[dict[int, int]]) -> None:
    """把多次运行的完成步数汇总成三个指标：平均 / P90 / makespan。"""
    steps_pool: list[int] = []    # 所有请求的完成步数汇进一个池子
    makespans: list[int] = []     # 每次运行的总时长
    for fin in results:
        steps_pool.extend(fin.values())
        makespans.append(max(fin.values()) + 1)   # +1：把步数换成「步数差」口径
    steps_pool.sort()
    # P90：排完序后取第 90% 位置的值（尾延迟——最慢的 10% 用户体验的下限）
    p90 = steps_pool[int(len(steps_pool) * 0.9) - 1]
    print(
        f"{name:24s} 平均完成步数={statistics.mean(steps_pool):6.1f}  "
        f"P90={p90:4d}  平均 makespan={statistics.mean(makespans):6.1f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    # 三种策略用同一批剧本（同 seed 生成同样请求）——控制变量，
    # 成绩差异才能归因于「策略」而不是「运气」。
    for label, runner in [
        ("A static", run_static),
        ("B continuous", run_continuous),
        ("C continuous+preempt", run_continuous_preempt),
    ]:
        results = []
        for seed in range(args.seeds):
            rng = random.Random(seed)
            reqs = m02.make_script(rng, args.n)
            results.append(runner(reqs))
            if args.trace and label == "B continuous" and seed == 0:
                print(f"— trace({label}, seed=0)：{len(results[0])} 个请求完成")
        summarize(label, results)

    # TODO(you)：回答（写进观察记录，S2 前再看一遍）：
    #  1) A 相对 B 的平均完成步数差多少 %？其中「批尾空转」贡献了哪些浪费？
    #  2) C 的 preemption 让谁变好、谁变差？（对比 B：短请求的 P90 变了吗）
    #  3) 把 C 的抢占策略换成「抢队头」重跑，哪个指标变化最大？
    #  4) 用 AQE 的话说：B→C 的收益来自什么？（运行时状态反馈进调度 = adaptive）


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录 ═══════════════════════════════
# - 三策略对比数字：
# - 「批尾空转」的具体例子（哪一批、空了几个 slot、空了多少步）：
# - preemption 的代价与受益者：
# - AQE 映射的新条目（同步进 notes_W3.md）：
