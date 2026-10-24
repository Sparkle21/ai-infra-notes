#!/usr/bin/env python3
"""
W10-02 · S5 会话 A/B 驱动器：bf16 vs int8 的端到端对比
=============================================================
先补几句背景（A/B 是什么、这个脚本干什么）：
    W10-01 证明了 int8 kernel「算得对、跑得快」——但那只是单个算子。
    用户真正关心的是端到端：同一个模型，bf16 版和 int8 版各自起服务，
    吞吐/时延/显存差多少？A/B 测试就是这场对照实验：只换一个变量
    （量化），其余全部不动，两轮数字摆在一起比。
    本脚本 = 打负载的「压力器」+ 收数字的「记录员」，不负责起停服务。

名词白话：
    - A/B 测试：两个版本各跑一轮、只差一个变量的对照实验；
    - steady / burst：两种负载形态——steady=恒定并发持续压（看吞吐
      上限），burst=一波并发同时涌入（看排队时延）；
    - P50 / P90：把所有请求的时延从小到大排队，第 50%/90% 位置的值。
      P90 比 P50 更能暴露「最差的那些请求」有多惨；
    - usage.completion_tokens：服务端回报的实际生成 token 数——
      吞吐账本比「数请求数」更细的计量单位。

职责边界（刻意收窄）：
    - 不自动化 serve 生命周期（起/停服务用 ../w04_serving_lab/01_serve_7b.sh，
      A/B 只是换 MODEL/参数——S5 现场手动跑，本脚本只负责「打负载 + 收数字」）；
    - 三口径：steady_load 吞吐（60s 完成数）、burst P50/P90 时延、显存（手动记录）。

用法（S5 现场每档各跑一遍）：
    # 档 1：bf16 服务起来后
    python 02_ab_driver.py --tag bf16
    # 档 2：换 int8 模型/参数重启服务后
    python 02_ab_driver.py --tag int8
    # 汇总（把两次的 ab_<tag>.json 合并成对比表）
    python 02_ab_driver.py --summarize

S5 的两档启动方式（写死在下面，现场按版本确认）：
    bf16: bash ../w04_serving_lab/01_serve_7b.sh                  # 原样
    int8: MODEL=<int8量化模型> bash ../w04_serving_lab/01_serve_7b.sh
          （或加 --quantization xxx，以你装的 SGLang 版本为准——W9/02 的侦察结果决定）
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent          # 输出文件（ab_<tag>.json）放本目录
# 所有请求共用同一条 prompt——保证 bf16/int8 两轮输入完全一致
#（A/B 的「只差一个变量」纪律），内容也刻意贴合 W10 的量化主题。
PROMPT = ("把下面这段话扩写成 300 字的科普文：模型量化是把权重和激活从 16 位浮点"
          "压到 8 位整数，用 scale 换回精度。")


def post_chat(endpoint: str, max_tokens: int, timeout: float = 120.0) -> tuple[float, int]:
    """发一条对话请求，返回 (耗时秒, 生成 token 数)。

    懒加载 requests：只在真发请求时才 import——--summarize 路径
    不发请求，机器上没装 requests 也能出汇总表。
    """
    import requests

    payload = {
        "model": "default",                                 # SGLang 单服务单模型，写死即可
        "messages": [{"role": "user", "content": PROMPT}],   # OpenAI 兼容的对话格式
        "max_tokens": max_tokens,                           # 最多生成多少 token（输出长度的旋钮）
        "temperature": 0.0,                                 # 贪心解码：输出确定，A/B 才可比
    }
    t0 = time.perf_counter()                     # 发出请求前开始抹表
    r = requests.post(f"{endpoint}/v1/chat/completions", json=payload, timeout=timeout)
    dt = time.perf_counter() - t0                # 整个往返耗时（排队+prefill+decode 全在内）
    r.raise_for_status()                         # HTTP 非 2xx 直接抛错
    return dt, r.json()["usage"]["completion_tokens"]   # 服务端统计的实际生成数


def measure(endpoint: str, tag: str) -> dict:
    """一轮完整测量：warmup → steady 60s 吞吐 → burst 16 并发时延分布。"""
    print(f"[{tag}] warmup...")
    post_chat(endpoint, max_tokens=16)    # 预热一发：把首请求的建连/编译类成本先花掉

    # ① steady 吞吐：并发 8 持续 60s
    # 白话：8 个线程像 8 条流水线各自不停发请求，谁完成立刻发下一条，
    # 看 60 秒这条「传送带」总共送出多少单 = 吞吐上限。
    print(f"[{tag}] steady_load 60s（并发 8）...")
    stop_at = time.perf_counter() + 60.0    # 60 秒后的统一停工时间
    n_done = [0]                            # 完成计数器（单元素列表：闭包里可变）
    lock = threading.Lock()                 # 8 个线程一起改计数器，上锁防打架

    def worker() -> None:
        while time.perf_counter() < stop_at:      # 没到停工时间就一直发
            try:
                post_chat(endpoint, max_tokens=128)
                with lock:                        # 改共享计数器前先拿锁
                    n_done[0] += 1
            except Exception:  # noqa: BLE001
                time.sleep(1.0)   # 单条出错（如超时）歇 1 秒再试，不让单点故障停掉整轮

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()          # 8 个线程同时开工
    for t in threads:
        t.join()           # 主线程等它们全部跑完（到停工时间自然退出）
    throughput = n_done[0] / 60.0    # 吞吐 = 完成请求数 ÷ 60 秒

    # ② burst 时延：16 并发、参差输出
    # 白话：16 个线程同一瞬间各发 1 条请求（一波涌入），各自抹表。
    # 输出长度刻意参差（64/192/320/448 四档）——像真实流量，也让
    # 「短请求被长请求堵住」的排队效应显形（continuous batching 的考场）。
    print(f"[{tag}] burst（并发 16）...")
    latencies: list[float] = []

    def burst_worker(i: int) -> None:
        try:
            # i%4 决定四种输出档：64+0/128/256/384 → 64/192/320/448 参差
            dt, _ = post_chat(endpoint, max_tokens=64 + (i % 4) * 128)
        except Exception as exc:  # noqa: BLE001
            print(f"   req{i} error: {exc!r}")   # 单条失败只记账，不炸整轮
            return
        with lock:
            latencies.append(dt)    # append 也是共享操作，同样要锁

    threads = [threading.Thread(target=burst_worker, args=(i,)) for i in range(16)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    latencies.sort()

    result = {
        "tag": tag,                                    # bf16 / int8——json 文件名也用它
        "throughput_req_per_s": round(throughput, 2),  # ① 吞吐（越高越好）
        "burst_wall_s": round(wall, 1),                # 16 条全部完工的总墙钟
        "latency_p50_s": round(statistics.median(latencies), 2) if latencies else None,      # ② 中位数
        "latency_p90_s": round(latencies[int(len(latencies) * 0.9) - 1], 2) if latencies else None,  # ② 尾部
        # 显存请手动记录（nvidia-smi），填进 json 再 summarize
        "vram_MiB_manual": None,                       # ③ 手动填（自动读需 GPU 机权限）
    }
    out_path = OUT / f"ab_{tag}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[{tag}] 写入 {out_path}：{result}")
    print(f"[{tag}] 手动步骤：nvidia-smi 记显存 → 编辑 json 的 vram_MiB_manual 字段")
    return result


def summarize() -> None:
    """把两轮的 ab_<tag>.json 拼成一张对比表（A/B 的「看结果」环节）。"""
    files = sorted(OUT.glob("ab_*.json"))   # 找本目录下所有测量结果
    if not files:
        print("没有 ab_*.json（先分别 --tag bf16 / --tag int8 各测一轮）")
        return
    rows = [json.loads(f.read_text()) for f in files]
    print(f"{'tag':>6} | {'吞吐 req/s':>10} | {'P50 s':>6} | {'P90 s':>6} | {'显存 MiB':>9}")
    print("-" * 55)
    for r in rows:
        print(f"{r['tag']:>6} | {r['throughput_req_per_s']:>10} | {str(r['latency_p50_s']):>6} | "
              f"{str(r['latency_p90_s']):>6} | {str(r['vram_MiB_manual']):>9}")
    # TODO(you)（S5 后，写 notes_W10.md）：
    #  - 吞吐差、时延差、显存差三个数字各自归因（哪个属于 W9-01 的三个账本之一？）
    #  - 若 int8 吞吐没优势：是 weight-only（带宽账）而你的负载是 compute-bound（长输出）吗？
    #    负载怎么设计才能让带宽账显出来（短 prompt 长 decode？高并发？）——A/B 结论的边界。


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--tag", choices=["bf16", "int8"])
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    if args.summarize:
        summarize()      # 汇总模式：不发请求，只读本地 json 拼表
        return
    if not args.tag:
        ap.error("需要 --tag bf16|int8 或 --summarize")   # 两种模式必须选一个
    import requests      # 提前 import：让下面 except 能拿到 requests.ConnectionError 这个名字

    try:
        measure(args.endpoint, args.tag)
    except requests.ConnectionError as exc:
        print(f"连接失败：{exc}\n→ S5 现场：先用 01_serve_7b.sh 起对应档位的服务，再跑本脚本。")


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录（S5 回填）═══════════════════════
# - A/B 汇总表（summarize 输出贴这）：
# - 三口径差异的归因（账本语言）：
# - A/B 结论的边界（什么负载下量化才划算）：
