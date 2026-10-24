#!/usr/bin/env python3
"""
W4-02 · S2 探测客户端：给 serve 打三种负载
=============================================================
先补几句背景（这个脚本干什么）：
    服务拉起来后，得有人不断向它发请求才能观察调度行为。
    本脚本就是「压力器」：能发出三种不同形状的负载，
    分别触发 W3 学过的三种调度现象。

目标：
    在 Mac 上把「请求怎么发」写死、验证逻辑；S2 现场只换 --endpoint。
    三种模式对应 04_observation_points.md 的实验：
      baseline   ：1 条短 + 1 条长输出，验证链路与 TTFT 口径
      burst      ：16 个并发、变长 prompt/输出 → 观察实验 A（continuous batching）
      steady_load：固定并发持续 N 秒 → nsys 抓 decode 阶段（S2 的 nsys 段）

名词白话：
    - 并发（concurrency）：同一时刻在跑的请求数；
    - thread（线程）：Python 里让多件事「同时」进行的机制——这里每个线程
      负责发一个请求，16 个线程齐发就是 burst 模式；
    - usage 字段：服务端回复里自带的「计数器」（输入了多少 token、
      输出了多少 token）——比自己数更准；
    - stream（流式）：回复逐字往外吐（像打字机）还是一次性全给。
      非流式拿到完整回复更简单，适合第一次实验。

运行：
    python 02_probe_client.py --endpoint http://127.0.0.1:1 --mode baseline
    （Mac 空跑：预期在 HTTP 连接处失败——报错=逻辑通，见 main 尾部自检）

依赖：requests（Mac 上 pip install requests；租卡机上一般自带）
"""
from __future__ import annotations

import argparse
import sys
import threading   # 线程库：burst/steady 模式的并发引擎
import time

try:
    import requests   # requests：Python 最常用的 HTTP 库（比内置 urllib 好用）
except ImportError:  # Mac 空跑环境可能没装：给出明确指引而不是堆栈
    print("缺依赖 requests → pip install requests（S2 现场租卡机一般自带）")
    sys.exit(2)

PROMPTS_SHORT = ["用一句话解释什么是 PagedAttention。"]
PROMPTS_LONG = [
    "把下面这段话扩写成 400 字左右的科普文：SGLang 的 continuous batching "
    "指的是每个 step 都允许新请求加入、完成请求退出，而不必等整批结束。"
] * 4


def post_chat(endpoint: str, prompt: str, max_tokens: int, timeout: float = 120.0) -> dict:
    """发一条 chat 请求，返回计时与用量。

    用 stream=False + usage 字段拿 completion_tokens——口径简单可靠；
    TTFT 观察留给进阶（可改 stream=True 读首 chunk 时间，TODO 见尾部）。
    """
    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": "default",           # SGLang 单模型服务一般忽略模型名，现场确认
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=timeout)   # 发请求（阻塞到回复回来）
    dt = time.perf_counter() - t0                          # 秒表：这一去一回花了多久
    r.raise_for_status()        # HTTP 状态码不是 2xx 就抛异常（提前暴露服务端问题）
    body = r.json()             # 回复体是 JSON 格式，解析成 Python 字典
    usage = body.get("usage", {})   # 服务端自带的用量统计；get 带 {} 默认值防缺字段
    return {"latency_s": dt, "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens")}


def mode_baseline(endpoint: str) -> None:
    print("[baseline] 短输出请求...")
    print("   ", post_chat(endpoint, PROMPTS_SHORT[0], max_tokens=32))
    print("[baseline] 长输出请求...")
    print("   ", post_chat(endpoint, PROMPTS_LONG[0], max_tokens=512))


def mode_burst(endpoint: str, concurrency: int = 16) -> None:
    """并发变长负载：16 个请求同时打出去，各自记完成时间。

    为什么输出长度要「故意参差」？——让请求们陆续完成，
    才能在日志里观察到 continuous batching 的「补位」：
    有人完成退出，空出的座位立刻被 waiting 里的新请求填上。
    """
    results: dict[int, dict] = {}
    lock = threading.Lock()   # 锁：多线程同时写字典会乱，锁保证一次只有一个写

    def worker(i: int) -> None:
        prompt = PROMPTS_LONG[i % len(PROMPTS_LONG)]
        max_tokens = 64 + (i % 4) * 128      # 输出长度故意参差 → batch 收缩可见
        try:
            out = post_chat(endpoint, prompt, max_tokens)
        except Exception as exc:  # noqa: BLE001
            out = {"error": repr(exc)}       # 单个请求失败不炸全局：记下错误继续
        with lock:
            results[i] = out

    t0 = time.perf_counter()
    # 每个请求一个线程，齐发；join = 等全部线程干完再往下走
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0   # wall = 挂钟时间（从发出到全部回来的总时长）
    ok = [r for r in results.values() if "error" not in r]
    print(f"[burst] 并发={concurrency} 完成={len(ok)} 总耗时={wall:.1f}s")
    for i in sorted(results):
        r = results[i]
        if "error" in r:
            print(f"   req{i:02d}  ERROR {r['error']}")
        else:
            print(f"   req{i:02d}  latency={r['latency_s']:6.2f}s  completion={r['completion_tokens']}")
    # TODO(you)（S2 现场）：把完成顺序与 latency 画成「谁先完成」的时间线，
    # 对照 serve.log 里 ADD/COMPELETED 行——验证 continuous batching 的补位。


def mode_steady_load(endpoint: str, seconds: float = 60.0, concurrency: int = 8) -> None:
    """稳态负载：维持固定并发直到时限（给 nsys 一个稳定窗口）。

    与 burst 的区别：burst 是「一波打完就停」；steady 是「水位恒定」——
    每个线程完成一个立刻发下一个，总并发始终不变，服务进入稳定状态，
    这时用 nsys 录下的时间线才好分析（否则录到的是开机/收尾的噪声）。
    """
    stop_at = time.perf_counter() + seconds   # 截止时刻（绝对时间）
    n_done = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal n_done   # nonlocal：允许内层函数改外层函数的变量
        while time.perf_counter() < stop_at:   # 只要还没到点，就一直发
            try:
                post_chat(endpoint, PROMPTS_LONG[0], max_tokens=128)
                with lock:
                    n_done += 1
            except Exception as exc:  # noqa: BLE001
                print("   worker error:", repr(exc))
                time.sleep(1.0)   # 出错别疯狂重试：歇 1 秒再试，避免雪上加霜

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"[steady_load] 并发={concurrency} 时长={seconds}s 完成请求数={n_done}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--mode", default="baseline", choices=["baseline", "burst", "steady_load"])
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    try:
        if args.mode == "baseline":
            mode_baseline(args.endpoint)
        elif args.mode == "burst":
            mode_burst(args.endpoint)
        else:
            mode_steady_load(args.endpoint, seconds=args.seconds)
    except requests.ConnectionError as exc:
        # Mac 空跑的预期终点：连接失败 = 脚本逻辑走通，S2 换真端点即可。
        print(f"连接失败（Mac 空跑预期行为）：{exc}")
        print("→ 逻辑自检通过：参数解析/线程/异常路径都没炸。S2 现场把 --endpoint 指向真服务。")
    # TODO(you)（选做进阶）：stream=True 读 TTFT（首 chunk 时间），
    # 给实验 A/B 增加一个「首 token 延迟」口径——decode 与 prefill 交错最敏感的指标。


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录（S2 现场回填）═══════════════════
# - baseline：短/长请求的 latency 与 completion_tokens：
# - burst：完成时间线（谁先谁后）与 serve.log 的对照结论：
# - steady_load：60s 完成数（= 吞吐口径）：
