#!/usr/bin/env python3
"""
W1-03 · guard 与 recompile：编译缓存的钥匙是什么
=================================================================
先补一句背景（W1-01 的延续）：
    torch.compile 编译一次很贵（秒级）。所以它把编译结果存起来，
    下次遇到「一样的情况」就直接复用，不再重编。
    怎么判断「一样不一样」？编译产物入口处放了一组快速检查，叫 guard
    （哨兵）：检查输入的形状（shape）、数据类型（dtype）等是否和上次一致。
    全部一致 → 缓存命中，直接用；有一项不一致 → 缓存未命中，重新编译
    （recompile）。recompile 太频繁，加速就变成减速。

    类比：编译产物是已做好的「专用模具」，guard 是门口的验票员——
    拿同一种票（同 shape/dtype）就放行进快通道；换了票型就得重新做一个模具。

目标：
    guard 是 dynamo 在编译产物入口放的一组「快速检查」（shape、dtype、
    Python 对象身份……）。任何一项变化 → cache miss → 重新编译。
    本文件用计数器把这件事看得清清楚楚：
      1. 同 shape 反复调用 → 编译 1 次，之后全命中；
      2. 换一个 shape → 触发一次 recompile（第 2 次编译）；
      3. 换 dtype / 换模块属性 → 又一次 recompile；
      4. 动态 shape（dynamic=True）下 shape 变化不再逐个重编。

运行（Mac，CPU 即可）：
    python 03_guard_recompile.py

名词白话：
    - shape：张量的形状，如 (8, 8) 表示 8x8 的矩阵；
    - dtype：数据类型，如 float32（常用）/ float64（双精度）；
    - cache miss：缓存未命中——要找的现成结果不在，只能重新干一遍。

vLLM 连接点（W3/W4 铺垫）：
    decode 阶段 batch size 每个 step 都在变（continuous batching）。
    如果每变一次就 recompile，服务就废了——所以 vLLM 对 cudagraph 的
    capture 是「按 batch size 桶」来的。今天先看见 guard 的存在，
    W4 在 serve 日志里你会再遇见它。

验收：
    输出里能看到 recompile 计数随实验 1/2/3 递增，实验 4 不增（或增得少）。
"""
from __future__ import annotations

import torch

torch._dynamo.reset()   # 清零所有编译缓存，保证实验可重复（不然上次跑的缓存会干扰计数）


def counters() -> tuple[int, int]:
    """返回 (编译次数, cache 命中帧数)。版本敏感：字段名以 torch.__version__ 为准。

    白话：dynamo 内部维护一本台账（counters 字典），每编译一次、每命中一次
    缓存都会记一笔。这里就是去查这本台账。
    """
    counters_ = torch._dynamo.utils.counters
    frames = counters_["stats"]["unique_graphs"] if "stats" in counters_ else 0
    return int(frames), int(counters_["stats"].get("ok", 0))


def plain_fn(x: torch.Tensor) -> torch.Tensor:
    """被测函数：极简的「乘2→截断负数→加1」链，三个小算子都是好融合的原料。"""
    return torch.relu(x * 2.0) + 1.0


def main() -> None:
    cfn = torch.compile(plain_fn)  # 先包壳；真正的编译发生在第一次调用时

    # ══ 实验 1：同 shape 重复调用 → 只编译一次 ═══════════════════════════
    x = torch.randn(8, 8)
    for _ in range(5):
        cfn(x)                     # 第 1 次编译，后 4 次全部命中缓存（guard 全过）
    n1, _ = counters()
    print(f"实验1 同 shape×5 调用后  编译次数 = {n1}")

    # ══ 实验 2：换 shape → recompile ════════════════════════════════════
    cfn(torch.randn(16, 16))      # (8,8)→(16,16)：shape 检查不过，只能重新编译一份
    n2, _ = counters()
    print(f"实验2 换 shape 后        编译次数 = {n2}   （+{n2 - n1} = shape guard 未命中）")

    # ══ 实验 3：换 dtype → 再 recompile ═════════════════════════════════
    cfn(torch.randn(16, 16, dtype=torch.float64))   # float32→float64：类型检查不过，又编一份
    n3, _ = counters()
    print(f"实验3 换 dtype 后        编译次数 = {n3}   （+{n3 - n2} = dtype guard 未命中）")

    # ══ 实验 4：dynamic=True → shape 变化不再逐个重编 ═══════════════════
    torch._dynamo.reset()       # 先清缓存，计数从零开始，实验间互不干扰
    dfn = torch.compile(plain_fn, dynamic=True)
    # dynamic=True 的意思：编译时把形状当「变量」而不是「常数」——
    # 一份代码适配多种形状，代价是运行时多算几个形状相关的参数。
    for s in (8, 16, 32, 64):
        dfn(torch.randn(s, s))  # 4 种形状轮流调用
    n4, _ = counters()
    print(f"实验4 dynamic=True 下 4 种 shape → 编译次数 = {n4}")
    # TODO(you)：解释 dynamic=True 用什么代价换来了「shape 免疫」
    # （提示：符号 shape 让生成的 kernel 里出现 stride/shape 的运行期参数，
    #  某些特化优化（如完全展开、常量折叠到形状）就做不了了。
    #  白话：模具不再是「为 8x8 量身定制」而是「可调尺寸的万能模具」——
    #  什么尺寸都能用，但每个尺寸都达不到量身定制的极致效果。）

    # ══ guard 到底长什么样？（掏出来看）══════════════════════════════════
    # TODO(you)：跑 TORCH_LOGS=guards python 03_guard_recompile.py，
    # 挑一条 guard 打印抄到观察记录——形如 TENSOR_MATCH 或 GUARD 说明
    # 它检查的具体字段。对照数据库：guard ≈ plan cache 的 key 前置校验。

    # ══ recompile 上限的「保险丝」════════════════════════════════════════
    # torch._dynamo.config.cache_size_limit 默认 8（版本敏感）。
    # 超限行为：新 shape 直接走 eager，不再尝试编译——静默降速，最阴险的一类坑。
    # 白话：「保险丝」= 重编太多次后 torch 放弃治疗，新形状一律用慢速老模式跑，
    # 而且不报错、不提醒——服务就这么不明不白地慢了，你却以为自己在享受加速。
    print(f"\ncache_size_limit = {torch._dynamo.config.cache_size_limit}")
    # TODO(you)：把 limit 临时改成 1（torch._dynamo.config.cache_size_limit = 1）
    # 再跑实验 2，观察「超过 limit 后发生了什么」，写进观察记录。


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录 ═══════════════════════════════
# - 四个实验的编译次数：
# - 抄下的一条 guard：
# - dynamic=True 的代价（我的理解）：
# - 超 cache_size_limit 后的现象：
