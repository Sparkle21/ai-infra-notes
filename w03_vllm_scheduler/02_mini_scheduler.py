#!/usr/bin/env python3
"""
W3-02 · 最小调度器：先把「状态机 + 资源账本」写在自己手上
=================================================================
先补几句背景（这个文件在模拟什么）：
    你问 ChatGPT 一句话，服务端不是只伺候你一个——同时有成百上千个请求，
    大家分率同一块显存。谁先上、谁等待、显存不够时牺牲谁，
    这就是「调度器」的活。vLLM 的调度器只有几百行核心逻辑，
    我们用 200 行把它 reconstruct（重建）一遍，彻底搞懂。

    一个请求的一生：到达 → 排队(waiting) → 获准进入运行(running) →
    逐 token 生成（每步多吃一点显存）→ 完成 → 释放显存。
    两个队列 + 一本显存账，就是调度器的全部世界。

名词白话：
    - prompt：你发给模型的问题部分（输入）；
    - token：模型处理文字的最小单位；生成回复就是逐个 token 吐；
    - KV cache：模型「记住自己读过的内容」的内存。每生成 1 个 token
      都要追加一段 KV，所以占的显存随生成不断增长——这是调度矛盾的核心；
    - preemption（抢占）：显存不够时，强行把某个在跑的请求踢出去、
      释放它的显存给幸存者（被踢的回头重算，代价很大）；
    - watermark（水线）：账本里预留的安全余量块，防止贴着上限运行时
      突然不够用。

目标：
    不碰 GPU、不碰 vLLM，用 200 行写一个推理调度器的最小内核：
      - 请求 = (到达时间, prompt 长度, 输出长度)
      - 资源 = KV cache 块账本（PagedAttention 的分页账本的极简版）
      - 调度 = 每个 step 决定「谁进 batch、谁出 batch」

    你只写两个函数（标 TODO(you)），其余 harness 全部就绪。

运行：
    python 02_mini_scheduler.py            # 跑默认 8 个请求的剧本
    python 02_mini_scheduler.py --seed 42  # 换剧本

模型抽象（与真实 vLLM 的对应，现在看不懂没关系，W3 读源码时回看）：
    Request          ≈ vllm SequenceGroup
    blocks 账本      ≈ kv_cache_manager / BlockSpaceManager
    step()           ≈ scheduler.schedule() + worker 执行
    waiting/running  ≈ 同名两个队列

验收：
    跑通后填观察记录；02 的产出会被 03 复用（03 import 本文件的类）。
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field


# ── 世界参数 ──────────────────────────────────────────────────────────────
# 白话：KV cache 不是无限长的，被切成固定大小的「块」管理（像书一页一页记）；
# 三个数字定义了资源压力：块总数越少，抢占越容易发生，竞争越真实。
BLOCK_TOKENS = 16          # 每个 KV block 装 16 个 token（vLLM 默认 16）
TOTAL_BLOCKS = 64          # 账本总块数（故意设小，让资源压力真实存在）
MAX_BATCH = 8              # 一个 batch 最多几个请求


@dataclass
class Request:
    """一个推理请求。dataclass = 自动生成 __init__ 等样板代码的类。

    前三个字段是「剧本设定」（生成时就定死）；后面两个是「运行时状态」（每步变化）。
    """
    rid: int                     # 请求编号（日志里 r0、r1 就是它）
    arrive_step: int       # 第几步到达
    prompt_len: int              # 问题有多长（token 数）——决定入场时要一次占多少块
    output_len: int              # 要生成多少个 token 才算完
    # 运行时状态
    generated: int = 0           # 已经生成了几个 token
    blocks: int = 0        # 已占块数（prompt + 已生成）

    @property
    def needed_tokens(self) -> int:
        """当前总共需要记多长的 KV = 问题长度 + 已生成长度。"""
        return self.prompt_len + self.generated

    def blocks_for(self, extra_tokens: int = 0) -> int:
        """当前若再生成 extra_tokens 个 token，需要占多少块（向上取整）。

        为什么要向上取整？——块是「整块分配」的：151 个 token 要 10 块
        （9 块装 144 个，第 10 块只装 7 个也占一整块）。最后一块的空位
        就是「内碎片」，分页管理的代价。
        """
        import math

        return math.ceil((self.needed_tokens + extra_tokens) / BLOCK_TOKENS)

    @property
    def done(self) -> bool:
        """生成够了 output_len 个 token → 请求完成。"""
        return self.generated >= self.output_len


@dataclass
class Ledger:
    """KV block 账本：vLLM 用 watermark（低水位）预留安全余量。

    白话：就是一本「显存存折」——总额度多少、已用多少、
    以及故意不花完的保底余量（防止一步没算准就崩）。
    """
    total: int = TOTAL_BLOCKS    # 总块数（额度）
    used: int = 0                # 已用块数
    watermark: int = 2      # 预留块数（对照 vLLM 的 watermark_blocks）

    @property
    def free_for_admission(self) -> int:
        """「可用于接收新请求」的剩余块 = 总数 - 预留 - 已用。

        注意：预留的 watermark 块不对外发放——它专门留给在跑的请求
        下一步长大用（每步 decode 都可能要新块）。
        """
        return max(0, self.total - self.watermark - self.used)


@dataclass
class SimReport:
    finished_at: dict[int, int] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)


def make_script(rng: random.Random, n: int) -> list[Request]:
    """生成剧本：到达时间聚簇，长短 prompt 混合——制造资源竞争。

    白话：故意让请求们几乎同时到、有的问题特长——这样显存很快就挤爆，
    抢占才会真的发生，你才能观察到调度器的「艰难时刻」。
    """
    reqs = []
    for i in range(n):
        reqs.append(
            Request(
                rid=i,
                arrive_step=rng.choice([0, 0, 1, 1, 2, 3]),   # 大多在前几步内到达（聚簇）
                prompt_len=rng.choice([32, 48, 120, 200]),   # 长 prompt 制造 prefill 压力
                output_len=rng.randint(8, 40),
            )
        )
    return sorted(reqs, key=lambda r: (r.arrive_step, r.rid))   # 按到达先后排好


# ══════════════════════ 你要写的两个函数 ══════════════════════════════════
def admit(req: Request, ledger: Ledger, running: list[Request]) -> bool:
    """准入决策：waiting 里的这个请求这一步能不能进 running？

    vLLM 的真实决策考虑：剩余块（watermark 之后）、block 大小、batch 上限。
    你要写：
      1. batch 没满（len(running) < MAX_BATCH）；
      2. 账本在扣掉 watermark 后，装得下它的整个 prompt（req.blocks_for() 块）。
    """
    # TODO(you)：把上面两个条件写成代码，返回 bool。
    # 写法提示：
    #   if len(running) < MAX_BATCH and req.blocks_for() <= ledger.free_for_admission:
    #       return True
    #   return False
    # （看完提示先自己盲写一遍再对照——两行代码，但每一行都对应 vLLM 里的真检查。）
    raise NotImplementedError


def pick_decode_victims(running: list[Request], ledger: Ledger) -> list[Request]:
    """所有 running 请求这步都要各长 1 个 token（decode）。
    若账本装不下「所有 running 各 +1 token」，需要抢出牺牲者（preemption）。

    vLLM 的策略（V0 默认 recompute 模式）：从**队尾**（最新加入的）开始抢，
    直到剩下的请求都能继续。返回被抢的 Request 列表（从 running 移除，
    generated 归零、释放全部块、回 waiting 队列头）。
    """
    # TODO(you)：实现「从最新加入的开始抢」；返回牺牲者列表。
    # 写法提示：
    #   victims = []
    #   running 内部本来就是按加入顺序排的 → 从末尾往前抢；
    #   每抢一个：victims.append(它)，并检查剩下的请求各 +1 token 后账本装不装得下；
    #   装得下就停手，返回 victims。（抢哪个/释放几个块的后续处理 harness 已代劳。）
    # 进阶（写完基础版再改）：从队头抢（保护新请求），对比两个策略的日志差异。
    raise NotImplementedError


# ══════════════════════ 调度主循环（harness，勿改）════════════════════════
def simulate(reqs: list[Request], verbose: bool = True) -> SimReport:
    """调度主循环：每个 step 依次干四件事（到达→准入→防炸→长 token）。

    这就是你以后在 vLLM scheduler.py 里会看到的同名循环的极简版。
    读代码时把四步的注释当作说明书。
    """
    rep = SimReport()                        # 报告：每个请求第几步完成 + 全程日志
    waiting: list[Request] = []              # 等待队列：还没获准进运行的请求
    running: list[Request] = []              # 运行队列：正在逐 token 生成的请求
    ledger = Ledger()                        # 显存账本
    t = 0                                    # 时钟：第几步（一次 forward = 一步）
    ptr = 0                                  # 剧本指针：已「到达」到第几个请求
    while len(rep.finished_at) < len(reqs) and t < 500:   # 全部完成或超过 500 步就停
        # 1) 到达：把这一步该来的请求放进 waiting
        while ptr < len(reqs) and reqs[ptr].arrive_step <= t:
            waiting.append(reqs[ptr])
            ptr += 1
        # 2) 准入（对 waiting 逐个问；未通过的留在队列）
        still: list[Request] = []
        for r in waiting:
            if admit(r, ledger, running):
                r.blocks = r.blocks_for()    # 一次占下 prompt 需要的全部块
                ledger.used += r.blocks      # 账本记上
                running.append(r)            # 进运行队列
                rep.log.append(f"t={t:3d} ADMIT   r{r.rid} (+{r.blocks}blk)")
            else:
                still.append(r)              # 没获准的继续排队（不删除，下步再问）
        waiting = still
        # 3) decode 一步（含 preemption 处理）
        # 先算：如果所有 running 各再长 1 个 token，总共还要新占几块？
        need = sum(r.blocks_for(1) - r.blocks for r in running)
        # 装不下（连 watermark 都得动了）→ 必须抢出牺牲者，否则有请求会「爆显存」
        if need > ledger.free_for_admission + ledger.watermark and running:
            for victim in pick_decode_victims(running, ledger):
                ledger.used -= victim.blocks   # 归还它的全部块
                victim.blocks = 0
                victim.generated = 0            # 已生成的 token 白算了（recompute 的代价）
                running.remove(victim)
                waiting.insert(0, victim)      # 抢占者回队头（recompute 模式优先重进）
                rep.log.append(f"t={t:3d} PREEMPT r{victim.rid} -> waiting")
        # 4) 长 token & 收尾
        for r in running:
            r.generated += 1                   # 每个在跑的请求长 1 个 token
            new_blocks = r.blocks_for()        # 长大后需要占几块
            ledger.used += new_blocks - r.blocks   # 补上差价（多数步是 0，偶数步 +1）
            r.blocks = new_blocks
            if r.done:                         # 生成够数 → 完成，释放全部块
                ledger.used -= r.blocks
                r.blocks = 0
                running.remove(r)
                rep.finished_at[r.rid] = t
                rep.log.append(f"t={t:3d} FINISH  r{r.rid} (steps={t - r.arrive_step})")
        t += 1
    if verbose:
        for line in rep.log:
            print(line)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    reqs = make_script(rng, args.n)
    rep = simulate(reqs)
    print(f"\n完成 {len(rep.finished_at)}/{len(reqs)} 个请求，总步数 = {max(rep.finished_at.values()) + 1}")
    # TODO(you)（观察并记录）：
    #  1) 有没有发生 PREEMPT？发生在谁身上（新请求还是老请求）？
    #  2) r 长输出短输出的完成步数差，说明账本竞争主要卡在哪类请求？
    #  3) 换 --seed 跑三次，PREEMPT 的频率稳定吗？（为 03 的策略对比做铺垫）


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录 ═══════════════════════════════
# - 是否出现 PREEMPT / 牺牲者是谁：
# - 三种 seed 下的 PREEMPT 频率：
# - 与 vLLM scheduler.py 对上的证据（文件:方法）：
