#!/usr/bin/env python3
"""
W8-01 · FA2 forward 复现：online softmax 是唯一的新知识
=============================================================
先补几句背景（attention 在算什么、为什么难写快）：
    白话版：序列里每个词都要「环顾四周」——拿自己的提问（Q）去和
    所有词的索引（K）比对，得到一组注意力分数，softmax 归一化成
    权重，再按权重把所有词的内容（V）加权吸收进来。
    公式里 B=batch（几个独立样本）、H=头数（几组独立注意力）、
    N=序列长度（多少个词）、d=每头维度（每个词的向量宽度）。

    为什么它难写快？朴素做法要先把整个 N×N 的分数矩阵算出来、
    存到显存，再 softmax、再乘 V——N=4096 时这个矩阵就要好几 GB。
    FA 的全部魔法就一句话：**不存大矩阵，边扫边算边归并**。
    而「边扫边归并」在数学上可行，靠的就是 online softmax。

名词白话：
    - causal（因果）mask：只许看自己及前面的词、不许偷看未来——
      LLM 生成时必须如此（否则训练时直接抄到答案了）；
    - running 量：随扫描不断更新的「小结」——不用回头翻全量数据；
    - scale（1/sqrt(d)）：除以 sqrt(d) 是为了让分数方差不随维度
      撑爆（d 越大点积越大，不缩放的话 softmax 会退化成非 0 即 1）。

任务：
    O = softmax(QK^T / sqrt(d)) @ V      Q/K/V: [B, H, N, d]，O 同形

═══════════════════════════════════════════════════════════════════════════
online softmax 推导（先在纸上过一遍再写代码）：
  softmax 的朴素定义要求「看到全部分数」才知道分母。但可以维护两个 running 量：
  （数字直觉：已见分数 [3, 5]，来了新块 [7]——最大值从 5 跳到 7，
   之前撜的 exp 和必须整体乘上 exp(5-7) 缩回来。「换基准、老账折算」
   就是下面那条 rescale 的来历。）
      m = 目前见过的最大分数      l = 目前见过的 exp 分数和
  来了一个新块（分数 s）之后：
      m' = max(m, max(s))
      l' = l * exp(m - m') + sum(exp(s - m'))     ← 老贡献按新 max 重缩放！
  最后输出：O = acc / l（acc 的维护同理，见下）
  这就是数据库人熟悉的「流式聚合」：max/sum 都能增量维护，代价是老的 sum
  要按新基准 rescale——softmax 的 exp 结构恰好让 rescale 只是一个乘法。

  acc 的对应维护（V 加权累积）：
      acc' = acc * exp(m - m') + exp(s - m') @ V_block
═══════════════════════════════════════════════════════════════════════════

运行（容器）：
    TRITON_INTERPRET=1 python 01_fa2_forward.py
"""
from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def fa_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, H, N,
    stride_qb, stride_qh, stride_qn,      # q[b,h,n,d] 的四级跨度
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_on,
    scale,                                 # 1/sqrt(d)
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # ══ 3D grid：(q 块 × 头 × batch)——每个 program 负责一段 q 的全部输出 ══
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # 本块的 q 行号
    offs_n = tl.arange(0, BLOCK_N)                        # k/v 列号（循环内平移）
    offs_d = tl.arange(0, HEAD_DIM)                       # 特征维（constexpr，见签名）
    mask_m = offs_m < N

    # q 的 [BLOCK_M, d] tile：一次性载入并在整个 k 循环里重用
    q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh \
             + offs_m[:, None] * stride_qn + offs_d[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # running 状态（online softmax 的三件套——每行 q 各自记各自的账）
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)   # running max：见过最大分数
    # 白话：初值 -inf = 「还没见过任何分数」——任何真分数都比 -inf 大。
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                 # running sum：exp 分数之和
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)        # running 加权 V（分子）

    # causal 时 k 只需要走到本 q 块的右边界（FA 的一半工作量省在这）
    # 白话：第 i 行 q 只允许看 k 的 0..i 列——k 列号一旦越过本块 q 的
    # 最大行号，整块都是 -inf（全被 mask 掉），干脆不进循环。
    if CAUSAL:
        k_end = tl.minimum(N, (pid_m + 1) * BLOCK_M)
    else:
        k_end = N

    for start_n in range(0, k_end, BLOCK_N):
        offs_k = start_n + offs_n
        mask_n = offs_k < N

        k_ptrs = k_ptr + pid_b * stride_kb + pid_h * stride_kh \
                 + offs_k[None, :] * stride_kn + offs_d[:, None]
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)     # [d, BLOCK_N]
        # 看下标拼法：offs_d 在行、offs_k 在列 → 拿到的直接是 K^T 的 tile，
        # q @ k 不用显式转置就是 [BLOCK_M, BLOCK_N] 的分数矩阵。

        # ── TODO(you)：六步补完 online 循环体 ─────────────────────────────
        # 每步配白话（与文件头推导一一对照）：
        # 1) 分数：s = tl.dot(q, k) * scale          # [BLOCK_M, BLOCK_N]
        #    ⚠️ causal mask：score 只在 q_pos >= k_pos 时有效，
        #    s = tl.where(offs_m[:, None] >= offs_k[None, :], s, float('-inf'))
        #    （非 causal 分支没有这步——用 if CAUSAL: 包住）
        #    白话：-inf 的格子 exp 后自动变 0 = 权重 0 = 不吸收未来。
        # 2) 新 max：m_new = tl.maximum(m_i, tl.max(s, axis=1))
        #    白话：新基准 = max(老最大, 本块最大)。
        # 3) 本块概率：p = tl.exp(s - m_new[:, None])
        #    白话：减 max 防 exp 溢出（W5-02 学过），这里多一层含义——
        #    减的是**新**基准，所以老数据稍后要跟着折算。
        # 4) rescale 因子：alpha = tl.exp(m_i - m_new)
        #    白话：老账折算率——老账按老基准 m_i 记，换到新基准要乘
        #    exp(老-新)（新 ≥ 老，所以 ≤1，账只会缩不会涨）。
        # 5) 状态更新：
        #    l_i = l_i * alpha + tl.sum(p, axis=1)     # 老和折算 + 本块新增
        #    acc = acc * alpha[:, None]                # 老分子同步折算
        # 6) 吃进本块 V：v = tl.load(v_ptrs ...)  # [BLOCK_N, d]，v_ptrs 照 k_ptrs 的
        #    样子用 stride_v* 拼；acc += tl.dot(p.to(v.dtype), v)
        #    白话：p 当权重、v 当内容，本块的加权 V 累进分子。
        #    p.to(v.dtype)：dot 两边 dtype 必须一致（TC 契约）。
        #    然后 m_i = m_new（别忘了收尾归位——忘了它下一圈折算就全错了）
        raise NotImplementedError("六步补完后删掉本行")

    # 归一化：O = acc / l（softmax 的最后一步除分母，折算到底才完整）
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)     # 全 -inf 行保护（理论不出现，防手抖）
    acc = acc / l_safe[:, None]
    o_ptrs = o_ptr + pid_b * stride_ob + pid_h * stride_oh \
             + offs_m[:, None] * stride_on + offs_d[None, :]
    tl.store(o_ptrs, acc, mask=mask_m[:, None])


# ── host 侧封装：HEAD_DIM 作为 constexpr 传入（每个 d 一次特化）────────
def fa_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    assert q.ndim == 4 and q.shape == k.shape == v.shape
    B, H, N, d = q.shape
    assert d in (16, 32, 64, 128), "HEAD_DIM 必须是 16 的倍数（dot 约束）"
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 32, 32
    # grid 三维：q 块数 × 头数 × batch——与 kernel 里三个 program_id 一一对应
    grid = (triton.cdiv(N, BLOCK_M), H, B)
    fa_fwd_kernel[grid](
        q, k, v, o, B, H, N,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        d ** -0.5,
        CAUSAL=causal, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=d,
    )
    return o


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    """SDPA 参考实现（PyTorch 官方 kernel，S4 的性能对照组）。"""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def main() -> None:
    torch.manual_seed(0)
    B, H, N, d = 1, 4, 128, 64
    q = torch.randn(B, H, N, d)
    k = torch.randn(B, H, N, d)
    v = torch.randn(B, H, N, d)

    for causal in (False, True):
        ref = reference(q, k, v, causal)
        out = fa_forward(q, k, v, causal)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
        print(f"causal={causal} ✓（对照 SDPA）")

    # TODO(you)（数值过了之后）：
    #  1) 白板推导检查：不看文件头，重写 m'/l' 两条递推，直到不看也能写对。
    #  2) 对照 Triton 官方 06-fused-attention：它的 qk_scale、diagonal mask、
    #     STAGE 划分（两段循环）与你的差异；STAGE 的「分两段省 mask 判断」
    #     是你清单的第一张卡片。
    #  3) 思考：为什么 causal 的 k_end 截断能省一半计算？对照 W3 的
    #     「batch 收缩」——调度层和 kernel 层的「省」是一回事吗？
    #  4) 换 N=257 跑一遍（N 不整除 BLOCK）——mask 边界考点。


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录 ═══════════════════════════════
# - causal / 非 causal 数值结果：
# - N=257 边界用例结果：
# - 官方实现与我差异最大的三处：
# - rescale 那行（alpha）我第一次写对了吗：
