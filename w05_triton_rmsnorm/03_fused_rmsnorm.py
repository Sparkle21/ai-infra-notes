#!/usr/bin/env python3
"""
W5-03 · fused RMSNorm（本周里程碑：数值正确的版本）
=============================================================
RMSNorm 是什么（LLaMA/Qwen 等 decoder-only 模型的标配）：
    y = x / sqrt(mean(x^2) + eps) * w
    ——比 LayerNorm 少一个减均值，但结构同款：行归约 + 行广播 + 逐元素乘权重。

    逐个词解释这个公式：把一行的「均方根」算出来（归约），
    用它把整行缩放到标准量级（广播），再乘可学习权重 w（逐元素）。
    为什么需要它：防止网络层数加深后数值越变越大（爆炸）或越变越小（消失）。

名词白话：
    - eps（ε）：公式里的小常数（如 0.000001）——分母可能为零时用来兜底，
      避免除零；
    - rsqrt：1/√x 的缩写（「倒数平方根」），比分开算除法和开方更快；
    - broadcast（广播）：把一个数（如整行共用的 rms）「复制铺开」
      到整行每个位置——与归约正好是一对反操作（多→一 vs 一→多）。

为什么它是练手圣体：
    1. 覆盖两大模式：reduction（算 rms）+ broadcast（rms 广播回整行）+ pointwise（乘 w）；
    2. 三种算子各自独立时是 3 个 kernel、HBM 来回 3 趟；融合后 1 个 kernel 1 趟——
       W6 的 benchmark 与论证笔记全拿它当主角；
    3. vLLM 里就有它的 Triton 实现（vllm/model_executor/layers/layernorm.py，
       或 FlashInfer 后端）——你写完可以去对答案（先自己写，再对！）。

运行（容器内）：
    TRITON_INTERPRET=1 python 03_fused_rmsnorm.py
"""
from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")   # 先保证数值对（同 W5-01 的铁律）

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def rmsnorm_kernel(
    x_ptr,          # [n_rows, n_cols] 输入
    w_ptr,          # [n_cols] 逐元素权重（每行共用同一套 w）
    out_ptr,        # [n_rows, n_cols] 输出
    stride,         # 行跨度（第 row 行的开头在 row*stride 处）
    n_cols,
    eps,            # 防除零的小常数（运行期传进来，不是 constexpr——它不影响代码结构）
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)        # 一个 program 包一行（同 W5-02 的行级模式）
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols            # ⚠️ 贯穿 load 与 store 的边界保护

    # TODO(you)：五步补完（softmax 的变奏，注意平方与 rsqrt）：
    # 1) x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0)
    #    （这里 other=0.0 是对的：x^2 后被 mask 位置贡献 0，正好不污染分母——
    #     和 softmax 的 -inf 各有各的道理，想清楚为什么）
    #  2) ss = tl.sum(x * x, axis=0) / n_cols        # 均方（分母用真实列数，不是 BLOCK_SIZE！）
    #     ⚠️ 这是本 kernel 最容易踩的坑：分母若用 BLOCK_SIZE（含假下标），
    #     不整除的行算出来的 rms 就会偏小——数值错但不报错，最难查。
    #  3) rms = 1 / tl.sqrt(ss + eps)                 # eps 防零
    #  4) w = tl.load(w_ptr + cols, mask=mask, other=0.0)   # 权重也按列取一份
    #  5) tl.store(out_ptr + row * stride + cols, x * rms * w, mask=mask)
    #     （x * rms * w：行内每列同乘 rms（广播）再各乘自己的 w（逐元素））
    raise NotImplementedError("五步补完后删掉本行")


def triton_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.ndim == 2 and x.is_contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n_cols)
    rmsnorm_kernel[(n_rows,)](x, w, out, x.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK)
    return out


def torch_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """reference：eager 三算子（pow/mean/div+mul 语义版），W6 的对照口径。

    白话：这是「不融合」的写法——每步一个 torch 算子，中间结果都要落回
    内存。它算得慢但可读性最好，当标准答案正合适。
    """
    ss = x.pow(2).mean(dim=-1, keepdim=True)   # 每行的均方（keepdim 保持形状方便后面广播）
    return x * torch.rsqrt(ss + eps) * w


def main() -> None:
    torch.manual_seed(0)
    # 三个形状分别对应：单头 d_model / decode 期大 batch / FFN 宽度（S3 会扫它们）
    for n_rows, n_cols in [(16, 4096), (128, 4096), (64, 11008)]:
        x = torch.randn(n_rows, n_cols)
        w = torch.randn(n_cols)
        ref = torch_rmsnorm(x, w)

        out = triton_rmsnorm(x, w)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
        print(f"[{n_rows}, {n_cols}] fused ✓")

    # 边界：非 2 的幂列宽（BLOCK 补齐路径）
    # 白话：1000 不是 2 的幂 → BLOCK 取 1024，多出 24 个假下标靠 mask 兼容——
    # 这一组就是在验证「mask + other=0.0 不污染均方分母」。
    x = torch.randn(8, 1000)
    w = torch.randn(1000)
    torch.testing.assert_close(triton_rmsnorm(x, w), torch_rmsnorm(x, w), rtol=1e-5, atol=1e-5)
    print("[8, 1000] 非 2 幂列宽 ✓")
    
    # 全零行（eps 的存在意义）
    # 白话：全零行的均方=0，没有 eps 就是除零 → inf/NaN；eps 兜底后输出 0。
    x = torch.zeros(2, 64)
    w = torch.ones(64)
    out = triton_rmsnorm(x, w)
    assert torch.isfinite(out).all(), "全零行输出必须有限（eps 兜底）"
    print("全零行有限 ✓（eps 生效）")

    # TODO(you)（写完数值后）：
    #  1) 打开 vLLM 源码 vllm/model_executor/layers/layernorm.py，
    #     对照它的 rmsnorm triton kernel 与你的差异（变量名/写法/参数顺序），
    #     把「它有而我没有」的细节记进观察记录——那通常是踩过坑的人留下的。
    #  2) 思考：为什么它把 BLOCK_SIZE 做成 constexpr 而不是运行期参数？
    #     （提示：编译期常量才能展开循环/寄存器分配——W7 的 tl.dot 还会用到。）
    #  3) 进阶（选做）：宽行循环分块版——n_cols 8192+ 时单个 BLOCK 放不下，
    #     用 for 块循环累积 ss。写完对数值，这版就是「工业版」的形状。


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录 ═══════════════════════════════
# - 与 vLLM 实现的差异点：
# - constexpr 的原因（我的话）：
# - 全零行 / 非 2 幂列宽有没有踩坑：
# - 本文件完成后，W6 的三口径（eager/decomposed/fused）中我已有两口径：
#   eager（torch.nn.RMSNorm）在 w06/01_bench_core.py 里等我。
