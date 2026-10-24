#!/usr/bin/env python3
"""
W1-01 · torch.compile 四层栈总览：Dynamo → AOTAutograd → Inductor → CUDAGraphs
===============================================================================
小白先补三句背景：
    - PyTorch 平时怎么跑代码？你写一行它立刻算一行，这种「边解释边执行」的模式
      叫 eager 模式（eager = 急着马上干）。灵活直观，但每个小算子都要单独调用
      一次底层函数，调用开销累积起来很可观。
    - torch.compile 就是一个「把整段模型代码翻译成更快的底层代码再执行」的开关。
      翻译流水线一共四道工序，就是标题里那四层。
    - 四层各干什么（一句一层，配厨房类比）：
        1. Dynamo      ：盯着你的 Python 代码执行一遍，把它「画」成一张计算流程图
                         （叫 FX graph）。≈ 把口述菜谱整理成标准步骤清单。
        2. AOTAutograd ：负责自动求导（训练用）；我们只做推理，它基本直接放行。
        3. Inductor    ：拿流程图去生成真正高效的代码——GPU 上生成 Triton kernel，
                         Mac CPU 上生成 C++/OpenMP（OpenMP = CPU 多线程并行库）。
                         ≈ 按步骤清单直接排好一条流水线。
        4. CUDAGraphs  ：把一串 GPU 调用「录成录像带」，以后整段一键重放，
                         省掉每次逐条发指令的开销。仅 NVIDIA GPU 有效。

目标：
    用一个最小模型，把每层的「输入→输出」亲手掳出来看一遍：
      1. Dynamo      ：捕获 Python → FX graph（symbolic trace 的产物）
      2. AOTAutograd ：训练场景才有存在感；推理 forward 下基本透传（看到即可）
      3. Inductor    ：FX graph → 生成代码（Mac CPU 上是 C++/OpenMP；GPU 上是 Triton）
      4. CUDAGraphs  ：mode="reduce-overhead" 才启用；GPU 专属（Mac 上知道即可）

运行（Mac，CPU 即可）：
    python 01_stack_overview.py
    TORCH_LOGS=output_code python 01_stack_overview.py   # 多打印生成的 kernel 代码

高频名词（一行一个，后面每个文件都会用到）：
    - 算子（op） ：模型里的一个计算步骤，如「卷积」「加法」「乘系数」。
    - kernel    ：底层真正执行计算的函数；一个算子（或几个融合后的算子）对应一个 kernel。
    - 融合(fusion)：把相邻几个小算子合并成一个大 kernel。为什么快？省掉中间结果
                  在显存/内存里的来回搬运——W5–W8 我们会反复算这笔账。

读代码时对照（数据库类比，可选，不懂可跳过）：
    FX graph  ≈ 逻辑计划（没有物理算子选择）
    Inductor 代码 ≈ whole-stage codegen 的产物（物理计划落成机器码前最后一步）

验收：
    脚本跑完，你手里应该有：一份 FX graph 打印、一份生成的 kernel 代码位置。
"""
from __future__ import annotations

import os
import tempfile

import torch
import torch.nn as nn

# ── 一个「值得编译」的小模型：刻意写出可融合的算子链 ────────────────────
# 为什么这样设计：conv + bn + relu + mul 是教科书级融合组；
# eager 模式下每个算子各跑一个 kernel，融合后应合并成更少的 kernel。
# 算子白话：conv = 卷积（提取特征）；bn = BatchNorm（把数值拉回正常范围）；
#          relu = 激活函数（负数归零）；mul = 逐元素乘个系数。
# 它们都是「逐元素类」小算子——不需要跨位置的信息，最适合融合。
class MiniFuse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)     # 图片 3 色通道进、8 通道出、3x3 小窗口；padding=1 让尺寸不变
        self.bn = nn.BatchNorm2d(8)                   # 规范化层；8 = 通道数，必须和 conv 的输出通道一致
        self.scale = nn.Parameter(torch.tensor(1.5))  # 一个可学习常数，本文件里就当「乘 1.5」用

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)      # 第 1 步：卷积提特征
        h = self.bn(h)        # 第 2 步：数值规范化
        # TODO(you)：思考——推理模式（eval）下 BN 会被折叠成什么？
        #   提示：BN 训练时用「当前这批数据」的均值/方差，推理时改用「训练完存下来的」
        #   全局均值/方差——于是它变成固定的 y = x·γ + β，只是一次乘加，最适合融合。
        #   去后面落盘的生成代码里找证据（应该看不到独立的 bn kernel）。
        h = torch.relu(h)     # 第 3 步：激活（负数归零）
        # TODO(you)：思考——这个 mul 有没有资格融进上面？
        #   提示：逐元素乘法不需要邻居位置的信息 → 是融合候选。
        #   验证方法：去落盘目录数一数，conv 之外到底还剩几个函数。
        return h * self.scale  # 第 4 步：乘一个可学习系数


def main() -> None:
    torch.manual_seed(0)         # 固定随机种子：让每次运行产生同一套随机权重/输入，结果可复现
    model = MiniFuse().eval()    # 推理场景要 eval()（BN 的行为差异：训练用批统计，推理用存好的统计）
    x = torch.randn(1, 3, 32, 32)  # 一张 32x32 的 3 通道假图片（最前面的 1 是 batch 大小）

    # ══ 第 0 层基线：eager，先拿正确答案 ════════════════════════════════
    # no_grad：告诉 torch「不用记录求导信息」——推理都该包上，省内存也更快。
    # ref = reference，标准答案；后面编译版的输出要跟它对数（数值必须一致，只许快、不许错）。
    with torch.no_grad():
        ref = model(x)

    # ══ 第 1 层：Dynamo —— 掏出 FX graph ═══════════════════════════════
    # explain() 是「只诊断、不加速执行」：让 Dynamo 把捕获流程走一遍，然后报告
    # 抓到几张图、断了几次、抓到哪些算子——正适合第一次学习。
    # 版本敏感：不同 torch 版本的调用签名有差异；若这种写法报错，
    #   试另一种形式：torch._dynamo.explain(model, x)（参数直接传进去）。
    explanation = torch._dynamo.explain(model)(x)
    print("== Dynamo 捕获结果 ==")
    # 三个数字怎么读：graph_count = 抓到几张计算图（理想是 1 张）；
    # graph_break_count = 中途放弃、退回逐行执行的次数（断一次就少优化一段）；
    # ops = 抓到的算子清单长度。
    print(f"  graph count      : {explanation.graph_count}")
    print(f"  graph break 数    : {explanation.graph_break_count}")
    print(f"  op 数（汇总）     : {len(explanation.ops)}")

    # TODO(you)：把 graph break 的原因逐条抄进 notes_W1.md（若有），
    # 并回答：这个 break 掉回 eager 后，哪一段代码是「没被编译的」？

    # ══ 第 2 层：Inductor —— 看生成的代码 ═══════════════════════════════
    # TORCH_LOGS=output_code：环境变量开关，让 Inductor 把生成的 C++/Triton 代码
    # 打印到 stderr（错误输出流）；setdefault = 若你已手动设置就不覆盖。
    os.environ.setdefault("TORCH_LOGS", "output_code")
    compiled = torch.compile(model)  # 这行只是「包一层壳」，还没有真正编译
    with torch.no_grad():
        out = compiled(x)            # 首次调用才触发编译+执行（懒编译），所以这一步会卡几秒到几十秒，正常

    # 和第 0 层的标准答案对数。rtol/atol = 允许的相对/绝对误差；
    # 编译只许改变「跑多快」，不许改变「算出什么」。
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    print("数值一致 ✓（eager vs compiled）")

    # Inductor 的落盘目录：/tmp/torchinductor_<user>/ （版本敏感，路径可能带 hash）
    # 「落盘」= 生成的代码会写成真实文件存在磁盘上，下次运行直接复用
    #（这就是第二次跑同一个模型变快的原因之一）。
    root = os.path.join(tempfile.gettempdir())
    candidates = [d for d in os.listdir(root) if d.startswith("torchinductor")]
    print(f"Inductor 产物目录候选：{[os.path.join(root, c) for c in candidates]}")
    # TODO(you)：进目录找到 kernel 对应的 .py（CPU 上通常是 __wrapper_*.py + c 字符串），
    # 回答三件事：
    #   1) conv 之外的那些算子（bn/relu/mul）被合成几个函数？
    #   2) 代码里有没有循环展平/向量化痕迹？
    #   3) 和你想象中的「融合」差距在哪？

    # ══ 第 3 层：CUDAGraphs —— 只看一眼 ═════════════════════════════════
    # mode="reduce-overhead" 才启用；Mac CPU 上无意义，S1 会话里跑：
    #   compiled_ro = torch.compile(model, mode="reduce-overhead")
    # 面试级一句话：CUDAGraphs 把一整段 kernel launch 序列录制成一张图，
    # 重放时省掉 CPU 侧逐个 launch 的开销——decode 阶段小 kernel 密集时的关键。
    # 白话版：launch = CPU 向 GPU 下达「跑这个 kernel」的命令。推理时每生成一个
    # token 都要发起成百上千个小 kernel，CPU 发号施令的速度反而成了瓶颈（overhead）；
    # 录成「一盘录像带」整段重放，发号施令只需要一次。
    print("CUDAGraphs：本文件不跑，见上方注释 + S1 会话。")


if __name__ == "__main__":
    main()

# ═══════════════════════════════ 观察记录（跑完就填）═════════════════════
# - graph break 数 & 原因：
# - 生成的 kernel 里，bn/relu/mul 的融合形态：
# - 让我意外的点：
