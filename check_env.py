#!/usr/bin/env python3
"""
环境自检：跑任何学习代码之前先跑我。
====================================
用法：
    python check_env.py

这个脚本做什么（一句话）：
    检查这台机器上「学习要用的软件装齐没有」，并按 10 周计划告诉你：
    哪些内容现在这台机器就能跑、哪些需要换环境（Ubuntu 容器 / 租的 GPU）。

先认识几个会反复出现的名词（一行白话一个）：
  - torch    ：PyTorch，深度学习框架。W1/W2 的实验全靠它。
  - triton   ：一种「用 Python 语法写 GPU 代码」的语言，W5–W8 手写 kernel 用。
  - vllm     ：当前最主流的开源 LLM 推理服务框架，W3/W4 读它的源码。
  - CUDA     ：NVIDIA 显卡的开发接口；只有 NVIDIA 卡才谈得上 CUDA。
  - MPS      ：苹果 M 系列芯片上 PyTorch 的 GPU 加速方式。
  - kernel   ：真正执行计算的小函数（GPU 或 CPU 上）。本课程天天见。
  - TRITON_INTERPRET=1：环境变量开关。打开后 Triton kernel 改用纯 Python 模拟执行，
               不需要 GPU —— 在 Mac 上验证 kernel「算得对不对」全靠它。

我会告诉你：
  1. 当前是哪个 Python、装没装 torch / triton / vllm；
  2. torch 能用 CPU / CUDA / MPS 哪个后端（后端 = 实际干活的硬件）；
  3. 按 10 周计划逐项判断「哪些周的代码在当前环境能跑」。

对应 plan.md 第二节「环境与硬件」：Mac 日常工位 + 租卡集中测试场。
"""
from __future__ import annotations

# importlib.util：用来「探问」某个包装没装——find_spec 只找安装位置、不真正导入
#（真导入一个没装的包会直接报错崩掉，探问不会）
import importlib.util
import os
import platform
import shutil  # shutil.which：在 PATH 里找某个命令行工具装在哪（相当于终端里敲 which nsys）
import sys


def _has(module: str) -> bool:
    """问一句「这个包装了吗」：find_spec 去找它的安装位置，找得到（不是 None）就是装了。"""
    return importlib.util.find_spec(module) is not None


def main() -> None:
    # sys.version.split()[0]：只取版本号，如 '3.11.6'；platform.machine()：CPU 架构（arm64 / x86_64）
    print(f"python      : {sys.version.split()[0]}  ({platform.machine()})")
    # os.environ.get(名字, 默认值)：读环境变量，没设置就显示默认值而不是报错
    print(f"TRITON_INTERPRET = {os.environ.get('TRITON_INTERPRET', '(未设置)')}")
    print()

    # ── torch：W1/W2 的地基 ────────────────────────────────────────────
    torch_ok = False
    if _has("torch"):
        import torch

        torch_ok = True
        backends = ["cpu"]              # 后端 = 实际干活的硬件；CPU 永远可用，先记上
        if torch.cuda.is_available():   # 有 NVIDIA 显卡吗？
            backends.append(f"cuda:{torch.cuda.get_device_name(0)}")  # 有 → 连卡名一起报，如 cuda:NVIDIA H20
        if getattr(torch, "mps", None) is not None and torch.mps.is_available():
            backends.append("mps")      # 苹果 M 系芯片的 GPU 加速可用（Mac 上的「显卡」）
        print(f"torch       : {torch.__version__}  后端: {', '.join(backends)}")
        # torch.compile 是 2.0 才加入的 API；老版本没有它，W1/W2 的实验全部免谈
        print(f"  compile   : {'可用' if hasattr(torch, 'compile') else '不可用（版本太老）'}")
    else:
        print("torch       : 未安装 → W1/W2 跑不了；Mac 上装 CPU 版即可：pip install torch")

    # ── triton：W5–W8 的地基 ──────────────────────────────────────────
    triton_ok = False
    if _has("triton"):
        import triton

        triton_ok = True
        interpret = os.environ.get("TRITON_INTERPRET") == "1"  # 开关是否已打开（"1" = 开）
        note = "interpret 模式已开 ✓" if interpret else "interpret 模式未开（Mac 容器里记得 export TRITON_INTERPRET=1）"
        print(f"triton      : {triton.__version__}  ({note})")
    else:
        print("triton      : 未安装 → W5–W8 需在 OrbStack 的 Ubuntu 容器里装（Mac 原生装不上，见 w05 README）")

    # ── vllm：W3 读源码用 clone 仓库即可，serve 只在租卡机 ─────────────
    # serve = 把模型跑起来、对外提供问答服务。Mac 没有显卡跑不动 7B 模型，
    # 所以 Mac 上只读 vllm 源码，真正起服务放到租的 GPU 机器上做（流程见 sessions/）。
    if _has("vllm"):
        import vllm

        print(f"vllm        : {vllm.__version__}")
    else:
        print("vllm        : 未安装（正常——Mac 上只读源码，serve 在租卡机上装）")

    # ── CLI 工具：只在租卡机上有意义 ──────────────────────────────────
    # 两个 NVIDIA 官方的性能分析器（profiler = 帮你看「时间/资源花在哪」的工具）：
    #   nsys（Nsight Systems）：看整条时间线——哪些 kernel 何时运行、GPU 有没有闲着；
    #   ncu （Nsight Compute）：钻进单个 kernel 内部——带宽用了多少、计算单元忙不忙。
    for tool in ("nsys", "ncu"):
        path = shutil.which(tool)
        print(f"{tool:<10} : {path if path else '不在 PATH（租卡机上才有；Mac 的 nsys 是 GUI 版，另装）'}")

    print()
    print("结论速查：")
    if torch_ok:
        print("  - 当前环境可跑：W1/W2（torch.compile 实验）、W3（调度模拟器，纯标准库+dataclasses）")
    if triton_ok:
        print("  - 当前环境可跑：W5–W8 kernel 的数值验证（务必 TRITON_INTERPRET=1）")
    if torch_ok and "cuda" in torch.backends.__dict__:
        pass  # 占位：租卡机上会命中 cuda 分支，提示见上
    print("  - 真实 benchmark / nsys / ncu：只在租卡实例上（流程见 sessions/）")
    # benchmark = 性能测试（比快慢）。记住本课程的分工口号：
    # 「数值在 Mac 验（算得对不对），性能在租卡测（跑得快不快）」。


if __name__ == "__main__":
    main()
