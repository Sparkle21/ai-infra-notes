# W5 · Triton 入门（11/16–11/22）

> 计划：学 API 与 block 编程模型；写 fused RMSNorm；Mac 容器里 `TRITON_INTERPRET=1` 对数值。
> 里程碑：数值正确的 fused RMSNorm
> 环境：**Mac OrbStack（Ubuntu 容器）+ triton + TRITON_INTERPRET=1**（Mac 原生装不了 GPU 版 triton）

## 容器环境（一次性，10 分钟）

```bash
# OrbStack 起一个 Ubuntu 容器，目录挂载进去（按你的实际路径改）
orb create -m "$PWD" ubuntu24 triton-lab
orb -m triton-lab bash -c "apt update && apt install -y python3-pip python3-venv"

# 容器内装 CPU torch + triton（不需要 CUDA 版！interpret 模式纯 Python 执行）
orb -m triton-lab bash
  python3 -m venv ~/venv && source ~/venv/bin/activate
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install triton
  # 验证
  TRITON_INTERPRET=1 python -c "import triton; print(triton.__version__)"
```

## 文件与学习顺序

| 序 | 文件 | 学什么 | 验收 |
|---|---|---|---|
| 01 | [01_vector_add.py](01_vector_add.py) | tl.load/tl.store、program_id、block 概念 | 数值 PASS |
| 02 | [02_fused_softmax.py](02_fused_softmax.py) | 一维 reduction + online 数值技巧 + fused 版意义 | 数值 PASS + 观察记录 |
| 03 | [03_fused_rmsnorm.py](03_fused_rmsnorm.py) | 本周里程碑：fused RMSNorm（行级 reduction） | 数值 PASS |
| — | [notes_W5.md](notes_W5.md) | Triton 心智模型笔记 | 写完 |

## block 编程模型一分钟版

- GPU 上有几千个「program」（≈ 一个 CUDA block），每个 program 处理一段数据；
- 你写的 kernel 是**单个 program 的视角**：`tl.program_id(0)` 告诉你「我是谁」；
- `tl.arange(0, BLOCK)` 造一段下标，`tl.load/tl.store` 按下标整块读写；
- `tl.sum/tl.max` 做 block 内 reduction；
- 屏蔽（mask）负责边界：形状不整除 BLOCK 时的正确性全靠它。

## 本周纪律（铁律的源头）

- **每个 kernel 三步走**：先写 reference（torch 原生）→ 再写 triton → interpret 对数值。
- 数值对了才有资格谈性能；性能是 W6/S3 的事，本周不跑计时。
- mask 写错是第一大坑：所有文件里 mask 相关行都标了 ⚠️，重点盯。
