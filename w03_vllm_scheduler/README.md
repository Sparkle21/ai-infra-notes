# W3 · vLLM 调度器（11/02–11/08）

> 计划：读 scheduler、continuous batching、preemption（recompute vs swap）、chunked prefill；对照 AQE 运行时自适应记映射。
> 里程碑：调度数据流标注图（草稿）→ [notes_W3.md](notes_W3.md)
> 环境：Mac + 本地 clone 的 vllm 仓库（只读）；模拟器纯 Python（torch 都不需要）

## 准备：clone 仓库

```bash
git clone --depth 1 https://github.com/vllm-project/vllm.git ~/src/vllm
```

## 文件与学习顺序

| 序 | 文件 | 学什么 | 验收 |
|---|---|---|---|
| 01 | [01_reading_map.md](01_reading_map.md) | vLLM 调度相关源码的地形图 | 每个条目填上证据（文件:类/方法） |
| 02 | [02_mini_scheduler.py](02_mini_scheduler.py) | 自己写一个最小调度循环（FCFS/strict） | 跑通并填观察记录 |
| 03 | [03_continuous_batching_sim.py](03_continuous_batching_sim.py) | 加 continuous batching / preemption 策略 | 三个策略函数写完，对比曲线 |
| — | [notes_W3.md](notes_W3.md) | 调度数据流标注图（ASCII 草稿）+ AQE 映射 | 草稿完成 |

## 为什么先写模拟器再读源码

调度逻辑的本质是**状态机 + 资源账本**（谁在跑、KV cache 还剩几块、谁等着），
这与 GPU 无关——纯 Python 模拟器让你把「决策」和「执行」剥离开看。
写完模拟器再读 vLLM 的 scheduler.py，你会带着「这里应该有个 if」的预期去读，
读源码的速度和理解深度完全不同。

## AQE 映射（本周随时记）

| AQE（你熟） | vLLM 调度（新） | 备注 |
|---|---|---|
| 运行时统计（runtime filtering/join 数） | running/waiting 队列、block 使用率 | |
| 动态切换 join 策略 | chunked prefill 开关、preemption 策略 | |
| 空stage动态合并 | continuous batching 的 slot 回收 | |
| 代价估算驱动重优化 | ？（大概率没有——写进幼稚清单） | |

## 提醒

- vLLM V0/V1 两代架构并存（V1 走 async scheduler）；**以你 clone 的版本 main 分支为准**，
  01_reading_map 里两边入口都给了，先确认你仓库里哪个活着。
