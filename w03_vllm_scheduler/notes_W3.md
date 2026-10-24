# notes_W3 · 调度数据流标注图（草稿）+ AQE 映射

> 本周里程碑。S2 会话后升级为正式版：把日志里抓到的真实证据标到图上。

## 一、vLLM 一个 step 的数据流（ASCII，照 01_reading_map 的证据画）

```
（TODO(you)：照着下面的模板填，箭头旁标「数据/判断」）

HTTP 请求
   │ prompt tokens
   ▼
waiting 队列 ──(准入：块够吗？batch 满吗？)──► running 集合
   ▲                                          │
   │            ◄──(preempt: 块不够，抢谁？)── ┘
   │                                          ▼
   │                              scheduler.schedule() 每 step 一次
   │                                          │ scheduler_output:
   │                                          │   谁的哪些 token 这一步算
   │                                          ▼
   │                              worker/executor（GPU 真执行）
   │                                          │ 新 KV → 写入 block 账本
   │                                          ▼
   └── 完成的请求 ──── 采样/detokenize ──── 流式返回
```

填图要求：至少把「准入判断」「preempt 判断」「token 预算」三处的**源码证据**
（文件:方法）标在箭头旁边。

## 二、四个机制的「一句话 + 证据」

| 机制 | 一句话（自己的话，不抄注释） | 源码证据 |
|---|---|---|
| continuous batching | | |
| preemption (recompute vs swap) | | |
| chunked prefill | | |
| PagedAttention block 账本 | | |

## 三、AQE 映射（本周最终版）

| AQE 的做法 | vLLM 的对应 | 差距（=幼稚清单素材） |
|---|---|---|
| 运行时收集 join 数/基数统计 | ？（block 使用率有统计吗？用在哪？） | |
| 统计驱动策略切换 | ？（chunk 大小/抢占策略是静态配置吗？） | |
| 逻辑计划 → 物理计划分层 | 调度与执行在一个循环里 | |

## 四、模拟器实验结论（02/03 的数字）

- static vs continuous：
- preemption 代价：
- 我的结论一句话：

## 五、S2 想验证的问题（转给 w04/04_observation_points.md）

1.
2.
3.
