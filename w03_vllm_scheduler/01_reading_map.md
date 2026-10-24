# W3-01 · vLLM 调度源码地形图

> 仓库：`~/src/vllm`（记下 clone 时的 commit 短 hash）。
> 读法：每个条目先在源码里找到「证据」（文件:类/方法），再写一句你的理解。找不到就写「没找到+版本差异猜测」。

## 一、一个推理 step 的旅程（先背这张图）

```
engine step
   │
   ├─ api_server / entrypoints     HTTP 层，把请求塞进队列（与调度无关，略读）
   │
   ├─ v1/core/  (V1 架构；V0 在 vllm/core/scheduler.py)
   │     ├─ scheduler              ★ 每个 step 调度一次：谁 prefill、谁 decode
   │     ├─ kv_cache_manager       ★ block 账本：分配/释放/前缀复用
   │     ├─ kv_cache_utils         block 表（PagedAttention 的分页管理）
   │     └─ scheduler_output       传给 executor 的「这一步干什么」清单
   │
   ├─ worker / executor            拿着 scheduler_output 真正跑 GPU（W4 观察）
   │
   └─ outputs → 采样 → detokenize → 回 HTTP
```

## 二、逐点侦察清单

| # | 位置（版本敏感，先确认存在） | 管什么 | 我的证据 | 一句话理解 |
|---|---|---|---|---|
| 1 | `v1/core/scheduler.py`（或 V0 `core/scheduler.py`） | 主循环：`schedule()` 的输入输出 | | |
| 2 | 同上 | **调度顺序**：prefill 优先还是 decode 优先？哪段代码定的？ | | |
| 3 | `v1/core/kv_cache_manager.py` | block 分配策略：watermark（低水位）在哪定义 | | |
| 4 | 搜索 `preempt` | **preemption**：recompute vs swap 的选择代码 | | |
| 5 | 搜索 `chunked` / `max_num_batched_tokens` | **chunked prefill**：长 prompt 怎么切 | | |
| 6 | 搜索 `prefix_cach` | prefix caching：命中判断（plan.md 标「非主场，认识即可」） | | |
| 7 | 搜索 `num_new_tokens` / `token_budget` | 每 step 的 token 预算是怎么花的 | | |

## 三、读 scheduler.py 时要回答的五个问题

1. 一个 step 里 running 和 waiting 两个队列的**优先级**是什么？在哪行体现？
2. 什么时候会出现「有请求等着、GPU 却空跑」？（提示：prefill 与 decode 的交错策略）
3. preemption 牺牲谁保护谁？（被抢的是最新来的还是最早来的？）
4. chunked prefill 的 chunk 大小由什么决定？是写死的还是配置的？
5. 整个调度器里有没有任何一个「基于统计的自适应」？（AQE 类比：这是最关键的问题，答案大概率是没有——记进幼稚清单）

## 四、S2 会话预告

本周把「想看的现象」列成实验（A: continuous batching、B: chunked prefill、C: preemption），
S2 会话照着打日志验证——见 `../w04_serving_lab/04_observation_points.md`。

---
## 记录

- clone 的 vllm commit：
- 我走的是 V0 还是 V1 路径（怎么确认的）：
