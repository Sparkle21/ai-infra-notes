<!-- SGLang 推理日志逐行分析手册 v1。
     配套：../w04_serving_lab/01_serve_7b.sh（四档实验，SGLang 版，日志落盘 serve.$EXP.log）。
     分工：04_observation_points.md 管「看什么」（观察点清单），本手册管「每一行怎么读」。
     诚实口径：样例行文按 SGLang 2025 年版的典型格式给出，字段语义来自官方文档
     与源码 logging 调用，非本机实测；行文随版本变——抓字段，别背整行。
     样例数字以 H20 96GB + Qwen2.5-7B（bf16）为演示口径，S2 现场以实测回填。 -->

# SGLang 推理日志逐行分析手册：把每一行读成账本

> **这份手册解决什么问题**：`bash 01_serve_7b.sh` 之后，`serve.base.log` 里哗哗滚出来几十行启动日志、之后源源不断的 `Prefill batch` / `Decode batch` 统计行——每一行其实都在向你汇报账本（显存账、KV 账、吞吐账）。本手册把它们逐行翻译成白话，并标注**每行对应哪个启动参数、能做什么分析**。读完你应当能：不跑任何命令，只看日志就说出"这台机器装了多少 KV、batch 为什么上不去、谁被拉下了车"。

## 0. 三分钟预热：SGLang 是什么、日志为什么值得逐行读

**白话版**：SGLang 是与 vLLM 同代的主流推理引擎（serving 框架），三件事最出名：**RadixAttention**（把前缀缓存做成默认架构，见 3.1）、**调度激进的默认值**、**CUDA graph 用得满**（见阶段 ④）。日志 = 服务的"行车记录仪"：它不参与运算，只是把服务内部每一步的关键状态（加载了多少权重、划了多少 KV、现在 batch 里几条请求）按时间顺序写下来。分析推理性能问题，第一手证据永远是日志——先读日志形成假设，再用工具（`03_scraper.sh` 的采样、nsys 时间线）验证。

**为什么"逐行"值得**：启动日志的每一行都是 SGLang 替你算好的一笔账。比如 `max_total_num_tokens=...` 这一行，背后就是《三处空白》3.2 节那条公式（`2 × 层数 × KV头数 × head_dim × dtype字节`）在真实机器上的落地。逐行读懂它 = 把 W3/W4 学的调度机制和真机数字对上号。

**版本警告（先立规矩）**：SGLang 迭代极快，日志**行文**几乎每版都微调（字段顺序、措辞、开关名），但**字段语义**稳定。所以本手册的读法是"认字段不认整行"：每节先给典型样例，再逐字段拆——现场日志和样例长得不完全一样时，找同义字段即可。SGLang 与 vLLM 的术语/参数对照表见第 7 节（机制同源、叫法不同，W3 的调度模拟器两家通用）。

---

## 1. 基本功：一行日志的解剖 + 三条命令

### 1.1 日志文件是谁写的

`01_serve_7b.sh` 里这行：

```bash
nohup python -m sglang.launch_server ... >serve.base.log 2>&1 &
```

- `>` 把**标准输出**（正常信息）写进 `serve.base.log`；
- `2>&1` 把**错误输出**也并进同一个文件（出错信息不会散落到别处）；
- 结果：心跳信息和 WARNING/ERROR 混在一起按时间排序，`grep` 按关键词过滤即可。

### 1.2 一行的解剖

SGLang 的典型一行长这样（与 vLLM 不同，它用自己的日志器，时间戳带进程标记）：

```text
[2026-11-12 08:30:15 TP0] Decode batch. #running-req: 16, #token: 8192, ...
```

逐段拆：

| 段 | 样例 | 白话 |
|---|---|---|
| 时间戳 | `2026-11-12 08:30:15` | 年-月-日 时:分:秒。**两行时间戳相减 = 事件耗时**（日志分析最常用的招） |
| 进程标记 | `TP0` | **0 号张量并行进程**（张量并行 = 把一个大矩阵切到多卡同算，文章空白二的对象）。单卡永远 TP0；`--tp-size 4` 时有 TP0–TP3 各打各的日志，分析时先 `grep TP0` 只看主进程 |
| 正文 | `Decode batch. ...` | 实际内容，下面逐行拆的就是它 |

### 1.3 三条命令走天下

```bash
tail -f serve.base.log            # 实时跟踪（会话现场盯屏用，Ctrl-C 退出）
grep 'Decode batch' serve.base.log    # 捞全部 decode 统计行
grep 'TP0' serve.base.log | tail -20  # 多卡时只看主进程最近 20 行
```

第 6 节还有一整箱现成命令（提 #running-req 序列画曲线、数 retract 次数等），照抄即可。

---

## 2. 启动日志逐行读（服务就绪之前）

启动日志按时间分五个阶段。**建议的读法**：先 `grep -E 'Load weight end|max_total_num_tokens|Capture cuda graph end|fired up' serve.base.log` 把各阶段"关键账目行"一次捞齐（第 6 节第一条命令），再回来逐行细读。

### 阶段 ①：配置回显

```text
[...] server_args=ServerArgs(model_path='Qwen/Qwen2.5-7B-Instruct', ...,
      tp_size=1, context_length=8192, chunked_prefill_size=4096,
      max_running_requests=None, schedule_conservativeness=1.0,
      quantization=None, ...)
```

- **`model_path='...'`**：回显 `--model` 参数。**先核对它**——加载错模型是最低级也最常见的浪费 GPU 小时事故。
- **`tp_size=1`**：张量并行度（`--tp-size`），1 = 单卡。决定日志里出现 TP 几个进程标记。
- **`context_length=8192`**：单条请求最大长度 = `--context-length` 的回显。**它是 KV 账本的"单条座位价格"**：值越大，一条请求最多可占的 KV 越多（阶段 ③ 的容量除它 = 并发座位数）。
- **`chunked_prefill_size=4096`**：chunked prefill 的每口上限（`--chunked-prefill-size`，V1 时代默认开）。3.4 节看它在日志里怎么现形。
- **`quantization=None`**：没开量化。W10 的 A/B 里 int8/fp8 档会回显量化方式——**A/B 对照第一件事就是确认两档的回显真的只差量化**（第 6 节命令⑦）。

> **能做什么分析**：这一段是"参数自白书"。四档实验（base/A/B/C）对比时，先 `diff` 各自日志的这段回显，确认**真的只差预期的参数**（对照实验的自变量控制；混进别的差异，后面的对比全部作废）。

### 阶段 ②：权重加载

```text
[TP0] Load weight begin. Host memory: ... GB
[TP0] Load weight end. Type=ModelLoader, Time elapsed: 45.2 s.
```

- **`Load weight begin / end`**：从数据盘把权重读进来、搬进显存的起止。**两行时间差 = 加载耗时**——几乎全是磁盘/网络带宽（读盘）+ PCIe 带宽（CPU→GPU 搬运），与推理性能无关，但可当"这台机器 IO 正不正常"的参照（同一模型第二次起服务显著变快 = 页缓存命中）。
- **权重占了多少显存？** SGLang 这段通常不直接报 GiB——从下一阶段的 `avail mem`（剩余显存）反推：`总显存 − avail ≈ 权重 + 已划开销`。心算校验：Qwen2.5-7B 约 7.6B 参数 × bf16 的 2 字节 ≈ 15.2 GB（GiB 口径 14.2）；对不上说明 dtype 或模型不对。

> **能做什么分析**：记下加载耗时和（反推的）权重大小。阶段 ③ 的容量核算都要用它做减法——**它是后面所有账的分母**。

### 阶段 ③：内存池划分（账本第一页：总盘子怎么分）

SGLang 的典型形态：

```text
[TP0] max_total_num_tokens=1,240,000, max_prefill_tokens=16384, ...
[TP0] Memory pool end._avail mem=17.30 GB
```

- **`max_total_num_tokens=1,240,000`**：**KV 仓库的总容量，单位是 token**——本手册最重要的一行。它 = `--mem-fraction-static` 划出的盘子，减掉权重和激活余量后，**全部铺成 KV 块**的存量。W3 学的 PagedAttention/KV 账本，总容量就在这一行。
- **`avail mem=...`**：划完之后手上还剩的余粮（给 CUDA graph 和运行时临时对象用）。太小说明 `--mem-fraction-static` 给得太满，后面 capture 阶段可能 OOM。

**账本公式（这行的来历）**：

```text
max_total_num_tokens ≈ (mem_fraction_static × 总显存 − 权重 − 激活等余量) ÷ 每token KV字节
```

以 H20 96 GiB、`--mem-fraction-static 0.90` 为例：

```text
总盘子 = 0.90 × 96 ≈ 86.4 GiB
KV 可用 ≈ 86.4 − 14.2(权重) − 约3(激活/图缓冲等开销) ≈ 68 GiB
÷ 56 KiB/token（Qwen2.5-7B bf16：2×28层×4个KV头×128×2B）≈ 1,240,000 tokens
```

> **能做什么分析**：
> 1. **对应参数 `--mem-fraction-static`**：它决定"静态盘子"（权重 + KV 一次性划走的部分，区别于运行时动态申请的激活——static 的含义就在这）。实验 C 把它收到 0.35，就是故意把 KV 挤瘦、逼出 retract（3.3 节）。
> 2. **算并发座位数**：`max_total_num_tokens ÷ context_length` ≈ 1,240,000 ÷ 8192 ≈ **151 条顶格并发的座位**。"长上下文贵在哪"就是这行除法：`--context-length` 翻 4 倍，座位数掉到 1/4。
> 3. **验证你的公式**：拿模型配置（层数/KV 头数/head_dim）自己算 KV/token，再拿这行日志对——对上了，说明《三处空白》3.2 的账你真会算了（S2 观察点清单"顺带记录"第一项）。

### 阶段 ④：CUDA graph capture（SGLang 特色，W1 概念的现身）

```text
[TP0] Capture cuda graph begin. This can take up to several minutes.
[TP0] Capture cuda graph end. Time elapsed: 5.96 s. mem usage=1.20 GB. avail mem=16.84 GB.
```

- **`Capture cuda graph begin ... several minutes`**：官方预防针——**这阶段慢是正常的，别以为卡死**。
- **它到底在干什么**：W1-01 学过的 CUDA graph（当时叫"录像带"）——把每个 batch 档位的 decode 一步所需 kernel 启动序列**预录**下来，之后每步直接"回放"，省掉每步几十上百次的 kernel 启动开销。decode 每步只产 1 个 token、本身极短，launch 开销占比高，这个优化对 decode 特别值钱。
- **`Time elapsed: 5.96 s`**：录了多少档位就花多少时间（档位范围由 `--cuda-graph-max-bs` 控制上限）。
- **`mem usage=1.20 GB`**：录像带本身占的显存（从阶段 ③ 的 avail mem 里扣——所以 ③ 里要留余粮）。
- **运行时怎么确认它在生效**：Decode batch 统计行的 `cuda graph: True` 字段（3.2 节）；batch 超出录制范围时该字段变 False，退回逐个 launch 的慢路。

> **能做什么分析**：
> 1. **对应参数 `--cuda-graph-max-bs`**：调大 = 覆盖更大 batch 档位（大 batch 也能回放）但 capture 更久、占显存更多；实验 A 把 `--max-running-requests` 收到 16 后，档位小、capture 秒级结束。
> 2. **诊断"decode 慢"时的必查项**：若 `cuda graph: False` 长期出现，等于白付了启动时的 capture 成本——查是不是 batch 超了 `--cuda-graph-max-bs`。

### 阶段 ⑤：服务就绪

```text
[TP0] The server is fired up and ready to roll!
INFO: Uvicorn running on http://0.0.0.0:8000
```

- **`The server is fired up and ready to roll!`**：SGLang 的招牌就绪行（字面意思"服务器点火完毕、可以开跑"）——**引擎侧初始化全部完成**。
- **`Uvicorn running on ...`**：HTTP 层也就位，正式对外服务。01 脚本的轮询（`curl /health`）从这前后开始返回成功，看到它才允许打流量。

### 附：启动失败（OOM）长什么样

```text
ERROR ... torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.34 GiB.
GPU 0 has a total capacity of 95.99 GiB of which 1.20 GiB is free...
```

- **读法**：`Tried to allocate X GiB`（还差多少）+ `of which Y GiB is free`（还剩多少）。X > Y = 盘子本身不够分。
- **官方调优顺序**（SGLang 官方超参调优文档）：OOM 时依次调 `--chunked-prefill-size`（调小：每口咬小点）→ `--mem-fraction-static`（调小：给运行时留余粮）→ `--max-running-requests`（收紧：并发别堆太满）。三个都有日志观测口（3.4 / ③ / 3.2）。
- **实验 C 特别提示**：0.35 收得过狠时可能"盘子 < 权重"直接 OOM——先心算 `0.35 × 96 = 33.6 > 14.2` 再上机。

---

## 3. 运行时日志逐行读（服务就绪之后）

SGLang 运行时没有"每 10 秒一条"的定时心跳，而是**按事件打行**：每组装一个 prefill 批打一条 `Prefill batch`，每个 decode 步（通常）打一条 `Decode batch`。两类行构成运行时的仪表盘。

### 3.1 Prefill batch 行逐字段拆

```text
[2026-11-12 08:30:15 TP0] Prefill batch. #new-seq: 4, #new-token: 4096,
      #cached-token: 2, token usage: 0.05, #running-req: 0, #queue-req: 4
```

- **`#new-seq: 4`**：这个 prefill 批**新上车几条请求**。
- **`#new-token: 4096`**：这批**真算了多少 token**（每条 prompt 的长度之和，受 `--chunked-prefill-size` 上限约束，3.4 节）。
- **`#cached-token: 2`**：**radix cache 白捡了多少 token**——SGLang 与 vLLM 最大的架构差异在此：RadixAttention 把"前缀树缓存"做成默认架构，相同开头（system prompt、few-shot、压测的固定 PROMPT）的 KV 直接从树里复用，这些 token 不用重算。**这一字段就是命中率的观测口**。
  - **怎么分析**：`#cached-token ÷ (#new-token + #cached-token)` = 本批命中率。w10/02 固定 PROMPT 的压测里应看到它居高；换每条不同的 prompt（02_probe_client 的 burst 模式）则掉回 0。**A/B 两轮的命中率必须同口径**，否则 prefill 成本不同、吞吐数字不可比。
- **`token usage: 0.05`**：KV 仓库（阶段 ③ 的 max_total_num_tokens）当前用了几成。**retract 的预警器**（3.3 节）。
- **`#running-req: 0` / `#queue-req: 4`**：打这行时在跑/在排队的请求数。

> **能做什么分析**：prefill 侧三问——命中率多高（省了多少 prefill）、一口咬多大（#new-token vs `--chunked-prefill-size`）、排队有没有积压（#queue-req）。**TTFT（Time To First Token，用户等到第一个字的时间）的账几乎全在 prefill 侧**，症状"首字慢"就从这行查起（第 5 节症状 3）。

### 3.2 Decode batch 行逐字段拆（最重要的一行）

```text
[2026-11-12 08:30:16 TP0] Decode batch. #running-req: 16, #token: 8192,
      token usage: 0.10, cuda graph: True, gen throughput (token/s): 812.40, #queue-req: 2
```

- **`#running-req: 16`**：当前 batch 里**正在 decode** 的请求数——拼车上现坐几个人。
  - **关联参数 `--max-running-requests`**：硬上限（回显里 `None` 时用大默认值）。实验 A 收到 16，batch 涨落才看得清。
  - **怎么分析**：**贴着上限不动** = batch 卡死（想更大吞吐得调参数）；**远低于上限又上不去** = 要么客户端压力不够，要么 token usage 高到准入失败（看下一字段）。
- **`#token: 8192`**：当前所有在跑请求的 KV token 总量（含 prompt + 已生成）。
- **`token usage: 0.10`**：KV 仓库用了几成。**和 prefill 行同源、更该盯的字段**——逼近 100% 就是 retract 前夜；它也是并发天花板的实时读数（usage 低 + running 低 = 机器很闲；usage 高 + queue 高 = KV 是瓶颈）。
- **`cuda graph: True`**：这一步 decode 走的是阶段 ④ 预录的"录像带"回放。**W1 的概念第一次在真机日志里现身**。变 False = 档位超出录制范围（batch > `--cuda-graph-max-bs`）或该形状没录到，退回慢路。
- **`gen throughput (token/s): 812.40`**：**窗口内平均 decode 产出**——整条传送带每秒送出多少个词。
  - **白话**：812 个词/秒由 16 条请求分，每条约 51 token/s——这正是单条 decode 速度（memory-bound，《三处空白》3.1 的 1:37 账）。
  - **怎么分析**：并发翻倍后这行**是否接近翻倍**？是 = 还在"加 batch 白赚吞吐"区间；停在原地 = 顶到某个上限，查 running/usage/queue 三个字段定位是谁顶的。**这是文章 3.1 节"1:37 拐点"的直接观测口**。
- **`#queue-req: 2`**：排队等上车的请求数。**持续增长 = 准入进不去**（十有八九是 KV——看 token usage）。

> **一行速记**：`gen throughput 看赚没赚，#running-req / #queue-req 看 batch 卡没卡，token usage 看 KV 挤不挤，cuda graph 看走的快路慢路`。

### 3.3 retract 行（实验 C 的主角；SGLang 版"抢占"）

```text
[2026-11-12 08:31:02 TP0] KV cache pool is full. Retract requests.
      #retracted_reqs: 1, #new_token_ratio: 0.9998 -> 1.0000
```

逐字段拆：

- **`KV cache pool is full`**：原因直说——KV 仓库满了（对照 3.2 的 `token usage` 应在高位）。
- **`Retract requests`**：**retract = SGLang 世界里的"抢占"**（vLLM 叫 preemption；机制同源：显存告急时把某个正在跑的请求"请下车"腾 KV）。SGLang 默认走 **recompute 路线**（丢 KV、之后重算 prefill；vLLM 世界的 swap 路线在 SGLang 里属于 HiCache 等进阶特性）。
- **`#retracted_reqs: 1`**：这次拉下车几条。拿请求号去对照客户端（02_probe_client）的 latency 输出——**每次 retract = 客户端侧一个 latency 尖刺**，两个证据源交叉验证。
- **`#new_token_ratio: 0.9998 -> 1.0000`**：SGLang 准入的**安全垫系数**（每来一条新请求，按它历史 token 增速预留 KV 余量；比例→1.0 表示"往死里留"）。**箭头是这次 retract 的自适应调整**——被挤了一次，调度器自动把安全垫加厚。这是 SGLang 调度里少有的**带运行时反馈**的参数（对照《三处空白》空白三说的"策略从不随统计变化"——SGLang 在这个单点上开了一个小口子，但也仅此而已：chunk 大小、准入水线、`--schedule-conservativeness` 仍是静态的）。

> **能做什么分析**：
> 1. **数频率**：`grep -c 'Retract' serve.C.log`。偶发 = 正常保险丝；频发滚动 = 配置性瓶颈（KV 太小 or 准入太松）。
> 2. **官方调优处方**（超参调优文档原话级建议）：偶尔看到 retract → 把 `--schedule-conservativeness` 加到 **1.3** 这类值（= 全局把安全垫垫厚，少 retract、多一点排队）。
> 3. **实验 C 的观察设计**：0.35 的盘子下压 16 条并发，对照 base 档数 retract 次数、看 `#new_token_ratio` 自适应的轨迹——W3 模拟器里的 watermark 在真机上就是这两个字段。

### 3.4 chunked prefill 的证据行（实验 B 的主角）

`--chunked-prefill-size 2048`（实验 B）= 一口最多咬 2048 token。长 prompt 被切成小块分步喂，证据就在**连续多条 Prefill batch 行**里：

```text
[...] Prefill batch. #new-seq: 1, #new-token: 2048, #cached-token: 0, ...
[...] Prefill batch. #new-seq: 1, #new-token: 2048, #cached-token: 0, ...
[...] Prefill batch. #new-seq: 1, #new-token: 1904, #cached-token: 0, ...
```

- 同一条 6000 token 的 prompt 出现 3 行 `#new-seq: 1`，`#new-token` 相加 = 2048+2048+1904 = 6000——**"长 prompt 被切成几块"就这样从日志里数出来**（观察点清单实验 B 第一项）。
- **prefill 块与 decode 的交错**：看时间顺序——一条 Prefill batch 行之后**不是清空 batch**，而是带着已有 running 继续 decode（`Decode batch` 行正常滚动）。这就是 chunked prefill 的意义：**长 prompt 的消化不再一整口吞下、把 decode 挤停，而是小口咬、decode 时延稳**。
- **怎么数块数**：prompt 长度 ÷ `--chunked-prefill-size` 向上取整 = 应看到的 Prefill 行数（同请求号）。对不上就是 chunked 没生效（回显里核对参数）。

---

## 4. 参数 ↔ 日志 对照总表（贴墙版）

每个参数一行：它改什么、**日志哪个字段会变**、调大调小的代价。参数以 01_serve_7b.sh 四档用到的为主。

| 参数（示例值） | 改的是什么 | 日志里的观测口 | 调整的代价/收益 |
|---|---|---|---|
| `--mem-fraction-static 0.90` | 静态显存盘子（权重+KV） | 阶段③ `max_total_num_tokens`、`avail mem`；运行时 `token usage` | 调大：KV 更多、座位更多；过头：capture/运行时 OOM（实验 C 反向用它逼 retract） |
| `--context-length 8192` | 单条请求最大长度 | 阶段① 回显；并发座位数 = max_total_num_tokens ÷ 它（反比） | 调大：能接长文档，座位数按比例缩水 |
| `--max-running-requests 256/16` | batch 并发上限 | `#running-req` 的天花板 | 调大：吞吐上限高（前提 KV 够）；调小：batch 涨落看得清（实验 A 的用途） |
| `--chunked-prefill-size 2048` | 每口 prefill 的 token 上限 | `#new-token` 的单行上限；长 prompt 的 Prefill 行数 | 调大：prefill 快、长 prompt 会挤 decode；调小：decode 时延稳、prefill 总耗时略增；**OOM 时官方第一调优项** |
| `--cuda-graph-max-bs N` | 录像带覆盖的 batch 档位 | 阶段④ capture 耗时与 `mem usage`；运行时 `cuda graph: True/False` | 调大：大 batch 也走快路，但 capture 慢、占显存；超范围 batch 退慢路 |
| `--schedule-conservativeness 1.0` | 准入安全垫（全局保守度） | retract 频率、`#new_token_ratio` | 调大（如 1.3）：少 retract、多排队；调小：激进准入、易 retract（官方文档的原话处方） |
| `--tp-size N` | 单模型切几张卡 | 进程标记 TP0…TP(N-1)、通信相关行 | 显存摊薄换通信税（all-reduce，文章空白二）；多卡分析先 `grep TP0` |
| `--quantization fp8/int8...` | 权重量化 | 阶段① 回显；权重占用 → `max_total_num_tokens` 变大 | 显存÷2 或更多、KV 座位变多（w10 A/B 的自变量；两档日志 diff 见第 6 节命令⑦） |
| `--log-requests`（默认关） | 逐请求日志行 | 开了才有请求级明细行 | 默认关正好（日志防淹没）；做单请求时延解剖时临时加 |

**用法**：做 A/B 前先决定"本次实验动哪一行的哪个参数"，再把两个日志的对应观测口字段抄成对照表——**参数差异 → 日志差异 → 指标差异**三段对得上，实验才算闭环。

---

## 5. 五个常见症状的诊断流程

每个症状 = 特征日志 + 推理链 + 动作。**推理链比结论重要**——症状在不同机器上的原因可能不同，链条是可迁移的。

### 症状 1：吞吐上不去（`gen throughput` 低位平躺）

```
推理链：gen throughput 低 → #running-req 是多少？
  ├─ 贴着 max-running-requests → 上限卡死：调大之（KV 够的前提下）
  ├─ 低 + #queue-req=0 → 压力不够：查客户端并发（02_probe_client 的参数）
  └─ 低 + #queue-req 高 + token usage 高 → KV 卡死：升 --mem-fraction-static
```

### 症状 2：请求越排越长（`#queue-req` 持续涨）

```
推理链：#queue-req 涨 → 准入进不去 → token usage 高位？
  ├─ 是 → 已在 retract 或即将（grep Retract 确认）→ 四选一：
  │      ① 加盘子（--mem-fraction-static）
  │      ② 降座位单价（--context-length 降到实际需要的长度）
  │      ③ 垫厚安全垫（--schedule-conservativeness 1.3——官方处方：多排队少 retract）
  │      ④ 限并发（--max-running-requests 收紧，让排队显性化）
  └─ 否 → #running-req 也低？那是客户端压力不足，不是服务端问题
```

### 症状 3：首字慢（TTFT 长，用户等第一个字久）

```
推理链：TTFT = 排队 + prefill。
  ① #queue-req 高 → 排队主导 → 同症状 2
  ② #cached-token 长期为 0 但前缀其实重复 → radix cache 没命中：
     检查 prompt 是否真的同前缀（w10/02 固定 PROMPT 的设计意图）
  ③ prompt 长且一口咬太大（#new-token 单行极大）→ 调小 --chunked-prefill-size
     让 decode 不被挤停（注意：这不改善 TTFT 本身，改善的是并发下别人的 TTFT）
```

### 症状 4：`#cached-token` 忽高忽低，压测数字不稳

```
推理链：命中率直接改写 prefill 成本 → 两轮 A/B 的命中率必须同口径。
  动作：两轮 prompt 同源（同 PROMPT / 同 few-shot 模板）；报告吞吐与 TTFT 时
        必须同时报命中率（#cached-token 占比），否则数字不可比。
```

### 症状 5：启动直接失败（ERROR/OOM）

```
推理链：读 ERROR 行的 Tried to allocate X vs free Y →
  ① 阶段① 回显里模型/dtype/量化对吗？（权重比预期大）
  ② 盘子（mem-fraction × 总显存）− 权重 < 0？（实验 C 收太狠）
  ③ capture 阶段 OOM？→ 阶段③ 的 avail mem 留太薄，--mem-fraction-static 回一点
  ④ 别的进程占着显存？（nvidia-smi 看，日志外证据）
  官方调优顺序：--chunked-prefill-size ↓ → --mem-fraction-static ↓ → --max-running-requests ↓
```

### 附：症状 6：启动好像卡死了（其实没有）

```
特征：卡在 "Capture cuda graph begin. This can take up to several minutes."
读法：官方预防针都写在行文里了——several minutes 是正常值（档位多、模型大就更久）。
      超过 10 分钟才值得 tail 日志查别的。验收：等 "Capture cuda graph end" 出现，
      抄下 Time elapsed 和 mem usage 进笔记。
```

---

## 6. 接线本仓库实验（命令直接抄）

配合 `w04_serving_lab` 的四档实验（`serve.base.log` / `serve.A.log` / `serve.B.log` / `serve.C.log`）：

```bash
# ① 启动账目一行捞齐（各阶段数字地基，抄进 notes_W4.md）
grep -E 'Load weight end|max_total_num_tokens|Capture cuda graph end|fired up' serve.base.log

# ② 实验 A：提全部 decode 统计行看 batch 涨落（贴给观察点表「batch 随时间涨落」）
grep 'Decode batch' serve.A.log

# ③ 实验 A 进阶：只提 #running-req 数字（输出可直接画曲线）
grep 'Decode batch' serve.A.log | awk -F'#running-req: ' '{split($2,a,","); print a[1]}'

# ④ 实验 B：数 chunked prefill 的块数（#new-token 相加 ≈ prompt 长度）
grep 'Prefill batch' serve.B.log | head -40

# ⑤ 实验 C：retract 事件全记录（几次、安全垫怎么自适应）
grep -iE 'retract' serve.C.log

# ⑥ 实验 C 前后对比：KV 总容量变化（0.90 vs 0.35 的账本差）
grep -h 'max_total_num_tokens' serve.base.log serve.C.log

# ⑦ A/B（w10）：两档配置回显 diff——确认真的只差量化（自变量控制）
diff <(grep server_args serve.bf16.log) <(grep server_args serve.int8.log)
```

**时间戳对账技巧**：日志时间和客户端（02_probe_client 的输出）时间对齐后，**同一个事件两边印证**——客户端看到的 latency 尖刺 ↔ 日志里的 Retract 行；客户端 burst 发车 ↔ 日志 #running-req 从 0 冲到 16。单一证据源会骗人，两个交叉就不会。

---

## 7. 版本差异、诚实口径与 vLLM 对照

1. **样例非实测**：本手册的日志样例按 SGLang 2025 年版的典型行文编写（依据官方文档与源码中的 logging 调用），数字为 H20 96GB + Qwen2.5-7B 的演示口径——**S2 现场以实际日志为准回填**，格式对不上时按"字段语义"对号。
2. **SGLang ↔ vLLM 翻译表**（W3 模拟器与《三处空白》讲的是机制，两家同源；读日志认字段时用这张表对上）：

| 概念 | SGLang（本手册） | vLLM（对照） |
|---|---|---|
| 显存盘子参数 | `--mem-fraction-static` | `--gpu-memory-utilization` |
| KV 总容量行 | `max_total_num_tokens=...` | `GPU KV cache size: ...` / `# CUDA blocks` |
| 并发上限 | `--max-running-requests` | `--max-num-seqs` |
| chunk 每口上限 | `--chunked-prefill-size` | `--max-num-batched-tokens` |
| 抢占 | **retract**（`KV cache pool is full...`） | preemption（`is preempted by PreemptionMode...`） |
| 前缀缓存 | **radix cache 默认开**（`#cached-token` 观测） | prefix caching 可选开关 |
| 加速执行 | CUDA graph 默认录（`cuda graph: True/False`） | V1 默认录、字段少直接暴露 |
| 就绪信号 | `The server is fired up and ready to roll!` | `Application startup complete.` |

3. **日志开关**：逐请求明细日志默认关（要解剖单请求时延就加 `--log-requests`）；统计行（Prefill/Decode batch）默认打、无需开关。个别版本的统计频率与措辞有差异——现场 `python -m sglang.launch_server --help | grep -iE 'log|stats'` 确认，这正是 01 脚本头部"版本敏感"警告的用意。
4. **下一步**（S2 会话后）：把实际日志的逐行摘录回填到本手册各节，形成你自己机器的对照手册——手册的价值随回填次数递增。
