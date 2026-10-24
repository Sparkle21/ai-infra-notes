# W4-04 · S2 观察点清单（会话前填「预期」，会话后填「实测」）

> 这张表是 S2 会话的作战地图：**每个实验必须先写预期**，实测与预期不符的地方就是最有价值的发现。
>
> 本表只说「看什么」；每一行长什么样、逐字段怎么读，见 `../docs/sglang_log_analysis.md`（SGLang 日志逐行分析手册——含 grep 命令箱与诊断流程）。

## 实验 A · continuous batching

| 观察点 | 在哪看 | 我的预期（S2 前填） | 实测（S2 后填） |
|---|---|---|---|
| batch size 随时间涨落（16→收缩） | serve.log 的 Decode batch 行（`#running-req`） | | |
| 参差输出长度的完成顺序 | 02_probe_client burst 输出 × 日志完成行 | | |
| 短请求是否被长请求拖住 | burst 里 64-token 请求的 latency | | |
| GPU 利用率随 batch 的变化 | 03_scraper 的 gpusmi.tsv | | |

## 实验 B · chunked prefill

| 观察点 | 在哪看 | 预期 | 实测 |
|---|---|---|---|
| 长 prompt 被切成多块的证据 | Prefill batch 行的 `#new-token`（同请求号多行相加 = prompt 长） | | |
| prefill 块与 decode 的交错 | 日志时间顺序（Prefill batch 后接 Decode batch 继续滚动） | | |
| decode 请求的时延在长 prefill 期间是否被压住 | steady_load 期间打一条带计时的请求 | | |
| `--chunked-prefill-size` 的生效 | 启动参数回显行 | | |

## 实验 C · retract（SGLang 版「抢占」）

| 观察点 | 在哪看 | 预期 | 实测 |
|---|---|---|---|
| Retract 行出现（`KV cache pool is full` 字样） | `grep -i retract serve.c.log` | | |
| 被拉下的是谁（新 or 老） | Retract 行前后文 + `#retracted_reqs` | | |
| 安全垫自适应（`#new_token_ratio` 怎么变） | Retract 行尾的箭头两侧 | | |
| 被拉下请求的完成时间惩罚 | burst 客户端的 latency 尖刺 | | |
| KV 水位（`token usage`） | Decode batch 行 | | |

## nsys 段（decode 阶段 batch 动态）

| 观察点 | 在哪看 | 预期 | 实测 |
|---|---|---|---|
| kernel launch 密度随 batch 波动 | s2_decode.nsys-rep 时间线（Mac GUI 看） | | |
| GPU 空闲间隙出现在什么位置 | 同上（两个 step 之间的 gap） | | |
| 一个 decode step 的 kernel 组成 | 放大单个 step（RMSNorm/GEMM/采样…） | | |

## 顺带记录（非主场，10 分钟内完成）

- [ ] 启动日志：`max_total_num_tokens` → W3 的账本图（总容量除 context-length = 并发座位数）
- [ ] radix cache 命中率：Prefill 行 `#cached-token`（SGLang 默认开；压测固定 prompt 时应居高）→ 认识即可
- [ ] CUDA graph：`Capture cuda graph end` 的耗时与 mem usage（W1 录像带概念的真机现身）→ 认识即可

## 会话前自检（S2 出发前过一遍）

- [ ] 三个实验的「预期」列全部填完（空着预期去现场 = 白花 GPU 小时）
- [ ] 01_serve_7b.sh 的参数与你装的 SGLang 版本核对过
- [ ] 02_probe_client.py 三种模式都空跑过
- [ ] 剧本里想看的 5 个问题从 notes_W3.md 第四节转过来了吗
