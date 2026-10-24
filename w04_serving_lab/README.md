# W4 · 调度实测准备 + S2 会话（11/09–11/15）

> 计划：准备 serve 脚本与观察点；顺带扫 block manager / radix cache（SGLang 版前缀缓存，非主场，认识即可）。
> 里程碑：**一张 SGLang 调度数据流标注图**（notes_W3.md 升级为正式版）
> 环境：脚本在 Mac 上写好、逻辑空跑；真跑在 S2 会话的租卡机上。

## 文件与分工

| 序 | 文件 | 干什么 | 验收（Mac 上） |
|---|---|---|---|
| 01 | [01_serve_7b.sh](01_serve_7b.sh) | S2 会话的 serve 启动脚本（参数已按实验分组） | `bash -n` 语法过；参数表过一遍 |
| 02 | [02_probe_client.py](02_probe_client.py) | 探测客户端：构造三种负载（baseline / 并发变长 / steady load） | 对假端点空跑（预期连接失败=逻辑通） |
| 03 | [03_scraper.sh](03_scraper.sh) | 会话现场的数据抓取器（nvidia-smi + 日志尾巴 + 时间戳） | 本地 `bash -n` 过 |
| 04 | [04_observation_points.md](04_observation_points.md) | **观察点清单**：每个实验想看什么、在哪看 | S2 前填满「预期」列 |
| 05 | [05_nccl_tp_probe.py](05_nccl_tp_probe.py) | S2.5（可选）的 TP/all-reduce 探针 | CPU 上逻辑分支跑通 |
| 06 | [06_session_S2_findings.md](06_session_S2_findings.md) | S2 回来后的发现底稿 | 会话后填 |

## 本周 Mac 侧三件事

1. 把 01–03 脚本全部「空跑级」验证（不碰真服务也不报错）；
2. 把 W3 模拟器里想验证的问题转进 04 的实验表；
3. 顺带阅读（非主场，各 30 分钟封顶）：kv_cache_manager 的 block 分配、prefix caching 的 hash 表——
   只要求能在 notes_W3 的图上标出它们的位置。

## S2/S2.5 会话

见 [../sessions/S2_sglang_serve.md](../sessions/S2_sglang_serve.md)、[../sessions/S2.5_nccl_tp.md](../sessions/S2.5_nccl_tp.md)。
