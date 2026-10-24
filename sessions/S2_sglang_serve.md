# S2 · SGLang serve 7B 观察调度（W4 末，4h）

> 规格：`gn8v.4xlarge`（1×H20 96G）· 预算 4h · 约 ¥146
> 对应学习代码：[../w04_serving_lab/](../w04_serving_lab/)（serve 脚本与探测客户端在 Mac 上已写好；逐行读日志配 [../docs/sglang_log_analysis.md](../docs/sglang_log_analysis.md)）

## 前置（Mac 上已完成）

- [ ] `w04/01_serve_7b.sh` 审过参数（`--max-running-requests` / `--mem-fraction-static` / `--chunked-prefill-size`）
- [ ] `w04/02_probe_client.py` 在本地对假服务空跑通过（脚本逻辑没错，只等真端点）
- [ ] 观察点清单（`w04/04_observation_points.md`）已列好每个实验想看到什么
- [ ] 权重路径确认：ModelScope 内网下载到数据盘（7B 十几分钟，可接受）

## 清单（只做这些）

1. **起服务**：`bash 01_serve_7b.sh`（后台 + 日志重定向 `serve.$EXP.log`）
   - 开机第一件事看启动日志：`max_total_num_tokens`（KV 账本总容量）、`Capture cuda graph`（SGLang 默认有这一步，几分钟属正常，别当卡死）。
2. **预热 + 基线探测**：`python 02_probe_client.py --endpoint http://127.0.0.1:8000 --mode baseline`
   - 单请求短/长输出各一条，确认链路通。
3. **观察调度行为**（核心，每个实验配 `04_observation_points.md` 里的预期）：
   - **实验 A · continuous batching**：并发 16 个变长请求，`Decode batch` 行看 `#running-req` 逐 step 增长与收缩。
   - **实验 B · chunked prefill**：长 prompt（~4k token）+ 并发 decode，`Prefill batch` 行的 `#new-token` 看切块与 decode 交错。
   - **实验 C · retract**：小 `--mem-fraction-static` + 长输出并发，逼出 `KV cache pool is full. Retract requests.` 行（含 `#new_token_ratio` 安全垫自适应）。
4. **nsys 抓 decode 阶段 batch 动态**：
   ```bash
   # 先 attach 模式或包一层启动；采 30–60s 足够
   nsys profile -o s2_decode --duration 60 python 02_probe_client.py --mode steady_load
   ```
   - 在时间线上找：launch 密度随 batch 变化的痕迹、GPU 空闲间隙。
5. 顺手记录：启动日志里的 `max_total_num_tokens`（回填 `w03` 的 KV 账本笔记；除以 context-length = 并发座位数）。

## 产物回收到 Mac

- `serve.*.log`（完整，含调度相关行——这是本周最值钱的数据）
- `s2_decode.nsys-rep`、各实验的响应时间/吞吐 stdout
- `04_observation_points.md` 的「实测」列当场补几笔，回 Mac 展开

## 回填位置

- 现象 → `w03/notes_W3.md`（调度数据流标注图的证据）与 `w04/06_session_S2_findings.md`

## 现场止损

- 模型下载慢 → 切 ModelScope 内网源；再不行降级为随机权重 + `--load-format dummy`（只看调度，不看质量，够用）。
- retract 逼不出来 → 记下参数组合，换更小 `--mem-fraction-static` 重试一次，仍不出现就记录「该版本默认行为」收工。
