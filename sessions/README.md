# 租卡会话清单（S1–S5 + 可选 S2.5）

> 计划来源：plan.md 第三节（共 6 场，约 19 GPU 小时，单卡合计 ≈¥690，含 S2.5 ≈¥833）。
> 每场会话一页清单：**只做清单上的事**，做完立刻释放实例。

| 会话 | 时间 | 规格 · 时长 | 只做这些事 | 清单 |
|---|---|---|---|---|
| S1 | W1 末（10/25 前后） | gn8v.4xlarge · 3h | torch.compile 前后对比 + nsys 数 launch + 存镜像 | [S1_compile_profile.md](S1_compile_profile.md) |
| S2 | W4 末（11/15 前后） | gn8v.4xlarge · 4h | serve 7B、打 log 观察调度、nsys 抓 decode | [S2_sglang_serve.md](S2_sglang_serve.md) |
| S2.5（可选） | W4 末 | gn8v-2x.8xlarge 2×96G · 2h | 多卡 NCCL all-reduce 的 kernel 形状 | [S2.5_nccl_tp.md](S2.5_nccl_tp.md) |
| S3 | W6 末（11/29 前后） | gn8v.4xlarge · 3h | fused RMSNorm benchmark + autotune + ncu | [S3_rmsnorm_bench.md](S3_rmsnorm_bench.md) |
| S4 | W8 末（12/13 前后） | gn8v.4xlarge · 5h | matmul 推到 Tensor Core + FA forward + ncu 读 stall | [S4_matmul_fa.md](S4_matmul_fa.md) |
| S5 | W10 中（12/20 前后） | gn8v.4xlarge · 4h | INT8 W8A8 实跑 + SGLang 量化 A/B + 量化 kernel SOL | [S5_quant_ab.md](S5_quant_ab.md) |

## 每场会话通用纪律（plan.md 第六节）

1. 开机 → 用 S1 存好的自定义镜像（或 conda env）→ `git pull` 拉最新学习代码。
2. **第一件事验证 ncu 可用**：`ncu --version` + 一个最小 kernel 跑 `ncu --set full`，确认指标不是 `<not supported>`。
3. 计时区间内绝不 debug：所有代码上卡前已在 Mac 上验证过数值（interpret 铁律）。
4. 产物当次 `scp` 回 Mac：`.nsys-rep` / `.ncu-rep` / benchmark 输出 / 服务日志。
5. **用完立刻释放实例**（不是「节省模式停机」——它释放配额，下次可能起不来）。
6. 模型权重放数据盘，不放系统盘（系统盘随实例销毁）。

## 通用止损（现场不要恋战）

- ncu 指标 `<not supported>` → 记录现象，降级用 nsys 时间线；现场排障不超过 20 分钟。
- 环境装不上 → 保留报错日志，降级为「只跑已装好的部分」，剩下的写进下次会话清单。
- 被 spot 回收 → 不心疼，清单在手随时重来（这就是会话制的设计意图）。
