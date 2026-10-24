# AI Infra 学习工作区（推理优化主线 · 10 周）

> 唯一上游计划：[docs/plan.md](docs/plan.md)（2026-10-19 → 2026-12-27）。
> 这里只放两样东西：**学习代码骨架**（注释先行，`TODO(you)` 留给你写）和**笔记底稿**（里程碑产物往里填）。

## 文章

- [DeepInfer 深度剖析：跟着一条请求走完 DeepSeek 推理引擎](docs/article_deepinfer.md)
- [投机解码深度剖析](docs/article_speculative_decoding.md)
- [LLM 推理调度 vs 数据库查询优化器](docs/article_llm_inference_vs_query_optimizer.md)
- [SGLang 服务日志逐行分析手册](docs/sglang_log_analysis.md)

## 怎么用这套目录（每周循环）

1. **读**：打开当周 `wNN_*/README.md`，知道这周学什么、验收什么、在哪个环境跑。
2. **写**：按 `01_ 02_ 03_` 编号顺序点开文件。先读文件头部的「目标 / 运行方式 / 验收」，再把 `TODO(you)` 补成代码。
3. **验**：跑文件自带的验证 harness（对照 torch 参考实现 `assert_close`），数值对了才算过。
4. **记**：把看到的数字/现象回填到文件尾部的「观察记录」区；周末汇总进当周里程碑文件。
5. **租卡周**（W1/W4/W6/W8/W10）：会话前过一遍 `sessions/SN_*.md` 清单，把攒的问题集中测完，产物 `scp` 回来回填笔记。

**铁律（plan.md 第五节）**：interpret 保证上卡时数值已经是对的；租来的小时只用来测性能，绝不在计时里 debug Python 报错。

## 环境矩阵（跑代码前先确认你在哪）

| 环境 | 覆盖范围 | 入口 |
|---|---|---|
| Mac 本机（CPU torch） | W1/W2 的 torch.compile 实验、W3 调度模拟器 | `python check_env.py` |
| Mac OrbStack（Ubuntu 容器 + triton，`TRITON_INTERPRET=1`） | W5–W8 全部 kernel 的**数值**验证 | `w05_triton_rmsnorm/README.md` |
| 租卡实例（gn8v.* / H20 / sm_90） | 一切真实 benchmark / nsys / ncu | `sessions/` |

## 目录地图

```
ai_infra/
├── check_env.py              # 环境自检：先跑我
├── docs/plan.md              # 10 周总计划（唯一事实来源）
├── sessions/                 # 租卡会话清单 S1–S5（+ 可选 S2.5）
├── w01_torch_compile/        # torch.compile 全景：四层栈 + graph break / guard 两个坑
├── w02_inductor_src/         # Inductor 源码：fusion pass / pattern matching → 幼稚清单
├── w03_vllm_scheduler/       # vLLM 调度器阅读 + 纯 Python 调度模拟器
├── w04_serving_lab/          # S2 会话准备：serve 脚本 / 探测客户端 / 观察点 / TP 通信
├── w05_triton_rmsnorm/       # Triton 入门：vector_add → fused softmax → fused RMSNorm
├── w06_fusion_validation/    # 融合验证：benchmark 方法论 + 「省 HBM 往返」论证
├── w07_matmul/               # naive → tiled(tl.dot) → BF16 Tensor Core
├── w08_flash_attention/      # FA2 forward 复现骨架 + 看不懂清单
├── w09_quant_reading/        # INT8/FP8 kernel 只读 + 方法论笔记
└── w10_quant_lab/            # INT8 W8A8 实跑 + vLLM A/B + 分享提纲
```

## 进度表（完成一项勾一项）

| 周 | 代码验收 | 里程碑产物 | 租卡 | 状态 |
|---|---|---|---|---|
| W1 | `w01` 01–04 全跑通并回填观察 | 融合决策评审草稿 | S1 | ☐ |
| W2 | `w02` source_map + 3 个 drill | **幼稚清单 v1** | — | ☐ |
| W3 | `w03` 模拟器三个策略函数写完 | 调度数据流标注图（草稿） | — | ☐ |
| W4 | `w04` 脚本备好、观察点过一遍 | **调度数据流标注图**（正式） | S2（+S2.5） | ☐ |
| W5 | `w05` 01–03 数值全部 PASS | 数值正确的 fused RMSNorm | — | ☐ |
| W6 | `w06` benchmark 脚本在本地空跑无误 | **「省 HBM 往返」论证笔记** | S3 | ☐ |
| W7 | `w07` 01/02 数值 PASS | naive matmul 数值正确 | — | ☐ |
| W8 | `w08` FA forward 数值 PASS | matmul ≥ cuBLAS 70% + 看不懂清单 | S4 | ☐ |
| W9 | `w09` 阅读笔记填完 | **《查询优化器方法论能给推理调度补什么》** | — | ☐ |
| W10 | `w10` kernel 数值 PASS + A/B 跑完 | **技术分享提纲** | S5 | ☐ |

## 约定

- `TODO(you)` = 留给你写的代码；其余 harness（输入生成、参考实现、断言）已写好，改 harness 前先想清楚为什么。
- 每个 `.py` 尾部有「观察记录」区，跑完就填——笔记长在代码旁边，不另开摊子。
- 不引入重依赖：核心只依赖 `torch` / `triton`；`requests` 仅 W4 探测客户端需要。
- 版本敏感的 API（`torch._dynamo.*`、vLLM V0/V1 路径）在注释里都标了「以你装的版本为准，先小跑验证」。
