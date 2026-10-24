# W9 · 量化薄壳：只读不跑（12/14–12/20）

> 计划：读 vLLM/SGLang 的 INT8/FP8 kernel 代码（只读不跑）；写笔记《查询优化器方法论能给推理调度补什么》。
> 里程碑：**《查询优化器方法论能给推理调度补什么》初稿** → [notes_W9_methodology.md](notes_W9_methodology.md)
> 环境：Mac 纯阅读 + 写作周（无代码验收，验收标准是笔记的完成度）。

## 文件与分工

| 序 | 文件 | 干什么 |
|---|---|---|
| 01 | [01_quant_primer.md](01_quant_primer.md) | 量化概念补课：W8A8、scale 的粒度、dequant 的位置（30 分钟读完） |
| 02 | [02_kernel_reading_map.md](02_kernel_reading_map.md) | vLLM/SGLang 量化 kernel 的地形图 + 侦察清单 |
| — | [notes_W9_methodology.md](notes_W9_methodology.md) | **本周里程碑：方法论笔记初稿** |

## 本周节奏建议

- 前半周：01 → 02（阅读），同时把 W8 看不懂清单里的量化相关条目迁移过来继续攻；
- 后半周：写方法论笔记——这是 10 周里唯一「纯写作」的里程碑，素材全部来自前八周：
  - W2 幼稚清单（Inductor 侧的「没有代价模型」）
  - W3 AQE 映射表 + S2 的调度观察（调度侧的「没有运行时自适应」）
  - W6–W8 的 roofline/账本（你亲手验证过的成本语言）
- 每天收尾 10 分钟：把当天读到的「写死的启发式」随手记进幼稚清单。

## 提醒

- 只读不跑：本周不装任何东西、不起任何服务。看不懂的代码写进 02 的清单，别恋战。
- 量化领域的术语噪声很大（W8A8/weight-only/FP8 E4M3/…）：以 01 primer 的框架归类，别被名词带跑。
