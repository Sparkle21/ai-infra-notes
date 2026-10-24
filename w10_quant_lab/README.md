# W10 · 量化实测 + 收口（12/21–12/27）

> 计划：整理全部清单与笔记；S5 会话实跑 INT8。
> 里程碑：**技术分享提纲**（选题 = 幼稚清单升级版）→ [notes_W10.md](notes_W10.md)
> 环境：容器 interpret 验 kernel 数值；S5 会话拿 GPU 数字与端到端 A/B。

## 文件与分工

| 序 | 文件 | 干什么 | 验收 |
|---|---|---|---|
| 01 | [01_int8_w8a8.py](01_int8_w8a8.py) | INT8 W8A8 kernel：quant → int8 dot → dequant epilogue | 数值 PASS（容差自定并论证） |
| 02 | [02_ab_driver.py](02_ab_driver.py) | S5 的 A/B 会话驱动器：起服务→打负载→收数字 | 空跑（无服务时优雅退出） |
| — | [notes_W10.md](notes_W10.md) | 技术分享提纲（10 周收口） | 提纲完整、每页有证据编号 |

## 本周三件事

1. **kernel**（01）：W7 的 matmul 换个 dtype + epilogue 乘 scale——你已经全部学过，
   这周是把已有积木拼起来的检验。
2. **A/B**（02 + S5）：bf16 vs int8 的 serve 吞吐/时延/显存三口径。
3. **收口**：幼稚清单终审（已验证/已推翻/仍存疑三栏）→ 挑出分享选题 → 写提纲。

## 收口流程（S5 之后）

```
w02/notes_W2_naive_list.md（终审标记）
        │ 挑出 3–5 条被证据砸实的
        ▼
w09/notes_W9_methodology.md（方法论骨架）
        │ 融合、删弱证据、补 S5 数字
        ▼
w10/notes_W10.md（分享提纲：一页纸标题 + 每页证据编号）
```

## 提醒

- A/B 的显存口径：`nvidia-smi` 手记即可（精确 VRAM 统计可从 SGLang 日志的 `token usage` × `max_total_num_tokens` 推算，见 docs/sglang_log_analysis.md）——口径写清楚比工具花哨重要。
- 分享提纲的受众定调：懂系统的工程师（非 GPU 专家）——W9 笔记的写作训练直接复用。
