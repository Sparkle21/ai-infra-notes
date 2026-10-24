# S5 · INT8 W8A8 实跑 + SGLang 量化 A/B（W10 中，4h）

> 规格：`gn8v.4xlarge`（1×H20 96G）· 预算 4h · 约 ¥146
> 对应学习代码：[../w10_quant_lab/](../w10_quant_lab/)；阅读背景：[../w09_quant_reading/](../w09_quant_reading/)

## 前置（Mac 上已完成）

- [ ] `w10/01_int8_w8a8.py` interpret 数值 PASS（量化 scale 逻辑已对）
- [ ] `w10/02_ab_driver.py` 审过参数（bf16 / int8 两档的启动命令模板）
- [ ] `w09` 的 open questions 里挑出想在端到端验证的 2–3 条

## 清单（只做这些）

1. **数值快验**：GPU 上重跑 `01_int8_w8a8.py` assert。
2. **kernel 级 INT8**：`python 01_int8_w8a8.py --device cuda --bench`
   - INT8 matmul（tl.dot 的 int8 输入）vs BF16 同形状；
   - 记录加速比；INT8 理论算力是 BF16 的 2 倍——实测到多少？
3. **一个量化 kernel 的 SOL**（ncu）：
   ```bash
   ncu --set full -o s5_int8 python 01_int8_w8a8.py --device cuda --kernel-only
   ```
   - 读 SOL：INT8 kernel 是算力受限还是带宽受限？（结合 roofline 拐点 ≈37 FLOP/Byte 判断）
4. **SGLang 端到端 A/B**：`python 02_ab_driver.py --plan ab`
   - 同一 7B 模型：bf16 权重 serve 吞吐/时延 vs int8 量化 serve（现成量化模型或 `--quantization` 选项，以版本为准）；
   - 记录：吞吐差、时延差、显存占用差（`nvidia-smi` 手记）。
5. **收口动作**：把 10 周的「幼稚清单」翻一遍，挑出本周证据能支撑/推翻的条目，当场做标记——这是分享提纲（W10 里程碑）的素材筛选。

## 产物回收到 Mac

- `s5_int8.ncu-rep`、A/B 两组数字、显存快照
- （可选）SGLang 量化路径的启动日志，佐证 `w09` 阅读笔记里的 kernel 选择逻辑

## 回填位置

- 数字 → `w10/notes_W10.md`（技术分享提纲的证据池）

## 现场止损

- 现成 INT8 模型下载不动 → 用 `--load-format dummy` 只对比调度路径开销，注明证据等级。
- A/B 显不出差距 → 本身就是结论（记录清楚：量化收益在哪个形状/负载下才出现），不是失败。
