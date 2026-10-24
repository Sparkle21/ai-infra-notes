# S3 · fused RMSNorm benchmark + autotune + ncu（W6 末，3h）

> 规格：`gn8v.4xlarge`（1×H20 96G）· 预算 3h · 约 ¥109
> 对应学习代码：[../w05_triton_rmsnorm/](../w05_triton_rmsnorm/) + [../w06_fusion_validation/](../w06_fusion_validation/)

## 前置（Mac 上已完成）

- [ ] `w05/03_fused_rmsnorm.py` 在 interpret 模式数值 PASS（铁律：上卡前数值必须已对）
- [ ] `w06/01_bench_core.py` 在 CPU 上空跑通过（计时逻辑无误，只是数字无意义）
- [ ] 待测问题清单已列（`w06/README.md` 的实验表填好了「预期」列）

## 清单（只做这些）

1. **数值快验**（上卡第一件事，1 分钟）：GPU 上重跑一遍 assert，确认 interpret 与真实执行一致。
2. **核心 benchmark**：`python 01_bench_core.py --kernel rmsnorm --device cuda`
   - eager / decomposed（mul+pow+mean+sqrt+div）/ fused 三口径；
   - 形状扫：`[16, 4096]`（单头）、`[8*1024, 4096]`（大 batch·decode 友好）、`[8192, 11008]`（FFN 宽度）。
3. **autotune**：`python 02_autotune.py`
   - `num_warps ∈ {1,2,4,8}` × `BLOCK ∈ {…}`，看 Triton autotuner 的选择；记录 best config per shape。
   - 注意 H20 sm_90 低算力带宽比：**结论只对 H20 负责，笔记标注**（plan.md 第二节选型结论）。
4. **ncu 对比融合前后 memory throughput**：
   ```bash
   ncu --set full -o s3_fused python 01_bench_core.py --kernel rmsnorm --variant fused --device cuda
   ncu --set full -o s3_eager python 01_bench_core.py --kernel rmsnorm --variant eager  --device cuda
   ```
   - 重点读：`Memory Throughput %`、`DRAM Throughput`、L2 命中——**fused 是否更接近带宽上限**。
5. 顺手记录：S1 里拿到的 H20 带宽参数，代入 `w06/03_hbm_traffic.py` 核对流量估算。

## 产物回收到 Mac

- `s3_fused.ncu-rep`、`s3_eager.ncu-rep`、autotune 结果表、benchmark stdout

## 回填位置

- 数字 → `w06/notes_W6.md`（「省 HBM 往返」论证笔记——本周里程碑）
- ncu 结论 → 同文件「单 kernel 证据」段

## 现场止损

- ncu 拒绝采集（权限/驱动）→ 降级只拿 nsys + wall clock，论证笔记里注明证据等级。
