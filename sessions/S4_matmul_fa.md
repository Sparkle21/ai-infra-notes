# S4 · matmul 推到 Tensor Core + FA forward 复现（W8 末，5h）

> 规格：`gn8v.4xlarge`（1×H20 96G）· 预算 5h · 约 ¥182（本周最长，任务分两段）
> 对应学习代码：[../w07_matmul/](../w07_matmul/) + [../w08_flash_attention/](../w08_flash_attention/)

## 前置（Mac 上已完成）

- [ ] `w07/01_naive`、`02_tiled` interpret 数值 PASS；`03_bf16` 的数值对比逻辑已写好
- [ ] `w08/01_fa2_forward.py` interpret 数值 PASS（小形状，如 `[1, 4, 128, 64]`）
- [ ] 「看不懂的地方」清单已攒了一批（回 Mac 后继续补）

## 段一：matmul（约 2.5h）

1. **数值快验**：GPU 上重跑 `w07/02`、`w07/03` 的 assert。
2. **BF16 性能阶梯**：`python 03_bf16_tensor_core.py --device cuda`
   - naive FP32 → tiled FP32（tl.dot）→ BF16（允许 Tensor Core）三档对比；
   - 记录每档 TFLOPS；目标：BF16 档 ≥ cuBLAS 的 70%（本周里程碑）。
   - H20 参考上限：BF16 ≈148 TFLOPS（plan.md 第二节），达不到就记录差距并分析原因（占优方向：tiling / 流水 / bank 冲突）。
3. **cuBLAS 参照**：同形状 `torch.matmul` 一组数字作分母。
4. （富余）改一个 tiling 参数看敏感性——为「tiling 参数不外推」积累第一手证据。

## 段二：FlashAttention forward（约 2.5h）

1. **数值快验**：GPU 上重跑 `w08/01` 的 assert（含与 `F.scaled_dot_product_attention` 对比，注意 causal 开关两端口径一致）。
2. **性能对比**：`python 02_fa_benchmark.py --device cuda`
   - 自己的 FA forward vs SDPA（PyTorch 官方后端）；形状含长序列（`[1, 8, 4096, 128]` 这类）。
3. **ncu 读 stall**：
   ```bash
   ncu --set full -o s4_fa python 02_fa_benchmark.py --kernel mine --device cuda
   ```
   - 重点读：warp stall 原因（long scoreboard？barrier？）与 `Memory Throughput`；
   - 对比 SDPA kernel 的指标——差距在哪一项，就是「看不懂清单」的具体化。
4. （富余）序列长度扫一档（1k→4k→16k），看标准注意力 O(N²) 显存 vs FA 的差别。

## 产物回收到 Mac

- benchmark stdout（matmul 阶梯 + FA 对比）
- `s4_fa.ncu-rep`（+ 若做了 matmul 的 ncu 也带回）

## 回填位置

- matmul 数字 → `w07/notes_W7.md`
- FA 数字与 stall 分析 → `w08/notes_W8.md`（里程碑：matmul ≥ cuBLAS 70% + 看不懂清单）

## 现场止损

- FA 数值 GPU 上对不上（interpret 对但 GPU 错）→ 大概率是 mask/边界条件；现场只记录现象与 `TRITON_DEBUG` 信息，回 Mac 修。
