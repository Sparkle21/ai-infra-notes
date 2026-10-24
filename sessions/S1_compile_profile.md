# S1 · torch.compile 前后对比 + 存镜像（W1 末，3h）

> 规格：`gn8v.4xlarge`（1×H20 96G，sm_90）· 预算 3h · 约 ¥109
> 对应学习代码：[../w01_torch_compile/](../w01_torch_compile/)

## 前置（Mac 上已完成）

- [ ] `w01/01`–`03` 在 CPU 上跑通，现象已记录（graph break / guard recompile 各一例）
- [ ] `w01/04_compile_vs_eager.py` 的 eager 路径数值已验证，`--mode compiled` 留到本会话
- [ ] 代码已 push 到私有仓库

## 开场 15 分钟（一次到位）

1. 用镜像/conda 起 env → `git pull`。
2. **验证 ncu 可用**：`ncu --version`，然后拿 `w01/02` 的模型小跑一次 `ncu --set full`（S5 才重度用 ncu，这里只确认不是 `<not supported>`）。
3. 装好的环境**存自定义镜像**（platform 侧操作）——本会话最重要的工程产出之一。

## 清单（只做这些）

1. **eager 基线**：`python 04_compile_vs_eager.py --mode eager --device cuda`
   - 记录 per-op 时间；ncu 不用，先拿 wall clock。
2. **compiled 对比**：`--mode compiled`
   - 记录首次编译耗时（冷）与稳态耗时（热）；对比 eager 提速比。
3. **nsys 抓两份时间线**：
   ```bash
   nsys profile -o s1_eager    python 04_compile_vs_eager.py --mode eager    --device cuda
   nsys profile -o s1_compiled python 04_compile_vs_eager.py --mode compiled --device cuda
   ```
4. **数 kernel launch**（`.nsys-rep` 拉回 Mac 用 Nsight Systems GUI 看，现场只做粗数）：
   - eager 一轮 forward 的 kernel 数 vs compiled 的 kernel 数；
   - 这就是「融合」在时间线上的直接证据——目标现象：compiled 的小 kernel 大片合并。
5. 顺手记录：H20 上 `torch.cuda.get_device_properties()` 的 SM 数 / 显存带宽（写进观察记录，W6 roofline 论证要用）。

## 产物回收到 Mac

- `s1_eager.nsys-rep`、`s1_compiled.nsys-rep`
- 两份 stdout（含计时数字）
- 自定义镜像 ID（记到本文件底部）

## 回填位置

- 数字 → `w01/04_compile_vs_eager.py` 尾部「观察记录」
- 时间线结论 → `w01/notes_W1.md`（Inductor 融合决策评审草稿的证据段）

## 现场止损

- compile 失败/报错 → 现场不修，只留 `TORCH_LOGS=graph_breaks` 的输出回 Mac 分析。

---
## 会话记录（回填）

- 镜像 ID：
- eager vs compiled 提速比：
- kernel launch 数（eager / compiled）：
