# notes_W5 · Triton 心智模型笔记

> 本周里程碑：数值正确的 fused RMSNorm（03 的三个形状全 ✓ + 边界两例 ✓）。
> 这页笔记的目标：三个月后不看教程还能写出行级 reduction kernel。

## 一、三行心智模型

1. kernel = 单个 program 的视角；grid = program 的编队方式。
2. 数据进出只有 `tl.load/tl.store`，block 内共享只有 reduction 类操作（tl.sum/max）与广播。
3. mask 不是可选项——任何「形状参数 × BLOCK」的地方都要问一句：越界了吗？

## 二、API 速查（本周实际用到的）

| API | 一句话 |
|---|---|
| `tl.program_id(axis)` | 我是谁（第 axis 维的第几个 program） |
| `tl.arange(0, BLOCK)` | 本 program 内的下标序列（必须 2 的幂） |
| `tl.load/store(ptr+offs, mask=, other=)` | 整块读写；other = 被屏蔽位置的填充值 |
| `tl.sum(x, axis=0)` / `tl.max` | block 内归约，返回标量 |
| `triton.cdiv` / `next_power_of_2` | host 侧的网格与 BLOCK 工具 |
| `tl.constexpr` | 编译期常量（BLOCK 等）——特化的钥匙 |
| 标量 × block 广播 | `x * rms`：rms 是标量，自动广播（SM 内寄存器操作，免费） |

## 三、两个「数值正确性」设计决策（面试级）

- softmax 的 other=-inf：被屏蔽处经 exp 归零，**不污染 sum 分母**。
- RMSNorm 的 other=0：被屏蔽处平方后归零，**不污染均方分子**；分母用真实 n_cols。
- 共同点：mask 位置的值要「参与运算后恰好无害」，填什么取决于后面的算子。

## 四、坑清单（踩过就勾）

- [ ] `other` 填错导致分母污染
- [ ] 分母误用 BLOCK_SIZE 而非真实列数
- [ ] 忘 `TRITON_INTERPRET=1`，容器里直接跑报平台错误
- [ ] BLOCK 忘了 2 的幂约束
- [ ] stride 用错（非连续输入）——本文件 assert 了 contiguous，工业版要处理

## 五、给 W6/S3 的交接

- 三口径就位：eager（torch.nn.RMSNorm）/ decomposed（pow+mean+rsqrt+mul）/ fused（本文件）。
- S3 的形状清单：[16,4096] / [8192,4096] / [8192,11008]。
- 待验证命题：「fused 的收益随行宽变大而增大」——先在笔记里写下你的预测再上卡。
