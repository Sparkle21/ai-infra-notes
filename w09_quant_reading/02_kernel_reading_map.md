# W9-02 · vLLM/SGLang 量化 kernel 地形图（只读）

> 仓库：`~/src/vllm`（W3 已 clone）；SGLang 可选 clone 到 `~/src/sglang`。
> 读法同 W2：先地图、后侦察、证据（文件:函数）填表。

## 一、vLLM 量化代码的地形

```
vllm/
├─ model_executor/layers/quantization/
│    ├─ __init__.py            ★ QuantizationConfig 注册表（哪些方案被支持，第一站）
│    ├─ fp8.py                 FP8 方案：参数打包、scale 管理、kernel 选择
│    ├─ gptq.py / awq.py       weight-only 方案（算法在离线侧，在线只剩 kernel 选择）
│    └─ ...（marlin/cutlass 等一堆后端名）
├─ model_executor/layers/
│    ├─ linear.py              ★ Linear 的统一入口：量化和普通权重走同一个接口
│    └─ sampler/...            （非本周目标）
└─ csrc/                       CUDA 源码（只看结构不求甚解）
     ├─ quantization/          ★ scale 乘法/向量 dot 的 CUDA 实现
     └─ ...
```

## 二、侦察清单（每条留下证据与一句话理解）

| # | 位置 | 看什么 | 证据 | 一句话 |
|---|---|---|---|---|
| 1 | `quantization/__init__.py` | 方法注册表：支持列表有多长 | | |
| 2 | `linear.py` | 量化 Linear 与普通 Linear 的接口统一在哪（apply 的分发点） | | |
| 3 | `fp8.py` | scale 的粒度（per-tensor? per-channel?）与存放格式 | | |
| 4 | `gptq.py` | weight-only 在线侧只剩什么（对比 FP8 在线侧有多少事） | | |
| 5 | `csrc/quantization/` | 找一个最短的 kernel：它的 epilogue 乘 scale 的位置 | | |
| 6 | 搜索 `input_scale` / `weight_scale` | dynamic 校准发生在哪一层的哪个函数 | | |
| 7 | SGLang（若 clone） | 它的量化层与 vLLM 的差异（结构对比，不求细节） | | |

## 三、带着三个问题读（方法论笔记的素材）

1. **kernel 选择**：fp8.py 里有没有「根据 shape/硬件挑 kernel」的分支？
   （找到 = 有 cost model 的雏形；没找到 = 又一条幼稚清单素材）
2. **scale 的生命周期**：谁算 scale、存在哪、什么时候进 kernel？（对照数据库：统计信息的采集-存储-消费链路）
3. **fallback 路径**：不支持某组合时降级到什么？（对照优化器的 fallback 计划）

## 四、看不懂清单（延续 W8 的机制，持续攒）

| # | 位置 | 卡点 | 猜测 |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

---
## 记录

- 本周阅读总时长（估）：
- 最意外的发现：
