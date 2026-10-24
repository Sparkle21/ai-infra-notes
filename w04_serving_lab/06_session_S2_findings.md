# W4-06 · S2/S2.5 会话发现底稿

> 会话结束当天填完。填完后的动作：把证据回填 `../w03_vllm_scheduler/notes_W3.md`（升级为正式图），
> 把打脸/印证的条目同步到 `../w02_inductor_src/notes_W2_naive_list.md`（若涉及调度侧幼稚条目）。

## 一、实验 A · continuous batching

- 预期 vs 实测（对照 04_observation_points.md 的表）：
- 最有价值的发现：

## 二、实验 B · chunked prefill

- 预期 vs 实测：
- 最有价值的发现：

## 三、实验 C · preemption

- 逼出来了吗？参数组合：
- 被抢者是谁（新/老）？与源码阅读结论一致吗：

## 四、nsys 时间线

- decode step 的 kernel 组成（放大一个 step 数出来的）：
- GPU 空闲间隙的位置与成因（猜测）：

## 五、S2.5（若做了）

- all-reduce 形态：
- 重叠情况：

## 六、给后续周的开销票

- [ ] 需要修的脚本 bug（下次会话前修好）：
- [ ] 值得加进幼稚清单的条目：
- [ ] W10 分享提纲可用素材：
