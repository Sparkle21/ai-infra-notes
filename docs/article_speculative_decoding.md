<!-- 投稿草稿，v2（v2 新增第七～九节：方法选型、多卡部署、集群部署，并扩充第十一节场景清单）。
     体例沿用《LLM 推理栈的三处空白》：文中论断分两类——
     (1) 公开论文/源码/issue 可查的事实，标注出处；(2) 作者的分析判断与粗算，标注「判断」或「粗算」。
     关键术语首现处补一句白话。作者本人尚未上卡实测，全文不出现作者"已实测"数据；
     引用的实测数字均来自论文作者、框架团队或 issue 作者。 -->

# 投机解码深度剖析：拿闲置算力换延迟的那笔账

> **摘要**：投机解码（speculative decoding）是近三年推理加速里少见的"白捡"型技术：输出分布与原模型严格一致，单流延迟却能降到原来的 1/2～1/5。但它不是免费的——它拿的是 GPU 在 decode 阶段**闲着的算力**，一旦闲置算力被大 batch 吃掉，它就从加速器变成减速器。本文沿"为什么能加速 → 为什么不失真 → 能加速多少 → 草稿从哪来 → 进了 serving 引擎会遇到什么 → 全部方法怎么选 → 上多卡、上集群又会遇到什么"的主线拆开讲，最后给出一份"哪些场景没问题、哪些场景有问题"的判断清单。

## 一、问题：decode 慢的不是"算"，是"搬"

先翻译几个后文反复出现的词：**target 模型**（目标模型）= 你真正想用的那个大模型，输出质量以它为准；**draft**（草稿）= 用某种便宜办法先猜出来的几个后续 token；**verify**（验证）= 让 target 模型一次性检查这串草稿对不对；**memory-bound**（访存受限）= 时间主要花在从显存搬数据，而不是在计算单元上算。

LLM 每 decode 一步只产出 1 个 token，但这一步必须把**全部权重从显存（HBM）读一遍**。拿 Llama-3.1-8B（bf16，约 16 GB 权重）在 H100（显存带宽 3.35 TB/s，纸面值）上粗算：

```
单步下限 ≈ 16 GB ÷ 3.35 TB/s ≈ 4.8 ms   →   bs=1 时理论上限 ≈ 208 token/s
```

SGLang 团队在 EAGLE-3 论文里报告的同配置无投机基线是 **158 token/s**（论文 Table 4），约为这个上限的 76%——说明这一步确实被"搬权重"卡住了（粗算）。

关键在于：搬一次权重，**计算单元能顺手算多少个 token 几乎不加钱**。矩阵乘法的"算术强度"（每搬 1 字节数据能做多少次运算）近似等于这一步同时处理的 token 数 M（bf16 权重下，2·M·N·K 次运算 ÷ 2·N·K 字节 = M）。H100 的 bf16 稠密算力约 989 TFLOPS，除以带宽得到"拐点"约 **295 FLOP/字节**（纸面值）：一步里处理的 token 数远小于 295 时，多算几个 token 的时间几乎被搬权重的时间完全掩盖。

打个比方：主编审稿，主要时间花在"去档案室把整套资料搬出来"；资料搬出来之后，审 1 句和审 5 句差不多快。**投机解码就是先让实习生起草 5 句，主编搬一次资料、一口气审 5 句。**

## 二、算法核心：起草—验证，以及为什么"不失真"

### 2.1 一轮的三个动作

```
1. 起草：用便宜的办法猜出 γ 个后续 token  x1, x2, …, xγ（γ 读作 gamma，即"草稿长度"）
2. 验证：target 模型做一次 forward，同时吃进 [已确认前缀 + x1…xγ]，
         得到每个位置自己的概率分布 p1, p2, …, pγ+1
3. 接受：从前往后逐个判定；遇到第一个被拒的位置，用 target 的分布重新采一个 token 补上，
         其后草稿全部作废；若 γ 个全被接受，再白送 1 个 token（从 pγ+1 采样，俗称 bonus token）
```

所以每一轮**至少产出 1 个 token**（最差情况等价于普通 decode），最多产出 γ+1 个。

### 2.2 贪心解码下：只看"撞没撞"

temperature=0（每步取概率最大的那个词）时规则最简单：草稿 token 等于 target 在该位置的 argmax 就接受，否则拒绝并用 target 的 argmax 替换。输出与不开投机时逐 token 相同（数学意义上；数值上见 6.6 节）。

### 2.3 采样解码下：修正版拒绝采样

真正的巧思在采样场景（Leviathan et al. 与 Chen et al. 2023 独立提出）。记某位置 target 分布为 p、草稿分布为 q，草稿从 q 里采出了 x：

```
以概率 min(1, p(x)/q(x)) 接受 x；
若拒绝，从"残差分布" r(x) = max(0, p(x) − q(x)) / Σ_y max(0, p(y) − q(y)) 中重采一个
```

白话：target 比草稿更看好 x，就照单全收；草稿"过度自信"了，就按比例打折接受；被拒之后，只从"target 想要、但草稿给少了"的那部分概率里补采。

**为什么最终输出严格服从 p**——证明只有三行：

```
P(输出 x) = q(x)·min(1, p(x)/q(x))  +  P(拒绝)·r(x)
          = min(p(x), q(x))          +  (1 − Σ_y min(p,q)) · max(0, p(x)−q(x)) / (1 − Σ_y min(p,q))
          = min(p(x), q(x))          +  p(x) − min(p(x), q(x))
          = p(x)                                                                         ∎
```

（第二行用到 Σ_y max(0, p−q) = Σ_y (p − min(p,q)) = 1 − Σ_y min(p,q)。）

这条证明的含义值得停一下：**草稿质量只影响速度，不影响结果**。草稿再烂，最坏也就是每轮只产 1 个 token；草稿再好，也不会让输出"偏向草稿"。这是投机解码和量化、剪枝等"有损加速"最本质的区别。

顺带得到一个重要量——单个位置的**接受率**：

```
α = Σ_x min(p(x), q(x)) = 1 − TV(p, q)
```

TV 是两个分布的总变差距离。白话：α 就是"草稿和正式答案撞车的概率"，草稿越像 target，α 越接近 1。

## 三、收益账：能快多少，γ 取多大

### 3.1 期望产出

假设每个位置的接受率都是 α 且相互独立（简化假设，真实情况见下文），一轮期望产出的 token 数为：

```
E[每轮产出] = 1 + α + α² + … + α^γ = (1 − α^(γ+1)) / (1 − α)
```

记起草一步的耗时为 target 单步的 c 倍，并假设验证 γ+1 个 token 的耗时 ≈ target 单步（第一节的结论），则相对普通 decode 的加速比为：

```
加速比 S = E[每轮产出] / (γ·c + 1)
```

### 3.2 代几个数（粗算）

| α | c | γ=2 | γ=4 | γ=8 | 最优 γ 及加速比 |
|---|---|---|---|---|---|
| 0.6 | 0.05 | 1.78x | 1.92x | 1.77x | γ=4，1.92x |
| 0.6 | 0.20 | 1.40x | 1.28x | 0.95x | γ=2，1.40x |
| 0.8 | 0.05 | 2.22x | 2.80x | 3.09x | γ=8，3.09x |
| 0.8 | 0.20 | 1.74x | 1.87x | 1.66x | γ=4，1.87x |
| 0.9 | 0.05 | 2.46x | 3.41x | 4.38x | γ=13，4.67x |
| 0.9 | 0.20 | 1.94x | 2.28x | 2.36x | γ=7，2.37x |

三个读法：

1. **α 决定天花板**。E[每轮产出] 的上限是 1/(1−α)：α=0.6 时无论 γ 多长，每轮最多也就 2.5 个 token。
2. **c 决定 γ 能拉多长**。起草越便宜，越值得多猜几步；起草贵（c=0.2）时 γ=8 甚至跌破 1x。
3. **γ 有最优值，不是越长越好**。尾部 token 被接受的概率是 α^i 指数衰减，而起草成本线性增长。

还有一个工程上常被混淆的点：论文和日志里常报的"**平均接受长度 τ**"（每轮实际产出的 token 数，含 bonus）**不等于加速比**。τ 只是分子；分母里的起草成本、验证开销、调度开销都没算进去。第六节会给一个 τ=3.1 却比不开还慢的真实案例。

## 四、草稿从哪来：一张族谱

所有投机解码方法的"验证"部分几乎一样，差别全在"怎么便宜地猜"。按草稿来源分五类：

### 4.1 独立小模型（经典 SpS）

用同家族的小模型（如 Vicuna-68M 给 Vicuna-13B 打草稿）自回归地猜 γ 步。最直观，但有三个硬伤：必须**共享词表**；小模型和大模型"想法"差距大，α 偏低；草稿模型要**单独维护一份 KV cache**（模型逐层存下的上下文记忆），占显存、增调度复杂度。

### 4.2 无模型：从已有文本里"抄"

- **Prompt Lookup Decoding（PLD）/ n-gram**：在 prompt 和已生成内容里找与最近几个 token 相同的片段，把它后面的内容直接当草稿。零训练、零额外显存。对"改写、摘要、RAG 引用原文、代码修改"这类输出大量复制输入的任务效果极好，对开放式闲聊几乎无效。SGLang 中对应 `NGRAM` 算法。
- **Lookahead Decoding**：用 Jacobi 迭代（把"逐个猜"改成"一次猜一整段、反复迭代修正"）在运行中自产 n-gram 池，再从池里取候选验证。

### 4.3 多头：在 target 上长几个"预言头"——Medusa

在 target 最后一层隐状态上加 K 个小预测头，第 k 个头直接预测"往后第 k+1 个 token"。不需要独立模型，草稿几乎零成本。Medusa-1（冻结主干只训头）约 2.2x，Medusa-2（与主干联合微调）约 2.3–3.6x（论文报告）。Medusa 还提出了 **tree attention**（第五节展开）。注意它的 typical acceptance 接受规则在采样场景下**放宽了**严格的拒绝采样，因此不再保证无损——EAGLE-3 论文在 temperature=1 的对比里专门把这类方法排除在外。

### 4.4 特征级草稿：EAGLE 一系（当前工业主流）

EAGLE 的核心观察：**预测 target 的"倒数第二层特征"比直接预测 token 更容易**，因为特征序列比 token 序列更平滑。它的草稿模型只有 1 层 decoder，输入是"token embedding + target 的顶层隐状态"拼接后过一个全连接层。

- **EAGLE-2**：草稿结构不变，把静态草稿树换成**动态树**——用草稿模型自己的置信度近似接受率，每步只扩展置信度最高的 top-k 个分支，最后再整体重排、剪枝。思路是"难的上下文多分叉对冲，容易的上下文少浪费"。
- **EAGLE-3**（2025）做了两处改动（论文消融实验，Llama-3.1-8B，MT-bench）：

| 版本 | 加速比 | τ |
|---|---|---|
| EAGLE-2 | 3.16x | 4.05 |
| + 去掉"特征回归"约束，直接预测 token | 3.82x | 5.37 |
| + 融合低/中/高三层特征（EAGLE-3） | 4.40x | 6.13 |

第一处改动配套了 **training-time test（训练时测试）**：训练阶段就模拟推理时的多步自回归——草稿模型后几步的输入是它自己前几步的输出，而不是真值。这解决了 EAGLE 一代的"误差累积"：论文 Figure 7 显示，EAGLE 的接受率随"自产输入"步数增加明显下降，EAGLE-3 几乎持平。第二处改动从 target 的第 2 层、中间层、倒数第 3 层各取一份隐状态拼接（SGLang 实现中 `layers_to_capture = [2, num_layers // 2, num_layers - 3]`），弥补"只看顶层特征信息不够"。

一个容易忽略的工程细节：EAGLE-3 的草稿模型使用**自己训练的、缩小过的 LM head**（如 Qwen3-8B 的 EAGLE-3 草稿模型 draft 词表 32,000，target 词表 151,936，见 sglang issue #5274 中的模型信息），缩小词表直接降低每步起草的成本 c。

EAGLE-3 论文还报告了一个此前没观察到的现象：**草稿模型的加速比随训练数据量增加而持续上升**（scaling law）。这意味着草稿模型的质量从"架构问题"部分变成了"数据问题"——LMSYS 随后开源了专门训练 EAGLE-3 草稿模型的 SpecForge。

### 4.5 模型自带：DeepSeek-V3 的 MTP

DeepSeek-V3 在预训练时就加了 **MTP（Multi-Token Prediction）** 模块作为辅助训练目标，推理时可以直接拿来当草稿。技术报告给出：第二个 token 的接受率在 **85%–90%**，配合投机解码达到约 **1.8 倍 TPS**。拿第三节公式核对：γ=1、α≈0.875 时 E[每轮产出] = 1 + 0.875 ≈ 1.88，与报告的 1.8x 量级吻合（粗算）。"预训练阶段就为投机解码留接口"，是模型侧与系统侧协同设计的一个信号。

### 4.6 并行起草：DFlash（2026）

前面所有方法的起草仍是自回归的——猜 8 个 token 要跑 8 次草稿 forward，这迫使 EAGLE-3 把草稿模型压到 1 层以控制 c。DFlash（ICML 2026）换了思路：用一个轻量的 **block diffusion（块扩散）** 草稿模型，**一次 forward 并行起草一整块**（如 16 个 token），起草成本几乎不随草稿长度增长，于是可以用更深的草稿模型换更高的 α；同时把 target 多层融合特征注入草稿模型**每一层**的 KV，而不是像 EAGLE-3 那样只喂给第一层。项目页报告在 Qwen3-8B 上最高约 6x 无损加速，比 EAGLE-3 高近 2.5 倍，已有 SGLang 接入（`--speculative-algorithm DFLASH`）。

我的判断：DFlash 的意义不在某个具体数字，而在于它直接攻击了第三节公式里的 γ·c 项——当起草成本与 γ 脱钩，"γ 有最优值"这条约束就松动了。

并行起草有一个天生的毛病：块内各位置是**同时**预测的，彼此看不见对方选了什么，可能拼出不连贯的组合（vLLM 博客举的例子："of course"和"no problem"都合理，逐位置独立预测却可能拼出"of problem"）。2026 年的两个后继都在补这个洞：**DSpark** 在 DFlash 式并行主干后加一个轻量的 Markov 头，按前一个已选 token 给下一位置的 logits 加偏置；**DFlash 2** 在每个位置保留 top-16 候选，再用一个轻量选择器对相邻候选对打分、追出一条连贯路径。两者都保持"一次 forward 起草"的成本结构，只在其后加一点很便宜的顺序修正。它们与其他方法的横向归纳见第七节。

### 4.7 横向对照

EAGLE-3 论文 Table 1 给出了同一 target（Vicuna-13B，temperature=0，五个数据集均值）下各方法的数字，可作为族谱的定量对照：

| 方法 | 草稿来源 | 加速比 | τ |
|---|---|---|---|
| Lookahead | Jacobi 自产 n-gram | 1.62x | 1.67 |
| PLD | 从 prompt 抄 | 1.74x | 1.80 |
| SpS | 独立小模型（68M） | 1.92x | 2.24 |
| Medusa | 多预测头 | 2.12x | 2.51 |
| Hydra | 多预测头（头间有依赖） | 2.80x | 3.50 |
| EAGLE | 特征级 1 层草稿 | 3.05x | 3.96 |
| EAGLE-2 | + 动态树 | 4.22x | 4.83 |
| EAGLE-3 | + 去特征约束 + 多层特征 | 5.51x | 6.62 |

同一张表也显示**任务差异很大**：EAGLE-3 在 HumanEval（代码，模板多）上 τ 达 7.54，在 CNN/DM（摘要）上 PLD 反而能到 2.42x，超过它在其他任务上的表现——因为摘要大量复制原文。temperature=1 时所有方法的 τ 都下降（如 Llama-3.1-8B 上 EAGLE-3 的 τ 从 6.23 降到 4.92），因为采样时 target 的分布更"散"，草稿更难撞中。

## 五、为什么要"树"：一次验证多条路

链式草稿的浪费在于：第 2 个位置一旦被拒，后面猜的全部作废。树状草稿在不确定的位置**同时押几个候选**，验证时一次性检查所有分支，取最长的被接受路径。

```
             ┌─ "is" ─┬─ "a"
  "This" ────┤        └─ "the"
             └─ "was" ── "a"
```

实现上要解决的问题是：多个分支塞进**同一次** forward，但每个节点只能看到自己的祖先，不能看到兄弟分支。办法是 **tree attention mask**（树注意力掩码）——把树压平成一个序列，构造一个只在"祖先—后代"之间放行的注意力掩码；同一深度的节点共用同一个位置编号。

采样场景下，树验证要保持无损需要多候选版本的拒绝采样（依次对兄弟节点做拒绝采样、每次拒绝后更新残差分布），比链式复杂；贪心场景下只需沿树找与 target argmax 一致的最长路径。

在 SGLang 里，树的形状由三个参数决定（参数名见 SGLang 文档与源码走读）：

| 参数 | 含义 | 对应第三节的量 |
|---|---|---|
| `--speculative-num-steps` | 起草 forward 的次数，即树的深度 | γ（深度） |
| `--speculative-eagle-topk` | 每步每个节点扩展的分支数 | 树的宽度 |
| `--speculative-num-draft-tokens` | 重排剪枝后最终送去验证的节点数 | 验证的"批量" |

topk=1 时树退化为链。SGLang 文档建议：三个参数**要么全不设（走自动调参），要么三个一起显式设定**。三者之间的权衡是：深度和宽度越大，猜中的机会越多（τ 上升），但起草步数（c·γ）和验证批量（num-draft-tokens）都在涨——后者正是下一节"大 batch 下闲置算力消失"的关键变量。

## 六、进了 serving 引擎：系统层面的六个坑

论文里的加速比大多是 batch size=1 的单流延迟。进入真实 serving 引擎（同时服务很多请求的推理服务进程），情况复杂得多。

### 6.1 batch 一大，闲置算力就没了

第一节的前提是"一步处理的 token 数远小于拐点（H100 约 295）"。开启投机后，一步要验证的 token 数是：

```
每步验证 token 数 ≈ batch size × 每请求验证节点数
```

batch size=64、链长 3（每请求验证 4 个 token）时就是 256，已经逼近拐点——验证不再"白送"，起草成本也乘上了 batch。SGLang 团队在 H100 + Llama-3.1-8B 上测的吞吐倍数（EAGLE-3 论文 Table 3，链长 3，不用树）印证了这个趋势：

| batch size | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 56 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| EAGLE | 1.40x | 1.38x | 1.23x | 1.02x | **0.93x** | 0.94x | 0.88x | 0.99x | 0.99x |
| EAGLE-3 | 1.81x | 1.82x | 1.62x | 1.48x | 1.39x | 1.32x | 1.38x | 1.34x | 1.38x |

EAGLE 一代在 batch 24 就跌破 1x；EAGLE-3 因为 α 更高，在 batch 64 仍保持 1.38x，但收益已从 1.8x 降到 1.4x 左右。注意这组实验特意**不用树**——树会让每请求的验证节点数翻几倍，更早撞到拐点。

SGLang 维护者在 issue #5274 中的回复很直白："投机解码是给小 batch 用的，不是给压测用的；batch 大时，额外开销和额外显存会拖累性能。"

### 6.2 长上下文让账反转过来：MagicDec

上一条有个例外。Together AI / MagicDec 的分析指出：decode 一步的耗时由两部分组成——读权重和读 KV cache。权重部分的算术强度随 batch 增大而上升（所以 batch 大了会变成算力受限），但**每个请求的 KV cache 是独占的**，读 KV 的算术强度是常数（约等于 GQA 的分组比 g，即多少个 query 头共用一组 KV 头），**永远是访存受限**。

于是在"大 batch + 长上下文"区间，读 KV 占了主导，验证 γ+1 个 token 和验证 1 个 token 读的是同一份 KV，验证又重新变得"几乎白送"。MagicDec 进一步让草稿模型只用**固定窗口的 KV**（StreamingLLM 式的滑动窗口 + attention sink），使草稿成本不随上下文增长；在这个区间里甚至可以拿 target 自己（只看窗口）当草稿。他们报告了一个反直觉的结论：这个区间里 **batch 越大，投机解码的加速反而越高**。

我的判断：6.1 和 6.2 合起来说明，"开不开投机"不是一个开关问题，而是一个依赖 batch size、上下文长度、α 三个运行时变量的**代价决策**。

### 6.3 KV cache 的回滚与额外显存

被拒绝的草稿 token 已经在验证时写进了 target 的 KV cache，必须回滚。分页式 KV 管理（PagedAttention / SGLang 的 token 级 KV 池）让回滚变得便宜——释放对应槽位即可，不需要搬数据。代价在显存：草稿模型的权重、草稿模型自己的 KV、树验证时临时多出的 KV 槽位都要预留。实践中开启投机通常要**调低 `--mem-fraction-static`**（SGLang 源码走读中的示例命令用了 0.7），这直接减少了 KV 池容量，进而减少可同时运行的请求数——又一次在吞吐上付费。

### 6.4 与 CUDA graph、overlap 调度的冲突

CUDA graph（把一串 kernel 提前录成一张图、整体重放以省掉 CPU 发射开销）要求输入形状固定，而投机解码每轮的接受长度不同、树形状可变，引擎必须按"batch × 草稿节点数"多录几套图，录制时间和显存都上涨。另一个更隐蔽的冲突是**overlap 调度**（CPU 准备下一批的同时 GPU 跑当前批）：下一轮要起草什么取决于这一轮验证接受到哪，天然打断流水线。issue #5274 中一位用户报告，在 SGLang 0.5.3 上开启 EAGLE3 时 overlap scheduler 被关闭、batch 上限也被压低；2025 年 7 月 LMSYS 的 MTP 博客也写明当时 MTP 尚不支持 overlap 调度。

这个坑在 2026 年有了系统性的解法：SGLang 重写了投机解码引擎（Spec V2），核心目标就是减少 CPU 与 GPU 之间的同步点，让"上一批的收尾（停止符检测、请求元数据更新）"和"下一批的 KV 分配"与 GPU 计算重叠。LMSYS 报告 Qwen3-8B 在单卡、并发 32 下吞吐提升约 33%，并已把 V2 设为默认、V1 标记为废弃。实操上要留意启动日志：若出现 `Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm`，说明所选算法还没进 V2，这部分收益拿不到。

### 6.5 一个真实反例：τ=3.1，却比不开还慢

同一个 issue 里，RTX 3090 + Qwen3-8B-AWQ（4-bit 量化，5.7 GB）+ EAGLE3 草稿模型，issue 作者实测：

```
不开投机：    125.06 token/s（accept len = 1.00）
开 EAGLE3：   115.31 token/s（accept len = 3.10，-7.8%）
各组参数（steps/topk/draft = 3/2/4、4/5/16、6/10/32、6/5/6）全部比基线慢 16%～24%
```

用第三节的框架拆一下（以下为粗算与推测）：

1. **量化让 target 单步变便宜，c 相对变大**。target 权重只有 5.7 GB，而草稿模型的 decoder 层、融合层和 LM head 仍是 fp16，按每步读取字节粗估 c 在 0.1～0.2 量级；起草 6 步就占掉接近一个 target 步的时间。
2. **4-bit GEMM 的拐点更低**。量化 kernel 要先反量化再算，算力开销更大，"多验证几个 token 几乎白送"的区间比 bf16 窄，验证 32 个树节点的成本明显大于普通单步。
3. **overlap 关闭 + batch 上限被压低**，把剩下的收益吃干净。

套公式：即使只算第 1 条，S ≈ 3.1 / (1 + 6×0.15) ≈ 1.6x 已是上限；再加上第 2、3 条，跌破 1x 并不意外。这个案例最大的价值是提醒：**τ 只是分子，分母要在目标硬件上实测**。

### 6.6 "无损"的边界

第二节证明的是**分布等价**。工程上还有两点要知道：

- 验证时 target 一次处理多个 token，走的 kernel 路径、归约顺序可能与单 token decode 不同，浮点误差会让 greedy 输出在极少数位置出现分叉（这是推理引擎"批量不变性"的一般问题，不是投机解码特有，但投机解码会放大它的可见度）。
- Medusa 的 typical acceptance 等"放宽接受条件"的变体本身就是有损的，选型时要分清。

## 七、方法全景：所有方法放在一起怎么选

第四节沿演进主线讲了几条代表性路线，这一节把目前公开的、主流引擎里能用到的方法**全部**摆出来，只从**推理过程**看它们的优劣：每一轮里草稿怎么产生、起草花多少、多占什么资源、验证时有什么特点。草稿怎么得来不在本节讨论范围内，也不引用具体硬件上的跑分。

### 7.1 全景表

验证环节所有方法基本相同（第二节），差别在"推理时草稿从哪来、怎么起草"。按草稿来源可分为七族，另加一族"不换草稿、只改验证或调度"的方法。表中"每轮起草代价"对应第三节公式里的 γ·c：

| 族 | 方法 | 推理时怎么起草 | 每轮起草代价 | 额外显存 / 状态 | 推理时的优势 | 推理时的劣势 |
|---|---|---|---|---|---|---|
| **无模型·检索型** | PLD / NGRAM | 在 prompt 与已生成内容里匹配最近几个 token，把其后的片段直接当草稿 | 查表，几乎为零 | 无 | 起草几乎不花钱；输出大量复制输入时（改写、摘要、代码修改、RAG 引用）一次能猜很长 | 匹配不上就没有草稿；开放生成几乎无收益；草稿长度随匹配情况忽长忽短 |
| | Lookahead | 每步顺带跑一组 Jacobi 迭代，在运行中自产 n-gram 池，再从池里取候选 | 每步多带一批猜测 token 进 forward | 候选池，很小 | 不依赖输入里有重复片段 | 用的正是闲置算力，batch 一大就没得用；α 不高 |
| | Suffix Decoding | 对当前请求和历史请求的输出建后缀树，按匹配长度和出现频率决定猜多长 | 查树，很小 | 后缀树缓存（主机内存） | 跨请求高度重复（agent 循环、批量代码修改）时一次能猜很长；没把握时自动少猜 | 新内容几乎无收益；缓存规模要控制 |
| | REST | 从外部语料库检索续写片段 | 一次检索 | 检索库 | 领域内固定表述多时有效 | 检索本身有延迟；收益取决于语料与业务的贴合度 |
| **独立草稿模型** | SpS（经典草稿模型） | 同家族小模型自回归猜 γ 步 | γ 次小模型 forward | 小模型权重 + 一份独立 KV | 最通用，不碰 target 内部状态 | α 偏低；多维护一份 KV、多占显存；要求共享词表 |
| | 跨词表草稿（vLLM TLI） | 异家族小模型自回归起草，按两套词表的交集把 token 映射回 target | 同上 + 词表映射 | 同上 | 同家族没有小模型时也能用 | 草稿只能落在词表交集内，α 更低；目前只支持贪心起草 |
| | PARD | 小模型一次 forward 同时猜多个位置 | 1 次小模型 forward，不随 γ 增长 | 小模型权重 + KV | 起草代价与 γ 脱钩；不读 target 内部状态，一个草稿可搭配同家族多个 target | 看不到 target 隐状态，α 上限低于特征级方法 |
| **自推测（不加新模型）** | Draft & Verify / LayerSkip | target 跳过部分中间层，或在浅层提前退出，用浅层输出起草 | γ 次"截短的 target" forward | 无额外权重，KV 可与 target 共用 | 不多占显存，部署最省事 | 浅层预测质量有上限；每次起草仍要跑 target 的一部分层，c 不算低 |
| | MagicDec 式窗口自推测 | target 自己只看滑动窗口内的 KV 来起草 | γ 次 target forward，但只读固定窗口 KV | 窗口 KV | 起草代价不随上下文变长而增长；长上下文 + 大 batch 下验证几乎白送（6.2） | 短上下文时拿 target 当草稿太贵，不划算 |
| **预测头** | Medusa | target 顶层并列 K 个预测头，一次各自给出往后第 k 个 token | 几个小头，几乎为零 | 极少，无独立 KV | 起草最便宜 | 各头互不知情，越往后越不准；通常要配树验证，验证节点多，大 batch 下更早撞拐点；typical acceptance 规则有损 |
| | Hydra | 预测头串联，后一个头看得到前一个头选了什么 | K 个小头顺序执行，仍很便宜 | 极少 | α 明显高于 Medusa | 头之间变成顺序依赖 |
| | MLP Speculator | 多级 MLP，每级以 target 隐状态和上一个猜出的 token 为输入 | 很小 | 极少，无独立 KV | 结构轻，没有 attention | 只依赖当前一步的隐状态，猜得越远越不准 |
| | ReDrafter | 以 target 隐状态为条件的 RNN 头，配合 beam search 产出多条候选 | 少量 RNN 步 | 小 | 头部容量比 Medusa 大，α 更高 | beam 带来多条候选，验证量随之增加 |
| **特征级外挂草稿** | EAGLE / EAGLE-2 | 1 层 decoder 读 target 顶层特征和 token 自回归起草；二代按置信度动态长树 | γ 次 1 层 forward | 1 层权重 + 很小的 KV | 读 target 内部状态，α 高；动态树在难处多分叉、易处少浪费 | 要等 target 本步隐状态出来才能起草，草稿与验证串行；树大时验证节点多 |
| | EAGLE-3 | 融合 target 低/中/高三层特征起草，草稿用缩小的词表 | γ 次 1 层 forward，LM head 更小 | 同上 | 外挂方案里 α 最高的一档；猜得长时接受率衰减小（4.4） | 同上；推理时要从 target 三处层抓取隐状态 |
| | P-EAGLE | 把 EAGLE 的起草改为一次 forward 并行产出 | 1 次 forward，不随 γ 增长 | 同 EAGLE | 进一步压低 c | 并行位置互不知情，连贯性要额外处理 |
| **模型原生 MTP** | DeepSeek-V3/V4、Qwen3.5 及之后、GLM-4.5 系列、MiMo 系列等 | 模型自带的 MTP 层接在主干后面起草；猜多步时重复调用 MTP 层，或依次用多个 MTP 层 | 每猜一步跑一次 MTP 层 | MTP 层权重随模型发布，开销小 | 与 target 天然同版本，α 高且稳定；引擎支持最成熟，风险最低 | 起草代价随步数线性增长；只有 1 层 MTP 的模型，猜多步时靠后位置的 α 下降 |
| | Gemma 4 MTP | 与 target 配对、单独打包的 MTP 草稿，直接复用 target 的 KV | 自回归 | 独立权重，但不另存 KV | 不另占 KV | 必须与对应的 target 版本配对使用 |
| **并行块起草** | DFlash | 块扩散草稿一次 forward 并行起草一整块，target 多层特征注入草稿每一层的 KV | 1 次 forward 出一整块 | 草稿权重（比 EAGLE 深）+ KV | 起草代价与 γ 脱钩，可以用更深的草稿换更高的 α（4.6）；低并发时优势最大 | 块内各位置互不知情，可能拼出不连贯组合；整块送验证，验证量大，并发升高后优势收窄 |
| | DSpark | DFlash 式并行主干 + Markov 头（按前一个已选 token 修正下一位置）+ 置信度头 | 1 次 forward + 很轻的顺序修正 | 略多于 DFlash | 补上块内连贯性；置信度可用来决定验证多少（见下方自适应验证） | 多一小段顺序计算 |
| | DFlash 2 | 每个位置保留 top-16 候选，再由轻量选择器对相邻候选打分、追出一条连贯路径 | 1 次 forward + 路径选择 | 同 DFlash | 补上块内连贯性 | 引擎支持尚在 PR 分支；模型卡标注的有效上下文为 4096 token，更长上下文下草稿质量没有保证 |
| | LiLiCorr | DFlash 主干给出每个位置的候选，再由相关器给候选组合打分、逐位挑选 | 1 次 forward + 相关器打分 | 同 DFlash | 与 DFlash 2 同一思路的另一种实现 | 草稿长度最好固定为块大小减 1，改短可能让接受长度不可预期地下降 |
| **不换草稿，改验证 / 调度** | 树验证（静态树 / 动态树） | 不改起草，一次 forward 同时验证多条候选路径（第五节） | —— | 临时 KV 槽位 | 同样一次验证，接受得更长；可叠加在大多数草稿方法上 | 每请求验证节点成倍增加，大 batch 下最早撞到拐点（6.1） |
| | 动态投机 | 按实时负载调整草稿长度，必要时关闭投机 | —— | —— | 负载变化时避免收益反转 | 切换阈值本身要按负载调 |
| | 自适应验证 | 按草稿置信度为每个请求决定送多少 token 去验证 | —— | —— | 少验证明显猜不中的尾巴，省下验证算力 | 依赖草稿给出可靠的置信度；vLLM 目前仅对 DSpark 开放 |
| | 宽松验证（Medusa typical acceptance、vLLM FLy） | 放宽拒绝采样的接受条件 | —— | —— | 接受长度更长 | **有损**，不再保证与原模型同分布 |

引擎侧：vLLM 文档列出的方法包括 EAGLE、MTP、独立草稿模型、PARD、n-gram、Suffix Decoding、LiLiCorr，以及 `dflash`、DSpark 等，另有实验性的自定义 proposer 接口，可以接入自研的起草方法；SGLang 文档列出 EAGLE-2/EAGLE-3、MTP、DFLASH、经典草稿模型、NGRAM，以及 UNO 等选项（UNO 本文未展开）。各方法的具体可用性以所用版本文档为准。

### 7.2 把全景表压成几条规律

以下为判断：

1. **起草方式决定 c 怎么随 γ 变化**。自回归起草（SpS、EAGLE、原生 MTP、ReDrafter）每多猜一个 token 就多一次 forward，γ 有明确的最优值；并行起草（Medusa、PARD、P-EAGLE、DFlash 系）一次 forward 出一整段，γ 可以拉长，但要额外解决"块内不连贯"——Hydra、DSpark、DFlash 2、LiLiCorr 都是在并行主干后补一点很便宜的顺序修正。
2. **草稿在推理时看到 target 的信息越多，α 越高，但对 target 的依赖也越强**。检索型完全不碰 target，可以放在 CPU 上、与 target 的计算重叠；独立小模型只依赖共用词表；预测头、特征级草稿、原生 MTP、DFlash 系都要等 target 本步的隐状态出来才能起草，草稿和验证在时间上串行，而且只能搭配对应版本的 target 使用。
3. **额外显存分三档**。检索型和自推测几乎不多占；预测头、原生 MTP、EAGLE 系只多一点权重和很小的 KV；独立小模型和 DFlash 系最多（较大的草稿权重 + 独立 KV）。显存多占一分，KV 池就小一分，能同时跑的请求就少一些（6.3）。
4. **验证量是大 batch 下的隐形成本**。树验证、整块验证、beam 候选都会放大每个请求送去验证的 token 数，让 6.1 的拐点来得更早；自适应验证、动态投机就是在压这一项。
5. **检索型和模型型解决的是两类问题**。前者赚的是"输出与已有文本重复"，后者赚的是"小网络能模仿大模型"。前者在合适的负载上近乎白捡，在不合适的负载上几乎为零。
6. **验证 / 调度侧的方法与草稿方法正交**。树验证、动态投机、自适应验证都可以叠加在已选定的草稿方法上；宽松验证是唯一会改变输出分布的一类，需单独评估。

### 7.3 选型顺序

**没有全面胜出的方法**。公开对比的定性结论是：原生 MTP 与 DFlash 谁更好随具体模型而变，同一模型家族里不同型号都可能反转；块式方法在低并发时优势最大，并发升高后优势收窄，开放式对话在高并发下可能几乎没有收益。因此选型首先看"负载长什么样"和"手上有没有现成可用的草稿"（我的判断）：

1. **先看负载**：输出大量复制输入，或跨请求高度重复（agent 循环、批量代码修改），先上 NGRAM / Suffix Decoding，起草几乎不花钱，也不多占显存。
2. **模型自带 MTP**：用 MTP 打底线。LMSYS 的建议是从 `draft_token_num=2` 起步，算力有余量再往 4 加，已经满载就停在 2～3。
3. **有与 target 配套发布的外挂草稿**（DFlash、DSpark、EAGLE-3）：与第 2 步的 MTP 做一次 A/B，按端到端吞吐择优；DFlash 2 这类引擎支持还在 PR 分支的，等进正式版再上生产。
4. **都没有**：选不依赖 target 内部状态的方法——同家族小模型、跨词表小模型或 PARD；显存紧张、不想多加任何权重，选自推测类。
5. **长上下文 + 大 batch**：关注 MagicDec 式窗口自推测（6.2）。
6. **负载波动大**：在已选方法上叠加动态投机 / 自适应验证；宽松验证只在业务能接受输出分布偏移时使用。
7. **最后一定要扫草稿长度**：自回归类通常在前几档就进入平台期；块式方法的最优点往往在中等长度，不必拉满到块大小。**按端到端吞吐挑，不按接受长度挑**——接受率更低但起草更便宜的方法照样可以更快，反之亦然。

## 八、多卡部署：单机内的五个问题

前面的加速账基本都在单卡上算。切到多卡后，第三节公式里的每一项都会被改写。

### 8.1 草稿与 target 抢同一组卡

当前 vLLM 和 SGLang 的默认做法都是让草稿模型跑在 target 的 TP 组上（TP，张量并行：把每层矩阵切到多张卡上一起算），共用同一批 SM 和显存带宽、走同一条关键路径。vLLM 的 RFC #42109 给出的数字是：TP=2 的 Llama-3-70B 在高并发下，一个 1B～8B 的草稿模型会让每步延迟增加 10%～30%。

我的判断：草稿模型很小，切到 8 张卡后每卡几乎没有计算，但每起草一步仍要付一轮 all-reduce（多卡之间把部分结果加总的通信）和一轮 kernel 发射的固定开销——**在 TP 下，c 往往比单卡时更大**。自回归起草（MTP/EAGLE）步数越多越吃亏；DFlash 一次 forward 起草，同步点少，在 TP 下的相对优势会比单卡时更明显。

### 8.2 流水线并行基本用不了

vLLM 文档写明：截至 0.15.0，流水线并行（PP：把不同层放到不同卡上，像流水线一样传递）不能与投机解码组合。我的判断有两层原因：一是每轮验证必须走完整条流水线、拿到接受结果后才能起草下一轮，与 PP 本身的流水线气泡叠加；二是 EAGLE-3、DFlash 需要 target **多个层**的隐状态，在 PP 下这些层分散在不同流水段，要跨段收集。学术界有 SpecPipe 这类专门把两者结合的工作，但工程上现在的结论是：**要用投机解码，并行方式优先选 TP/EP，避开 PP**。

### 8.3 MoE + 专家并行：收益曲线不再单调

MoE（混合专家：每个 token 只激活一小部分"专家"子网络）给第一节的账加了一个变量：验证 k+1 个 token 时，这些 token 可能被路由到不同专家，要加载的专家权重比单 token decode 多。MoESD 的分析和 Cohere 在 vLLM 上的实测都显示，MoE 的投机加速随 batch **先升后降**：

- 小 batch：单 token decode 只加载少数专家，验证多 token 会额外加载更多专家，收益受限；
- 中等 batch：反正几乎所有专家都要加载，验证不再额外加载权重，而 MoE 的稀疏性让整层仍处在访存受限区——**甜点区**；
- 大 batch：进入算力受限，收益消失。

对比之下，稠密模型的收益是随 batch 单调下降的。Cohere 还测到相邻 token 选择的专家有明显重叠：验证 4 个 token 实际激活约 2.5 倍 top-k 的专家，而不是朴素估计的 3.2～3.6 倍。

放到专家并行（EP：不同专家放在不同卡上，token 通过 all-to-all 通信发往对应专家所在的卡）上，我的判断是：每步 all-to-all 要分发的 token 数变成 batch × (k+1)，DeepEP 一类通信库的分发缓冲上限要按这个规模重新设——SGLang issue #20404 中维护者的复现命令在 prefill 侧显式设了 `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK`（12288～16384），可作旁证。另外，DeepSeek-V3 的 MTP 模块本身就含一个 MoE 层（模型卡公布的 685B = 671B 主模型 + 14B MTP 模块），**起草这一步也要走 EP 通信**，不是"在本卡上随手算一下"。

### 8.4 DP attention 与调度同步

DP attention（注意力部分按数据并行切分、每个 rank 各管一批请求，MoE 部分再汇合做 EP）是 DeepSeek 类大 MoE 部署的常见形态。我的判断：各 rank 的 batch 不一样，但 MoE 层要求所有 rank 一起参与通信，所以起草步数必须在 rank 之间对齐，没有请求的 rank 也要陪跑空 forward。这个组合在工程上是逐步打通的：2025 年 3 月 SGLang 还有"DP attention + EAGLE worker 报错"的 issue（#4847）；到 2025 年 7 月 LMSYS 宣布 MTP 已能与 DP attention、EPLB（专家负载均衡）、DeepEP、Two Batch Overlap、PD 分离组合使用。

### 8.5 显存在每张卡上都要再让一次

草稿权重、草稿自己的 KV、树验证临时多占的 KV 槽位，在每张卡上都要预留，`--mem-fraction-static` 往往要调低，可同时运行的请求数随之下降（6.3 节的账在多卡上照样成立）。DFlash 的注入 KV 按完整上下文存放，上下文越长占得越多；SGLang 的实现为此做了"立即物化"——提前把 target 隐状态投影成草稿 KV、不额外保存隐状态，并让同前缀请求共享 radix cache。

## 九、集群部署：跨节点的五个问题

### 9.1 PD 分离：多了一份要传的状态

PD 分离（prefill 和 decode 放在不同节点池，prefill 算完后把 KV cache 传给 decode 节点）下，我的判断是：decode 侧的草稿模型第一步就需要 prompt 末尾的 target 隐状态，以及草稿自己那一份 prompt 的 KV——要么随 target KV 一起从 prefill 节点传过来，要么在 decode 侧重算。传输协议里多了一类状态，就多一类出错的可能。

SGLang issue #20404 是一个真实现场。提交方用 PD 分离跑 DeepSeek-R1-0528，报了两个独立故障：一是开 DP attention + EP 后输出完全乱码（GSM8K 0.0%）；二是不开 DP attention、只开 MTP（NEXTN，steps=1、draft_tokens=2），1,319 条回答里只有 2 条走完思考阶段，其余都在推理中途被截断，表现为提前吐出结束符（strict 0.2%）；同模型、同拓扑去掉 MTP 则为 94.6%。维护者在 SGLang 主线上把 PD 分离 + DP attention + EP + MTP 全叠在一起也复现不出（GSM8K 约 94.8%），问题出在提交方用于测试的 fork 镜像。

这个案例的教训比 bug 本身更重要：**投机解码的"无损"是算法保证，不是实现保证**。草稿猜得差只会让它变慢；一旦输出变了，就是验证、接受、KV 簿记某个环节的实现出了错。所以集群里每换一种特性组合，都要跑准确率评测（如 GSM8K），不能只看吞吐。

### 9.2 规模越大、单卡并发越高，收益越薄

LMSYS 对 DeepSeek-V3 + MTP 做了两组集群实验：

| 部署 | 每 rank 并发 | 输入 / 输出长度 | 结果 |
|---|---|---|---|
| 2 个 decode 节点，共 16 卡 | 2 | 64K / 4K | 基线 51.0 token/s（不开 overlap、不开 MTP）；只开 overlap 60.4；开 MTP（4 token 窗口）82.0，**+60%**，τ=2.44 |
| 4 个 prefill 节点 + 12 个 decode 节点，共 128 卡 | 128 | 2K / 100 | MTP（step=1、draft_token_num=2）只有 **+14.2%**，且对比的是不开 overlap 的基线 |

第一组是"长上下文、低并发"，对应 6.2 节"读 KV 主导、验证几乎白送"；第二组是"短上下文、高并发"，对应 6.1 节"闲置算力被 batch 吃掉"。两组数字正好是前文两条理论在集群尺度上的印证。

### 9.3 路由与 SLO 变得更难预测

以下为判断：接受率因请求而异（代码、数学高，开放闲聊低），开投机后同一集群内请求间的每 token 延迟方差会变大；路由器不知道每个请求的接受率，同一 batch 通常共用一套草稿长度；按 token/s 做容量规划和自动扩缩容时，开投机前后是两条不同的吞吐曲线，要分别测，不能拿单流加速比去乘。

### 9.4 草稿与 target 的版本绑定

EAGLE/DFlash 草稿是对着某个具体 target 训练的。target 一做微调、换版本，外挂草稿就要重训——不重训不会出错，但 α 会悄悄下降，而且没有任何报错。集群里同时跑多个模型版本时，要**按部署维度监控接受长度**。草稿的训练上下文长度也是一个隐性边界：DFlash 2 模型卡注明训练上下文为 4096 token，我的判断是超出这一长度的请求，接受率可能明显走低，上线前要在真实长度分布上测。

### 9.5 新方向：草稿单独成池

vLLM RFC #42109 提出"分离式投机解码"：草稿模型放到独立的 GPU 池，N 个验证服务共享 M 个草稿服务；草稿侧在 target 验证本轮时，就按"本轮可能接受几个、bonus token 是什么"提前把下一轮的草稿算好放进缓存。他们的内部原型在 Llama-3.1-70B（验证）+ Llama-3.2-1B（草稿）、N:M=3:2 下，在同样 GPU 总数下每 token 延迟比同卡部署好约 1.4 倍。它直接针对 8.1 节的资源争抢，但目前只是 RFC 加内部原型。

## 十、动手：在 SGLang 里开起来、看什么

启动命令骨架（参数名取自 SGLang 文档与源码走读；草稿权重需与 target 配套）：

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path <与 target 配套的 EAGLE3 草稿权重> \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 16 \
  --mem-fraction-static 0.7
```

可选的 `--speculative-algorithm` 还包括 `EAGLE`、`NGRAM`、`DFLASH`、MTP 类模型的原生草稿，以及经典独立草稿模型等，具体以所用版本文档为准。

运行时最该盯的是 decode 周期统计行里的**接受长度**字段（典型形态如下，字段名和格式以实际版本为准，现场以实测回填）：

```
Decode batch. #running-req: 1, #token: 1234, token usage: 0.01, accept len: 3.21, cuda graph: True, gen throughput (token/s): 350.12
```

读法：`accept len` 就是第三节的 τ；它和 `gen throughput` 要**一起看**——τ 高但吞吐不涨，说明分母（起草、验证、调度）出了问题，参照 6.4、6.5 排查。逐字段读法见 [SGLang 日志逐行分析手册](./sglang_log_analysis.md)。

建议的最小实验（可放进 W4 serving 实验体系）：固定 prompt 集，batch size 扫 {1, 4, 16, 64}，对比"不开 / EAGLE3 链式（topk=1）/ EAGLE3 树式"三档，记录 accept len 与 gen throughput。预期（待验证假设）：小 batch 树式最快，中 batch 链式反超，大 batch 两者都可能低于基线——即 6.1 节那张表的本地复现。

## 十一、什么时候该开：一份判断清单

以下是基于前文事实整理的经验判断，不是定论。先分两类粗看：

**基本没问题、收益稳的场景**：单卡或单机小 TP 下的低并发（约 1～8 路）、延迟敏感负载——交互对话、agent 单链路、代码生成、数学与长推理输出；长上下文 + 中低并发（9.2 节 64K 输入 +60%）；模型自带 MTP、引擎官方支持、草稿长度保守（2～3）；MoE 模型落在中等 batch 的甜点区（8.3）；输出大量复制输入的任务用 NGRAM 零成本拿收益。

**容易出问题、要先测或干脆不开的场景**：高并发吞吐压测与离线批处理（6.1、7.3、9.2 都在说这件事）；PP 部署（8.2）；PD 分离 + DP attention + EP + MTP 全叠在一起——性能可以有，但正确性风险最高，必须跑准确率评测并锁定上游已验证的镜像（9.1）；消费级显卡 + 低比特量化 target（6.5）；开放闲聊、创意写作、高温采样；上下文超出草稿训练窗口、或 target 已微调而草稿没重训——不会出错但收益悄悄消失（9.4）；还在 PR 分支上的新算法（7）。

逐条展开：

| 场景 | 建议 | 依据 |
|---|---|---|
| 低并发、单流延迟敏感（交互对话、agent 单链路） | 开，优先 EAGLE-3 / MTP | 闲置算力最多（6.1） |
| 输出大量复制输入（改写、摘要、代码修改、RAG 引用） | 先试 NGRAM/PLD，零成本 | 4.2、4.7 中 PLD 在 CNN/DM 上的表现 |
| 高并发、短上下文、吞吐优先 | 谨慎，必须做 batch 扫描实测 | 6.1 的拐点账 |
| 高并发、长上下文 | 反而可能值得，关注 MagicDec 类方案 | 6.2 的 KV 访存账 |
| 消费级显卡 + 低比特量化 target | 先测再说，不要只看 τ | 6.5 的反例 |
| 高 temperature 采样 | 预期 τ 明显下降 | 4.7 中 temperature=1 的数据 |
| 没有配套草稿权重 | NGRAM 兜底，或用 SpecForge 自训 EAGLE-3 | 4.4 的 scaling law |
| 模型自带 MTP | 先用 MTP 打底线，再与 DFlash 做 A/B | 7 中各模型结论不一 |
| 单机多卡 TP | 优先同步点少的方法，草稿长度从保守值起扫 | 8.1 |
| 需要 PP | 改用 TP/EP，或放弃投机 | 8.2 |
| 大 MoE + EP + DP attention | 先找中等 batch 甜点区，按 batch×(k+1) 设通信缓冲 | 8.3、8.4 |
| PD 分离集群 | 每换一种特性组合都跑准确率评测 | 9.1 |
| 多模型版本并存的集群 | 按部署监控接受长度，target 升级时同步重训草稿 | 9.4 |

## 十二、结语：又一个"策略是静态的"现场

把全文压成一句话：**投机解码是用"多算一些可能白算的 token"去换"少搬几次权重"，它赚的是 GPU 在访存受限时闲着的那部分算力。** 草稿方法的演进（小模型 → 预测头 → 特征级 → 训练时测试 → 并行起草）都在优化同一个公式的分子 α 和分母 γ·c；而系统层面的所有坑，本质上都是"闲置算力"这个前提在大 batch、量化、调度流水线面前被打破——上了多卡和集群，又多出通信同步、专家路由、跨节点状态传输三类打破它的方式。

如果套用《LLM 推理栈的三处空白》的视角（我的判断）：投机解码是"状态动态、策略静态"的又一个现场——每一轮的接受长度、当前 batch 大小、上下文长度都在逐步变化，而"开不开、γ 取多少、树多宽"仍然主要靠启动参数和离线自动调参决定。EAGLE-2 的动态树已经在"单请求内"做了按置信度剪枝；把这个决策上提到"整个 batch、按实时代价估算选策略"，是一个尚未被系统性填上的位置。

## 参考资料

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023. https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023.
- Xia et al., *Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding*, Findings of ACL 2024. https://aclanthology.org/2024.findings-acl.456.pdf
- Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, 2024.
- Fu et al., *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding*, ICML 2024. https://arxiv.org/abs/2402.02057
- Li et al., *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test*, 2025. https://arxiv.org/abs/2503.01840
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024. https://arxiv.org/abs/2412.19437
- Chen et al., *MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding*, ICLR 2025. https://arxiv.org/abs/2408.11049
- Together AI, *Speculative decoding for high-throughput long-context inference*. https://www.together.ai/blog/speculative-decoding-for-high-throughput-long-context-inference
- Chen, Liang, Liu, *DFlash: Block Diffusion for Flash Speculative Decoding*, ICML 2026. https://z-lab.ai/projects/dflash/
- LMSYS, *SpecForge: Accelerating Speculative Decoding Training*. https://www.lmsys.org/blog/2025-07-25-spec-forge/
- yubofredwang 博客, *Speculative Decoding, EAGLE3 and SGLang*（SGLang 实现走读）. https://yubofredwang.github.io/posts/Speculative-Decoding-EAGLE-Sglang/
- sgl-project/sglang issue #5274, *The performance stress test results of EAGLE-3 are not good*. https://github.com/sgl-project/sglang/issues/5274
- SGLang 文档，Speculative Decoding. https://docs.sglang.io/docs/advanced_features/speculative_decoding
- LMSYS / Eigen AI, *Accelerating SGLang with Multiple Token Prediction*, 2025. https://www.lmsys.org/blog/2025-07-17-mtp/
- Z Lab, Modal, SGLang, *The next generation of speculative decoding: DFlash and Spec V2*, 2026. https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/
- vLLM Blog, *Exploring Speculative Decoding in vLLM on AMD GPUs*, 2026. https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus
- incoai / z-lab, *Qwen3.8-27B-DFlash2* 模型卡. https://huggingface.co/incoai/Qwen3.8-27B-DFlash2
- Cohere, *Why MoE Models Get More From Speculative Decoding*, 2026. https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding
- Huang et al., *MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE*. https://openreview.net/forum?id=FAeU7516MR
- vLLM 文档，Speculative Decoding（PP 组合限制）. https://docs.vllm.ai/en/latest/features/speculative_decoding/
- vllm-project/vllm RFC #42109, *Disaggregated Speculative Decoding with Standalone Parallel Draft Model*. https://github.com/vllm-project/vllm/issues/42109
- sgl-project/sglang issue #20404, *MI355X disagg DPA + MTP Errors*. https://github.com/sgl-project/sglang/issues/20404
- sgl-project/sglang issue #4847, *DP attention with Eagle worker raises AttributeError*. https://github.com/sgl-project/sglang/issues/4847
- *SpecPipe: Accelerating Pipeline Parallelism-based LLM Inference with Speculative Decoding*. https://arxiv.org/abs/2504.04104
- He et al., *REST: Retrieval-Based Speculative Decoding*, NAACL 2024. https://arxiv.org/abs/2311.08252
- Snowflake, *Fastest Speculative Decoding in vLLM with Arctic Inference and Arctic Training*（Suffix Decoding）. https://www.snowflake.com/en/blog/engineering/fast-speculative-decoding-vllm-arctic/
- vLLM 文档，Suffix Decoding. https://docs.vllm.ai/en/stable/features/speculative_decoding/suffix/
- vLLM 文档，LiLiCorr. https://docs.vllm.ai/en/latest/features/speculative_decoding/lilicorr/
- An et al., *PARD: Accelerating LLM Inference with Low-Cost PARallel Draft Model Adaptation*, ICLR 2026. https://arxiv.org/abs/2504.18583
- Zhang et al., *Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding*. https://arxiv.org/abs/2309.08168
- Elhoushi et al., *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding*. https://arxiv.org/abs/2404.16710
- Wertheimer et al., *Accelerating Production LLMs with Combined Token/Embedding Speculators*（MLP Speculator）. https://arxiv.org/abs/2404.19124
- Cheng et al., *Recurrent Drafter for Fast Speculative Decoding in Large Language Models*（ReDrafter）. https://arxiv.org/abs/2403.09919
- AWS, *P-EAGLE: Faster LLM inference with Parallel Speculative Decoding in vLLM*. https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/
