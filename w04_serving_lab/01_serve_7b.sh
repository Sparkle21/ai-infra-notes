#!/usr/bin/env bash
#
# W4-01 · S2 会话 serve 启动脚本（7B，OpenAI 兼容 API）
# =====================================================
# 先补几句背景（这个脚本干什么）：
#   "serve" = 把模型加载进显存、启动一个 HTTP 服务，对外提供对话 API。
#   本脚本在「租的 GPU 机器」上把 SGLang 服务拉起来（选型：当下主流推理引擎
#   已普遍转向 SGLang），并支持四档参数（基线/A/B/C）——四档是四个对照实验，
#   用于观察 W3 学过的调度机制在真实日志里长什么样。
#
#   SGLang 与 vLLM 机制同源、术语有差：vLLM 的 preemption 在 SGLang 里叫
#   retract；前缀缓存（radix cache）SGLang 默认开；启动多一步 CUDA graph
#   「录像」预热（W1-01 的概念）。逐行读日志见 ../docs/sglang_log_analysis.md。
#
# 名词白话：
#   - OpenAI 兼容 API：提供和 OpenAI 一模一样的接口格式（/v1/chat/completions
#     等），好处是测试脚本不用改就能直连（SGLang 同样提供）；
#   - nohup / &：让服务在后台跑，关掉终端也不死；
#   - 轮询（polling）：每隔几秒问一次「好了没」，直到就绪或超时。
#
# 用法（租卡机上）：
#   bash 01_serve_7b.sh            # 默认：实验基线参数
#   EXP=c bash 01_serve_7b.sh      # 实验 C：逼 preemption 的小配额
#
# 设计原则：
#   - 每个实验 = 一组参数差异，其余保持不变（对照实验的自变量控制）；
#   - 日志全部落盘（serve.$EXP.log），S2 回 Mac 靠它复盘；
#   - 版本敏感：SGLang 参数名随版本变，开机先
#     `python -m sglang.launch_server --help | grep <参数>` 确认（版本差异与
#     日志读法见 ../docs/sglang_log_analysis.md 第 7 节）。
#
set -euo pipefail
# ↑ 三重保险：-e 任一步失败立即停；-u 用了未定义变量报错；-o pipefail 管道中
#   任何一环失败都算整体失败。脚本宁停不精，防止带病继续跑出错误结论。

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"     # 数据盘路径或 ModelScope ID，现场确认；${VAR:-默认} = 没设就用默认
EXP="${EXP:-base}"                             # base | A | B | C
PORT="${PORT:-8000}"                           # HTTP 服务端口

# ── 公共参数（四档实验共享）；白话逐个说（这些直接决定 W3 学过的调度行为）──
#   --mem-fraction-static 0.90：静态显存盘子（权重+KV 一次性划走 90%，
#     运行时激活另算——static 的含义）对应 vLLM 的 --gpu-memory-utilization；
#     实验C 把它收到 0.35 = 故意只给一小块，把 KV cache 挤瘦，逼出 retract
#     （SGLang 版「抢占」）；
#   --context-length 8192：单条请求最长 8192 token（含输入+输出）；
#   注：SGLang 逐请求日志默认关（vLLM 里是 --disable-log-requests 显式关），
#       要解剖单请求时延就临时加 --log-requests。
COMMON=(
  --model "$MODEL"
  --port "$PORT"
  --mem-fraction-static 0.90
  --context-length 8192
)                                              # 逐请求日志默认关：要统计调度，不要内容

case "$EXP" in
  # 四档实验：只改调度相关的自变量，其余全部不变（对照实验）
  base)   # 基线：默认调度参数
    EXTRA=(--max-running-requests 256) ;;      # batch 最多 256 个并发请求（近似默认）
  A)      # 实验A：continuous batching 观察位（batch 上限收到 16，现象更清楚）
    EXTRA=(--max-running-requests 16) ;;       # 只许 16 个并发 → 日志里 batch 变化更易看清
  B)      # 实验B：chunked prefill（SGLang 默认开；显式给预算让日志可读）
    # --chunked-prefill-size 2048：一口 prefill 最多咬 2048 个 token。
    # 长 prompt 会被切成小块分步消化（chunked prefill）——「大口吞」改「小口咬」
    EXTRA=(--max-running-requests 16 --chunked-prefill-size 2048) ;;
  C)      # 实验C：retract（KV 配额收紧 + 长输出并发才逼得出来）
    EXTRA=(--max-running-requests 16 --mem-fraction-static 0.35) ;;   # 显存只给 35% → KV 紧张 → 触发 retract
  *) echo "unknown EXP=$EXP (base|A|B|C)" >&2; exit 2 ;;
esac

LOG="serve.${EXP}.log"
echo "[$0] EXP=$EXP  model=$MODEL  log=$LOG"
echo "[$0] 启动命令：python -m sglang.launch_server ${COMMON[*]} ${EXTRA[*]}"

# ── 启动（后台 + 日志）。退出码和 PID 打出来，方便现场掌控 ────────────────
# nohup：忽略挂断信号（关终端不死）；>$LOG 2>&1：正常输出和错误输出都写进日志文件；
# &：放后台跑，脚本继续往下走（去等服务就绪）。
nohup python -m sglang.launch_server \
    "${COMMON[@]}" "${EXTRA[@]}" \
    >"$LOG" 2>&1 &
SRV_PID=$!     # $! = 刚启动的后台进程编号（PID），之后靠它查活

echo "[$0] server pid=$SRV_PID"

# ── 就绪等待：轮询 /health（SGLang 提供），最多 20 分钟 ─────────────────────
# 注意：SGLang 启动比 vLLM 多一步 CUDA graph 录制（日志里 Capture cuda graph
# begin，官方预告 "several minutes" 属正常）——20 分钟余量把这段算在内。
echo "[$0] 等待就绪（模型加载 + CUDA graph 录制可能要几分钟）..."
for i in $(seq 1 60); do
  # curl -sf：-s 安静模式 -f 拿到 HTTP 错误码时当失败；成功即服务活着
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[$0] ✓ ready (${i}0 秒内)"; exit 0
  fi
  # 进程死了就别傻等：kill -0 只探测进程在不在（不发真信号），死了立即吐日志尾部帮助定位
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[$0] ✗ server 进程退出，看日志：tail -50 $LOG" >&2; tail -50 "$LOG"; exit 1
  fi
  sleep 10      # 每 10 秒问一次；60 次 × 10 秒 = 20 分钟超时
done
echo "[$0] ✗ 20 分钟未就绪" >&2; tail -30 "$LOG"; exit 1

# TODO(you)（Mac 预习时做）：
#  1. 通读本脚本，把每个参数在 scheduler 上的含义写进 04_observation_points.md；
#  2. 查你将安装的 SGLang 版本文档，确认 --chunked-prefill-size /
#     --mem-fraction-static 等参数名没变；变了就改本文件（改动记进 git commit）。
#  3. S2 现场注意：启动日志里找「max_total_num_tokens」字样——那是 KV 账本的
#     总容量（token 数），抄给 W3 笔记；「Capture cuda graph end」的耗时与
#     mem usage 也抄一份（W1 录像带概念的第一次真机现身）。
