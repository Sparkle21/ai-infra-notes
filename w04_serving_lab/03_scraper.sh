#!/usr/bin/env bash
#
# W4-03 · S2 会话现场数据抓取器
# =================================
# 先补几句背景（这个脚本干什么）：
#   实验跑着的时候，光看日志不够——还得知道「显存用了多少、GPU 忙不忙」
#   这些瞬时状态。本脚本就是一个采样器：每隔几秒拍一张快照存进文件，
#   实验结束后拿这些快照回 Mac 画曲线复盘。
#
# 用法（租卡机上，另开一个终端窗口）：
#   bash 03_scraper.sh serve.base.log          # 每 2s 抓一轮，Ctrl-C 停
#   bash 03_scraper.sh serve.c.log 1           # 采样间隔改 1s（实验 C 现象短）
#
# 产出（当前目录）：
#   scraper_gpusmi.tsv   —— 时间戳 + 显存 + 利用率（TSV，Mac 上直接画图）
#   scraper_logtail.txt  —— 每轮的日志尾巴（含时间戳分隔线，方便回放）
#
# 名词白话：
#   - TSV：制表符分隔的数据表（每行一条记录），Excel/Python 都能直接读；
#   - nvidia-smi：NVIDIA 官方命令，查显存/利用率等 GPU 状态；
#   - 采样（scrape/sample）：定期记录一次瞬时值。
#
# 为什么不用 nvidia-smi dmon：输出列名随驱动版本漂，TSV 自己写更稳。
#
set -euo pipefail
# ↑ 同 01 脚本：任一步失败即停/未定义变量报错/管道失败传导，防止带病采样。

# ${1:?提示}：第一个参数必填，缺了就打印提示并退出（用法自保护）
LOG="${1:?用法: bash 03_scraper.sh <serve日志路径> [间隔秒]}"
INTERVAL="${2:-2}"    # 第二个参数可选，默认 2 秒

SMI_TSV="scraper_gpusmi.tsv"
TAILS="scraper_logtail.txt"

# 表头（一次性写入）
# [[ ! -f xxx ]]：文件不存在时才写表头——重复运行不会把表头插到中间
if [[ ! -f "$SMI_TSV" ]]; then
  # \t 是制表符；TSV 的第一行是列名，之后每行一次采样
  printf 'ts\tmem_used_MiB\tmem_total_MiB\tgpu_util_pct\n' >"$SMI_TSV"
fi
: >"$TAILS"      # 「:>文件」= 清空文件内容（每次重新运行日志尾巴从头记）

echo "[scraper] 目标日志=$LOG 间隔=${INTERVAL}s  Ctrl-C 停止"
# trap：捕获 Ctrl-C（INT）/终止（TERM）信号，退出前留一句去向提示，不让进程死得不明不白
trap 'echo "[scraper] 结束：$SMI_TSV / $TAILS"' INT TERM

while true; do
  TS=$(date +%H:%M:%S)
  # nvidia-smi 的 query-gpu 输出最稳；机器无 GPU 时打印占位行（Mac 空跑验证用）
  if command -v nvidia-smi >/dev/null 2>&1; then
    # --query-gpu=...：只取需要的三个指标；csv,noheader,nounits = 纯数字好解析
    # read -r A B C < <(...)：把三个数拆进三个变量；head -1 只取第一张卡
    read -r USED TOTAL UTIL < <(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | head -1)
    printf '%s\t%s\t%s\t%s\n' "$TS" "$USED" "$TOTAL" "$UTIL" >>"$SMI_TSV"   # >> = 追加写入
  else
    printf '%s\t na\t na\t na\n' "$TS" >>"$SMI_TSV"   # Mac 上没 nvidia-smi：写 na 占位
  fi
  # 同时记日志尾巴：每轮附上时间戳分隔线 + 最新 5 行日志
  # （回看时能把「日志事件」和「GPU 状态」按时间对上号）
  {
    echo "===== $TS ====="
    tail -n 5 "$LOG" 2>/dev/null || echo "(日志不存在)"   # 2>/dev/null：读不到也别炸
  } >>"$TAILS"
  sleep "$INTERVAL"
done

# TODO(you)（Mac 预习）：
#  1. bash -n 03_scraper.sh 验证语法；
#  2. 无参数运行确认 usage 报错正常（预期：Usage/用法 提示 + 非零退出）；
#  3. S2 现场：实验 C（preemption）时把间隔调 1s——被抢/重算的窗口很短，
#     2s 可能正好跳过去。现象要同时在 serve.log 里 grep PREEMPT 佐证。
