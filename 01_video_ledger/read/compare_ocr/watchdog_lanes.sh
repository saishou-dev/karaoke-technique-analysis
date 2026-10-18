#!/bin/bash
# watchdog_lanes.sh ＝ WSL 内で走る比較実験の番犬（2026-10-18）
#  既存の番犬 hisho\ops\tools\job_watchdog.py は Windows のプロセスしか見られない（既知の弱点）ため、
#  WSL 側のジョブにはこれを同時に分離起動する。
#  2分ごとに run_candidates プロセスの CPU 時間を見て、--stall-min 分ふえなければ停滞とみなし、
#  対象を止めて WATCHDOG_STALLED.txt と受信箱（routine_inbox）へ記録する。ダイアログは一切出さない。
# 使い方: setsid nohup bash watchdog_lanes.sh > /dev/null 2>&1 &
set -u
BASE="$HOME/ocr_compare_20261018"
INBOX="/mnt/c/Users/conan/hisho/routine_inbox"
STALL_MIN="${1:-10}"
HB="$BASE/watchdog_heartbeat.txt"
declare -A last_cpu
declare -A last_change
while true; do
  date +"[%Y-%m-%d %H:%M:%S] watchdog tick" > "$HB"
  pids=$(pgrep -f "run_candidates.py" || true)
  if [ -z "$pids" ]; then
    echo "no target process" >> "$HB"
    if [ -f "$BASE/LANE_A_DONE" ] && [ -f "$BASE/LANE_B_DONE" ] && [ -f "$BASE/LANE_C_DONE" ]; then
      echo "all lanes done - watchdog exits" >> "$HB"; exit 0
    fi
  fi
  now=$(date +%s)
  for p in $pids; do
    cpu=$(awk '{print $14+$15}' "/proc/$p/stat" 2>/dev/null || echo "")
    [ -z "$cpu" ] && continue
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" | sed 's/.*--cand //;s/ .*//')
    echo "pid=$p cand=$cmd cpu=$cpu" >> "$HB"
    prev="${last_cpu[$p]:-}"
    if [ "$prev" != "$cpu" ]; then
      last_cpu[$p]="$cpu"; last_change[$p]="$now"
    else
      since=$(( (now - ${last_change[$p]:-$now}) / 60 ))
      if [ "$since" -ge "$STALL_MIN" ]; then
        msg="[$(date +%F' '%H:%M:%S)] STALLED pid=$p cand=$cmd cpu時間が ${since}分 ふえていない → 停止した"
        echo "$msg" >> "$BASE/WATCHDOG_STALLED.txt"
        kill -9 "$p" 2>/dev/null
        {
          echo "<!-- unread -->"
          echo "# 停滞検知（OCR比較実験・番犬）"
          echo ""
          echo "$msg"
          echo ""
          echo "ログ＝$BASE/lane_*.log ／ 再開＝bash run_all.sh <A|B|C>（済みの候補は飛ばす）"
        } > "$INBOX/$(date +%Y-%m-%d_%H%M)_ocr_compare_stalled.md"
        unset last_cpu[$p]; unset last_change[$p]
      fi
    fi
  done
  sleep 120
done
