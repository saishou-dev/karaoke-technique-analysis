#!/bin/bash
# watchdog_scan.sh ＝ 検証2の全フレーム走査の番犬（WSL 内・2分ごと・記録のみ）
#  対象＝コマンドラインに v2_fullscan.py を含む python（自分のジョブだけ）。CPU 時間が STALL_MIN 分ふえなければ
#  停滞として WATCHDOG_STALLED.txt と受信箱へ記録して終了する（プロセスの停止はしない＝停止はメイン側の判断）。
#  出力の SCAN_DONE.txt か SCAN_STOPPED.txt を見たら終了する。
set -u
D=/mnt/g/研究アーカイブ/成果物/動画解析/unread_fix_validate_20261030
OUT="$D/result/fullscan"
INBOX="/mnt/c/Users/conan/hisho/routine_inbox"
STALL_MIN="${1:-10}"
HB="$D/watchdog_heartbeat.txt"
last_cpu=""; last_change=$(date +%s)
while true; do
  date +"[%Y-%m-%d %H:%M:%S] tick" > "$HB"
  if [ -f "$OUT/SCAN_DONE.txt" ] || [ -f "$OUT/SCAN_STOPPED.txt" ]; then echo "done - exit" >> "$HB"; exit 0; fi
  p=$(pgrep -f "^/home/toshiaki/ocr_compare_20261018/.venv312/bin/python -u v2_fullsca[n].py" | head -1)
  now=$(date +%s)
  if [ -n "$p" ]; then
    cpu=$(awk '{print $14+$15}' "/proc/$p/stat" 2>/dev/null || echo "")
    echo "pid=$p cpu=$cpu" >> "$HB"
    if [ "$cpu" != "$last_cpu" ]; then last_cpu="$cpu"; last_change=$now
    elif [ $(( (now - last_change) / 60 )) -ge "$STALL_MIN" ]; then
      msg="[$(date +%F' '%H:%M:%S)] STALLED pid=$p 全フレーム走査の CPU 時間が ${STALL_MIN}分 ふえていない（停止はしていない）"
      echo "$msg" >> "$D/WATCHDOG_STALLED.txt"
      printf '<!-- unread -->\n# 停滞検知（見本追加の検証・全フレーム走査）\n\n%s\n\nログ＝%s/v2_fullscan.log／再開＝launch_scan.sh（済みの曲は飛ばす）\n' "$msg" "$D" > "$INBOX/$(date +%Y-%m-%d_%H%M)_unread-validate_stalled.md"
      exit 1
    fi
  else
    echo "no target" >> "$HB"
    if [ $(( (now - last_change) / 60 )) -ge "$STALL_MIN" ]; then echo "no target ${STALL_MIN}min - exit" >> "$HB"; exit 0; fi
  fi
  sleep 120
done
