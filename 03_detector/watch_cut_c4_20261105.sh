#!/bin/bash
# watch_cut_c4_20261105.sh（2026-11-06 Opus・c4-compare 追補6）＝Windows の Git Bash で回す見張り。読むのと写すだけ（プロセスには触らない）。
# 連結ジョブのログに「!!! <tag> 全体の締め切り」か「!!! <tag> 時間切れ」が出たら、その本の model.pt・norm.npy・ログを
# out/c4_compare_20261105/partial_c4/<tag>/ へすぐ写す（締め切りで切れた本は 18:06 の列が同じ所で回し直す＝重みが上書きされる前に残す）。
# 止まる＝2026-11-06 12:20 を過ぎ、写す物が無くなった時。
O=/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/out/c4_compare_20261105
P=$O/partial_c4
mkdir -p "$P"
echo "[$(date '+%F %T')] watch start" >> "$P/watch.log"
while true; do
  for L in "$O/c4_models_chain_20261105.log" "$O/c4_compare_chain_20261105.log"; do
    grep -oE "!!! c4_[a-z0-9_]+ (全体の締め切り|時間切れ)" "$L" 2>/dev/null | awk '{print $2}' | sort -u | while read T; do
      if [ ! -f "$P/$T/model.pt" ] && [ -f "$O/$T/model.pt" ]; then
        mkdir -p "$P/$T"
        cp -p "$O/$T/model.pt" "$O/$T/norm.npy" "$P/$T/" 2>/dev/null
        cp -p "$O/$T.log" "$P/$T/train.log" 2>/dev/null
        [ -f "$O/$T/last.pt" ] && cp -p "$O/$T/last.pt" "$P/$T/"
        echo "[$(date '+%F %T')] 写した＝$T（$(grep -oE "!!! $T [^＝]+" "$L" | head -1)）" >> "$P/watch.log"
      fi
    done
  done
  now=$(date +%s); stop=$(date -d '2026-11-06 12:20:00' +%s 2>/dev/null || echo 0)
  if [ "$stop" -gt 0 ] && [ "$now" -gt "$stop" ]; then echo "[$(date '+%F %T')] watch end" >> "$P/watch.log"; break; fi
  sleep 20
done
