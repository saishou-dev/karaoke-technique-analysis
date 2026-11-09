#!/bin/bash
# tether_delay3ways_20261109.sh ＝ tether_fall4tech_20261109.sh の写し（2026-11-09・delay-3ways）。Windows 側から見える「綱」。
#   呼び方（Windows から分離起動）：wsl.exe -u root -e bash <このファイル>
#   HUP/TERM が届けば STOP_REQUEST を置く＝連結ジョブが次の心拍（60秒以内）で自分の学習を止めて終わる。
O=/mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/out/delay_3ways_20261109
P='run_delay3ways_2026110[9].py'
trap 'date +"%F %T" > "$O/STOP_REQUEST"; echo "[$(date +"%F %T")] tether: 停止の合図（HUP/TERM）＝STOP_REQUEST を置いた" >> "$O/delay3ways_chain_20261109.log"; exit 143' HUP TERM INT
echo "[$(date +'%F %T')] tether start pid=$$" >> "$O/delay3ways_chain_20261109.log"
sleep 30
while pgrep -f "$P" > /dev/null; do sleep 30; done
echo "[$(date +'%F %T')] tether end（連結ジョブのプロセスが無くなった）" >> "$O/delay3ways_chain_20261109.log"
