#!/bin/bash
# tether_fall4tech_20261109.sh ＝ tether_c4_compare_20261105.sh の写し（2026-11-09・fall-4tech）。Windows 側から見える「綱」。
#   呼び方（Windows から分離起動）：wsl.exe -u root -e bash <このファイル>
#   HUP/TERM が届けば STOP_REQUEST を置く＝連結ジョブが次の心拍（60秒以内）で自分の学習を止めて終わる。
O=/mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/out/fall_4tech_20261109
P='run_fall4tech_2026110[9].py'
trap 'date +"%F %T" > "$O/STOP_REQUEST"; echo "[$(date +"%F %T")] tether: 停止の合図（HUP/TERM）＝STOP_REQUEST を置いた" >> "$O/fall4tech_chain_20261109.log"; exit 143' HUP TERM INT
echo "[$(date +'%F %T')] tether start pid=$$" >> "$O/fall4tech_chain_20261109.log"
sleep 30
while pgrep -f "$P" > /dev/null; do sleep 30; done
echo "[$(date +'%F %T')] tether end（連結ジョブのプロセスが無くなった）" >> "$O/fall4tech_chain_20261109.log"
