#!/bin/bash
# tether_c2_shift2_20261105.sh ＝ tether_c2_shift_20261104.sh と同型（2026-11-05）。Windows 側の番犬から見える「綱」。
#   呼び方（Windows から分離起動）：wsl.exe -u root -e bash <このファイル>
#   HUP/TERM が届けば STOP_REQUEST を置く＝連結ジョブが次の心拍（60秒以内）で自分の学習を止めて終わる。
O=/mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/out/c2_shift2_20261105
P='run_c2_shift2_2026110[5].py'
trap 'date +"%F %T" > "$O/STOP_REQUEST"; echo "[$(date +"%F %T")] tether: 停止の合図（HUP/TERM）＝STOP_REQUEST を置いた" >> "$O/c2_shift2_chain_20261105.log"; exit 143' HUP TERM INT
echo "[$(date +'%F %T')] tether start pid=$$" >> "$O/c2_shift2_chain_20261105.log"
sleep 30
while pgrep -f "$P" > /dev/null; do sleep 30; done
echo "[$(date +'%F %T')] tether end（連結ジョブのプロセスが無くなった）" >> "$O/c2_shift2_chain_20261105.log"
