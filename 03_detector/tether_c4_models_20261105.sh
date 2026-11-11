#!/bin/bash
# tether_c4_models_20261105.sh ＝ tether_c4_compare_20261105.sh の写し（2026-11-05・c4-compare 手順6・手順2〜4 と手順6 のどちらかの連結ジョブが居る間つながる）。Windows 側から見える「綱」。
#   呼び方（Windows から分離起動）：wsl.exe -u root -e bash <このファイル>
#   HUP/TERM が届けば STOP_REQUEST を置く＝連結ジョブが次の心拍（60秒以内）で自分の学習を止めて終わる。
O=/mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/out/c4_compare_20261105
P='run_c4_(compare|models)_2026110[5].py'
trap 'date +"%F %T" > "$O/STOP_REQUEST"; echo "[$(date +"%F %T")] tether: 停止の合図（HUP/TERM）＝STOP_REQUEST を置いた" >> "$O/c4_models_chain_20261105.log"; exit 143' HUP TERM INT
echo "[$(date +'%F %T')] tether start pid=$$" >> "$O/c4_models_chain_20261105.log"
sleep 30
while pgrep -f "$P" > /dev/null; do sleep 30; done
echo "[$(date +'%F %T')] tether end（連結ジョブのプロセスが無くなった）" >> "$O/c4_models_chain_20261105.log"
