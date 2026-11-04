#!/bin/bash
# tether_c2_shift_20261104.sh ＝ Windows 側の番犬（job_watchdog.py）から見える「綱」（2026-11-04・tether_reread.sh と同型）
#   番犬は Windows のプロセスしか見ない。コマンドラインに tether_c2_shift を含む wsl.exe を Windows 側に残し、
#   WSL 内の連結ジョブ（run_c2_shift_20261104.py）が生きている間だけ待たせる。終われば綱も終わる。
#   呼び方（Windows から分離起動）：wsl.exe -u root -e bash <このファイル>
#   番犬が綱を止めた時、HUP/TERM が届けば STOP_REQUEST を置く＝連結ジョブが次の心拍（60秒以内）で自分の学習を止めて終わる。
O=/mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/out/c2_shift_20261104
P='python -u run_c2_shift_2026110[4].py'
trap 'date +"%F %T" > "$O/STOP_REQUEST"; echo "[$(date +"%F %T")] tether: 停止の合図（HUP/TERM）＝STOP_REQUEST を置いた" >> "$O/c2_shift_chain_20261104.log"; exit 143' HUP TERM INT
echo "[$(date +'%F %T')] tether start pid=$$" >> "$O/c2_shift_chain_20261104.log"
sleep 30
while pgrep -f "$P" > /dev/null; do sleep 30; done
echo "[$(date +'%F %T')] tether end（連結ジョブのプロセスが無くなった）" >> "$O/c2_shift_chain_20261104.log"
