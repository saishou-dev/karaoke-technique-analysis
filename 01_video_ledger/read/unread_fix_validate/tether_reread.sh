#!/bin/bash
# tether_reread.sh ＝ Windows 側の番犬（job_watchdog.py --register reread_v08f --pattern "v3_fullscan_all"）から見える「綱」（2026-10-31）
#   番犬は Windows のプロセスしか見ない（WSL 内の python は見えない）。そこで、コマンドラインに v3_fullscan_all を含む
#   wsl.exe をこの綱として Windows 側に残し、WSL 内の走査（v3_fullscan_all.py）が1本でも生きている間だけ待たせる。
#   走査が全部終われば綱も終わる（＝番犬は「プロセス終了」または完了行で監視を終える）。
#   呼び方（Windows から WMI で分離起動）：wsl.exe -u root -e bash <このファイル> v3_fullscan_all
#   番犬が停滞と判定してこの wsl.exe を止めた時、HUP/TERM が届けば走査も止める（届かない止め方では止まらない＝最善努力）。
O=/mnt/g/研究アーカイブ/成果物/動画解析/reread_v08f_20261031
P='python -u v3_fullscan_al[l].py'
trap 'pkill -TERM -f "$P"; echo "[$(date +"%F %T")] tether: 停止の合図（HUP/TERM）を受けて走査を止めた" >> "$O/progress.log"; exit 143' HUP TERM INT
echo "[$(date +'%F %T')] tether start pid=$$" >> "$O/progress.log"
sleep 30
while pgrep -f "$P" > /dev/null; do sleep 30; done
echo "[$(date +'%F %T')] tether end（走査のプロセスが無くなった）" >> "$O/progress.log"
