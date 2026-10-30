#!/bin/bash
# launch_scan.sh ＝ 検証2の全フレーム走査と番犬を分離起動する（root で呼ぶ：wsl -u root -e bash launch_scan.sh）
D=/mnt/g/研究アーカイブ/成果物/動画解析/unread_fix_validate_20261030
setsid nohup bash "$D/run_job.sh" v2_fullscan.log v2_fullscan.py --out "$D/result/fullscan" < /dev/null > /dev/null 2>&1 &
sleep 2
setsid nohup su toshiaki -c "bash '$D/watchdog_scan.sh' 10" < /dev/null > /dev/null 2>&1 &
echo launched
