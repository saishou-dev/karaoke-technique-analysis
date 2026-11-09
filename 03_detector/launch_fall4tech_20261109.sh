#!/bin/bash
# launch_fall4tech_20261109.sh ＝ launch_c4_compare_20261105.sh の写し（2026-11-09・fall-4tech）
#   root で呼ぶ：wsl -u root -e bash launch_fall4tech_20261109.sh --deadline <epoch秒> [--only タグ,…]
#   cgroup research（無ければ作る・cpu.max 800000/100000＝8コア相当）・setsid nohup・run_job 経由（toshiaki・nice 10）。
#   連結ジョブと学習（子）の両方を root で research へ移し直す。pgrep は今回の名前（run_fall4tech_20261109・--tag fall_4tech_20261109）だけを拾う＝別の解析には触れない。
#   学習は1本ずつ順に回る（連結ジョブの中で直列）。2本目以降の子は連結ジョブ（research の中）から生まれる＝cgroup を引き継ぐ。
S=/mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/scripts
O=/mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/out/fall_4tech_20261109
SHIM=/home/toshiaki/fall4tech_20261109
PAT='run_fall4tech_2026110[9].py|--tag fall_4tech_2026110[9]/'
mkdir -p "$O"
rm -f "$O/STOP_REQUEST"
su toshiaki -c "mkdir -p '$SHIM/G:/研究アーカイブ/成果物/動画解析' && ln -sfn /mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928 '$SHIM/G:/研究アーカイブ/成果物/動画解析/g3v2_20260928'"
if [ ! -d /sys/fs/cgroup/research ]; then mkdir /sys/fs/cgroup/research; fi
echo "800000 100000" > /sys/fs/cgroup/research/cpu.max
echo "[$(date +'%F %T')] launch cgroup cpu.max=$(cat /sys/fs/cgroup/research/cpu.max) args=$*" >> "$O/fall4tech_chain_20261109.log"
setsid nohup bash "$S/run_job_fall4tech_20261109.sh" fall4tech_chain_20261109.log run_fall4tech_20261109.py --shim "$SHIM" "$@" < /dev/null > /dev/null 2>&1 &
for i in 1 2 3 4 5 6; do
  sleep 2
  for p in $(pgrep -f "$PAT" | sort -u); do echo $p > /sys/fs/cgroup/research/cgroup.procs 2>/dev/null; done
done
echo "launched"
for p in $(pgrep -f "$PAT"); do echo "$p $(cat /proc/$p/cgroup) nice=$(ps -o ni= -p $p) $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-140)"; done
