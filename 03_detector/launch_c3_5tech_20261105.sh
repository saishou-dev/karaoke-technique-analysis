#!/bin/bash
# launch_c3_5tech_20261105.sh ＝ launch_c2_shift2_20261105.sh の写し（2026-11-05・c3-5tech）
#   root で呼ぶ：wsl -u root -e bash launch_c3_5tech_20261105.sh
#   cgroup research（無ければ作る・cpu.max 800000/100000＝8コア相当）・setsid nohup・run_job 経由（toshiaki・nice 10）。
#   連結ジョブと学習（子）の両方を root で research へ移し直す。pgrep は今回の名前（run_c3_5tech_20261105・--tag c3_5tech_global_20261105）だけを拾う＝別の解析には触れない。
S=/mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/scripts
O=/mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/out/c3_5tech_global_20261105
SHIM=/home/toshiaki/c3_5tech_20261105
PAT='run_c3_5tech_2026110[5].py|--tag c3_5tech_global_2026110[5]/'
mkdir -p "$O"
rm -f "$O/STOP_REQUEST"
su toshiaki -c "mkdir -p '$SHIM/G:/研究アーカイブ/成果物/動画解析' && ln -sfn /mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928 '$SHIM/G:/研究アーカイブ/成果物/動画解析/g3v2_20260928'"
if [ ! -d /sys/fs/cgroup/research ]; then mkdir /sys/fs/cgroup/research; fi
echo "800000 100000" > /sys/fs/cgroup/research/cpu.max
echo "[$(date +'%F %T')] launch cgroup cpu.max=$(cat /sys/fs/cgroup/research/cpu.max) args=$*" >> "$O/c3_5tech_chain_20261105.log"
setsid nohup bash "$S/run_job_c3_5tech_20261105.sh" c3_5tech_chain_20261105.log run_c3_5tech_20261105.py --shim "$SHIM" "$@" < /dev/null > /dev/null 2>&1 &
for i in 1 2 3 4 5 6; do
  sleep 2
  for p in $(pgrep -f "$PAT" | sort -u); do echo $p > /sys/fs/cgroup/research/cgroup.procs 2>/dev/null; done
done
echo "launched"
for p in $(pgrep -f "$PAT"); do echo "$p $(cat /proc/$p/cgroup) nice=$(ps -o ni= -p $p) $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-140)"; done
