#!/bin/bash
# launch_reread.sh ＝ 読取器 v0.8f で過去の全収録を読み直す走査（v3_fullscan_all.py）を分離起動する（2026-10-31）
#   root で呼ぶ：wsl -u root -e bash launch_reread.sh [本数＝既定3]
#   launch_scan.sh と同型＝setsid nohup・run_job.sh 経由（cgroup research＋nice 10・toshiaki で実行）。
#   cgroup research は WSL の再起動で消えるので、無ければ作り直して CPU を8コア相当（cpu.max 800000/100000）に制限する。
#   出力＝/mnt/g/研究アーカイブ/成果物/動画解析/reread_v08f_20261031/（fullscan/<収録>/<曲>.npz・progress.log・logs/w<i>.log）
#   再開＝同じコマンドをもう一度（npz がある曲は飛ばす）。
D=/mnt/g/研究アーカイブ/成果物/動画解析/unread_fix_validate_20261030
O=/mnt/g/研究アーカイブ/成果物/動画解析/reread_v08f_20261031
N=${1:-3}
RUN=$(date +%Y%m%d_%H%M%S)
mkdir -p "$O/logs" "$O/fullscan"
chown toshiaki "$O" "$O/logs" "$O/fullscan" 2>/dev/null
if [ ! -d /sys/fs/cgroup/research ]; then mkdir /sys/fs/cgroup/research; fi
echo "800000 100000" > /sys/fs/cgroup/research/cpu.max
echo "[$(date +'%F %T')] launch run=$RUN workers=$N cgroup cpu.max=$(cat /sys/fs/cgroup/research/cpu.max)" >> "$O/progress.log"
for i in $(seq 0 $((N-1))); do
  setsid nohup bash "$D/run_job.sh" ../reread_v08f_20261031/logs/w$i.log v3_fullscan_all.py --worker $i --workers $N --run $RUN < /dev/null > /dev/null 2>&1 &
  sleep 3
done
sleep 5
# su（pam_systemd）が子を user.slice のセッションへ移すため、run_job.sh の cgroup 入りは効かない（10/31 実測）。
# 起動後に走査のプロセスを root で research へ移し直す（cgroup.procs はスレッドごとまとめて移る）。
for p in $(pgrep -f "python -u v3_fullscan_al[l].py" | sort -u); do echo $p > /sys/fs/cgroup/research/cgroup.procs; done
echo "launched run=$RUN workers=$N"
pgrep -fa "python -u v3_fullscan_al[l].py" | grep -v "^.* su "
cat /sys/fs/cgroup/research/cgroup.procs | tr '\n' ' '; echo
