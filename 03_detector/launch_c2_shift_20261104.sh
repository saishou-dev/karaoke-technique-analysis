#!/bin/bash
# launch_c2_shift_20261104.sh ＝ C2 補正の連結ジョブ（run_c2_shift_20261104.py）を分離起動する（2026-11-04・launch_reread.sh と同型）
#   root で呼ぶ：wsl -u root -e bash launch_c2_shift_20261104.sh [--only base|shift]
#   setsid nohup・run_job 経由（cgroup research＋nice 10・toshiaki で実行）。
#   cgroup research は WSL の再起動で消えるので、無ければ作り直して CPU を8コア相当（cpu.max 800000/100000）に制限する。
#   su（pam_systemd）が子を user.slice へ移すため、起動後に連結ジョブを root で research へ移し直す（子の学習は後から生まれるので継ぐ）。
#   学習台本の ROOT（G:/…）を解決させるための記号リンクを toshiaki の作業場に作る（台本は変えない）。
S=/mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/scripts
O=/mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/out/c2_shift_20261104
SHIM=/home/toshiaki/c2shift_20261104
mkdir -p "$O"
rm -f "$O/STOP_REQUEST"
su toshiaki -c "mkdir -p '$SHIM/G:/研究アーカイブ/成果物/動画解析' && ln -sfn /mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928 '$SHIM/G:/研究アーカイブ/成果物/動画解析/g3v2_20260928'"
if [ ! -d /sys/fs/cgroup/research ]; then mkdir /sys/fs/cgroup/research; fi
echo "800000 100000" > /sys/fs/cgroup/research/cpu.max
echo "[$(date +'%F %T')] launch cgroup cpu.max=$(cat /sys/fs/cgroup/research/cpu.max) args=$*" >> "$O/c2_shift_chain_20261104.log"
setsid nohup bash "$S/run_job_c2shift_20261104.sh" c2_shift_chain_20261104.log run_c2_shift_20261104.py --shim "$SHIM" "$@" < /dev/null > /dev/null 2>&1 &
# 連結ジョブは起動直後に1本目の学習（子）を生むので、連結ジョブと学習の両方を移す（11/04 17:47 実測＝子が user.slice に残った）。
for i in 1 2 3 4 5 6; do
  sleep 2
  for p in $(pgrep -f "python -u run_c2_shift_2026110[4].py|train_crnn_refdiff_2026102[3]" | sort -u); do echo $p > /sys/fs/cgroup/research/cgroup.procs 2>/dev/null; done
done
echo "launched"
for p in $(pgrep -f "python -u run_c2_shift_2026110[4].py|train_crnn_refdiff_2026102[3]"); do echo "$p $(cat /proc/$p/cgroup) nice=$(ps -o ni= -p $p) $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-120)"; done
