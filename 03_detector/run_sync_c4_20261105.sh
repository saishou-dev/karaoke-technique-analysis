#!/bin/bash
# run_sync_c4_20261105.sh ＝ run_job_c3_5tech_20261105.sh の写し（2026-11-05・c4-compare）。短い推論・採点を前で（待って）回す用。
#   root で呼ぶ：wsl -u root -e bash run_sync_c4_20261105.sh <python スクリプト> [引数…]
#   cgroup research（無ければ作る・cpu.max 800000/100000）に自分を入れてから toshiaki・nice 10 で実行。出力はそのまま標準出力へ。
D=/mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/scripts
if [ ! -d /sys/fs/cgroup/research ]; then mkdir /sys/fs/cgroup/research; fi
echo "800000 100000" > /sys/fs/cgroup/research/cpu.max
echo $$ > /sys/fs/cgroup/research/cgroup.procs 2>/dev/null || echo "cgroup: 入れず（権限なし）"
echo "cgroup=$(cat /proc/$$/cgroup) cpu.max=$(cat /sys/fs/cgroup/research/cpu.max)"
exec su toshiaki -c "source ~/ocr_compare_20261018/env.sh; export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8; cd '$D'; nice -n 10 ~/ocr_compare_20261018/.venv312/bin/python -u $*"
