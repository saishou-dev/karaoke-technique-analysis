#!/bin/bash
# run_job_delay3ways_20261109.sh ＝ run_job_fall4tech_20261109.sh の写し（出力先だけ delay_3ways_20261109）。cgroup research に自分を入れてから toshiaki・nice 10 で実行
# 使い方: run_job_delay3ways_20261109.sh <ログ名（out/delay_3ways_20261109 からの相対）> <python スクリプト> [引数…]
D=/mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/scripts
O=/mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/out/delay_3ways_20261109
LOG=$1; shift
echo $$ > /sys/fs/cgroup/research/cgroup.procs 2>/dev/null || echo "cgroup: 入れず（権限なし）" >> "$O/$LOG"
exec su toshiaki -c "source ~/ocr_compare_20261018/env.sh; export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8; cd '$D'; nice -n 10 ~/ocr_compare_20261018/.venv312/bin/python -u $* >> '$O/$LOG' 2>&1"
