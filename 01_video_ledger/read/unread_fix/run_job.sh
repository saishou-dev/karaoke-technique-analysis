#!/bin/bash
# 使い方: run_job.sh <ログ名> <python スクリプト> [引数…]（研究用 cgroup に自分を入れてから nice 10 で実行）
D=/mnt/g/研究アーカイブ/成果物/動画解析/unread_fix_20261030
LOG=$1; shift
echo $$ > /sys/fs/cgroup/research/cgroup.procs 2>/dev/null || echo "cgroup: 入れず（権限なし）" >> "$D/$LOG"
exec su toshiaki -c "source ~/ocr_compare_20261018/env.sh; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4; cd '$D'; nice -n 10 ~/ocr_compare_20261018/.venv312/bin/python -u $* >> '$D/$LOG' 2>&1"
