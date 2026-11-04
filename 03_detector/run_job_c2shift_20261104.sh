#!/bin/bash
# run_job_c2shift_20261104.sh ＝ run_job.sh（unread_fix_20261030）と同型。研究用 cgroup に自分を入れてから toshiaki・nice 10 で実行（2026-11-04）
# 使い方: run_job_c2shift_20261104.sh <ログ名（out/c2_shift_20261104 からの相対）> <python スクリプト> [引数…]
D=/mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/scripts
O=/mnt/g/研究アーカイブ/成果物/動画解析/g3v2_20260928/out/c2_shift_20261104
LOG=$1; shift
echo $$ > /sys/fs/cgroup/research/cgroup.procs 2>/dev/null || echo "cgroup: 入れず（権限なし）" >> "$O/$LOG"
exec su toshiaki -c "source ~/ocr_compare_20261018/env.sh; export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8; cd '$D'; nice -n 10 ~/ocr_compare_20261018/.venv312/bin/python -u $* >> '$O/$LOG' 2>&1"
