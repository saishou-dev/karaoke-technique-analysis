#!/bin/bash
# v1_pipeline.sh ＝ 検証1を通しで回す（9/24 と 10/6 の入力作り→採点）。toshiaki で実行。
D=/mnt/g/研究アーカイブ/成果物/動画解析/unread_fix_validate_20261030
PY=~/ocr_compare_20261018/.venv312/bin/python
source ~/ocr_compare_20261018/env.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$D"
echo "[$(date +%T)] extract 20260924"
nice -n 10 $PY -u v1_extract.py --out "$D/work_20260924" --stride 8 --session 20260924 --results results_wao_20260924 --g1 G1_events_20260924.csv --verify verify_20260924.csv || echo "rc=$? extract 0924"
echo "[$(date +%T)] extract 20261006"
nice -n 10 $PY -u v1_extract.py --out "$D/work_20261006" --stride 8 --session 20261006 --results results_wao_20261006 --g1 G1_events_20261006_wao.csv --verify verify_20261006_wao.csv || echo "rc=$? extract 1006"
echo "[$(date +%T)] score 20260924"
nice -n 10 $PY -u v1_score.py --work "$D/work_20260924" --verify verify_20260924.csv --g1 G1_events_20260924.csv --tag 20260924 --out "$D/result" || echo "rc=$? score 0924"
echo "[$(date +%T)] score 20261006"
nice -n 10 $PY -u v1_score.py --work "$D/work_20261006" --verify verify_20261006_wao.csv --g1 G1_events_20261006_wao.csv --tag 20261006 --out "$D/result" || echo "rc=$? score 1006"
echo "[$(date +%T)] V1 PIPELINE DONE"
