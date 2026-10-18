#!/bin/bash
# run_all.sh ＝ 9候補を3レーンで走らせる（2026-10-18）
# 各プロセスを taskset で専用4コアに固定し OMP_NUM_THREADS=4 を揃える＝処理時間を同条件で測るため。
# レーン間はコアを共有しないので、並行して走っても1,000コマあたり秒は比較できる。
# 使い方: bash run_all.sh <lane A|B|C>
set -u
BASE="$HOME/ocr_compare_20261018"
OUT="/mnt/g/研究アーカイブ/成果物/動画解析/ocr_compare_20261018"
source "$BASE/env.sh"
P="$BASE/.venv312/bin/python"
WORK="$OUT/work"
PRED="$OUT/pred"
mkdir -p "$PRED"
LANE="$1"
case "$LANE" in
  A) CORES="0-3";  LIST="paddle_rec" ;;
  B) CORES="4-7";  LIST="easyocr_readtext easyocr_rec_prep easyocr_rec" ;;
  C) CORES="8-11"; LIST="ncc nn_l2 tess_psm8 tess_psm8_raw tess_psm7" ;;
  *) echo "lane must be A|B|C"; exit 2 ;;
esac
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export KV_CORES=4
LOG="$BASE/lane_$LANE.log"
rm -f "$BASE/LANE_${LANE}_DONE"
{
  date +"[%H:%M:%S] lane $LANE start cores=$CORES list=$LIST"
  for c in $LIST; do
    if [ -f "$PRED/pred_$c.csv" ] && [ -f "$PRED/time_$c.json" ]; then
      date +"[%H:%M:%S] skip $c (already done)"; continue
    fi
    date +"[%H:%M:%S] START $c"
    taskset -c $CORES $P "$OUT/run_candidates.py" --work "$WORK" --out "$PRED" --cand "$c" --cores 4
    date +"[%H:%M:%S] END $c rc=$?"
  done
  date +"[%H:%M:%S] lane $LANE done"
} >> "$LOG" 2>&1
touch "$BASE/LANE_${LANE}_DONE"
