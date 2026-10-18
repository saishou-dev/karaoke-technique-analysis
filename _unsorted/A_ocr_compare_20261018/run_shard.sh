#!/bin/bash
# run_shard.sh ＝ 候補を分割実行する（壁時計を縮めるため。1プロセスは専用4コアのまま）
# 使い方: bash run_shard.sh <cand> <shard> <nshards> <cores e.g. 0-3>
set -u
BASE="$HOME/ocr_compare_20261018"
OUT="/mnt/g/研究アーカイブ/成果物/動画解析/ocr_compare_20261018"
source "$BASE/env.sh"
P="$BASE/.venv312/bin/python"
CAND="$1"; SH="$2"; NS="$3"; CORES="$4"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
LOG="$BASE/shard_${CAND}_${SH}of${NS}.log"
rm -f "$BASE/SHARD_${CAND}_${SH}of${NS}_DONE"
{
  date +"[%H:%M:%S] start $CAND shard $SH/$NS cores=$CORES"
  taskset -c "$CORES" $P "$OUT/run_candidates.py" --work "$OUT/work" --out "$OUT/pred" \
      --cand "$CAND" --cores 4 --shard "$SH" --nshards "$NS"
  date +"[%H:%M:%S] end rc=$?"
} > "$LOG" 2>&1
touch "$BASE/SHARD_${CAND}_${SH}of${NS}_DONE"
