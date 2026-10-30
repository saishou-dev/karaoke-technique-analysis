#!/usr/bin/env bash
# run_all_gran.sh = 切り出し粒度の比較を4レーンで走らせる（2026-10-29）
#
# jobs.txt の各行 = 条件|候補|間引き(sub)|分割数(nshards)
#   sub=4  … 2026-10-18 と同じコマ列を4つに1つへ間引いた列（10,794件）。全手法を同じ列で比べる。
#   sub=1  … 全件（43,170件）。見本照合と最近傍だけ全件でも走らせ、10/18 との再現確認に使う。
# 分割（nshards）は1枚ずつ独立の仕事に展開し、4つ並べて走らせる。
# 1プロセス専用4コア（OMP_NUM_THREADS=4）・同時4プロセス＝16コアを使い切る。
# ダイアログを出さない・セッションから切り離して走る（setsid nohup で起動すること）。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/ocr_compare_20261018/.venv312/bin/python"
OUT="$HERE/pred"
FULL="$HERE/pred/_full"
LOG="$HOME/ocr_gran_20261029/logs"
HB="$HOME/ocr_gran_20261029/heartbeat.txt"
mkdir -p "$OUT" "$FULL" "$LOG"
. "$HOME/ocr_compare_20261018/env.sh"

run_one() {
  IFS='|' read -r cond cand sub sh nsh <<< "$1"
  if [ "$nsh" -gt 1 ]; then sfx=".shard${sh}of${nsh}"; else sfx=""; fi
  if [ "$sub" = "1" ]; then o="$FULL"; else o="$OUT"; fi
  tag="${cond}_${cand}_sub${sub}${sfx}"
  if [ -f "$o/pred_${cond}_${cand}${sfx}.csv" ]; then echo "SKIP $tag" >> "$HB"; return 0; fi
  echo "START $tag $(date +%H:%M:%S)" >> "$HB"
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "$PY" "$HERE/run_gran.py" --cond "$cond" --cand "$cand" \
    --slots "$HERE/work/slots.json" --out "$o" \
    --sub "$sub" --shard "$sh" --nshards "$nsh" \
    > "$LOG/${tag}.log" 2>&1
  echo "RC=$? $tag $(date +%H:%M:%S)" >> "$HB"
}
export -f run_one
export PY OUT FULL LOG HB HERE

grep -v '^#' "$HERE/jobs.txt" | grep -v '^$' | awk -F'|' '{
  for (s = 0; s < $4; s++) printf "%s|%s|%s|%d|%s\n", $1, $2, $3, s, $4
}' | xargs -I{} -P 4 bash -c 'run_one "$@"' _ {}
echo "ALL_JOBS_DONE $(date +%F' '%H:%M:%S)" >> "$HB"
