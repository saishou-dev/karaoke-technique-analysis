#!/bin/bash
# 読取ランナー（2026-10-30・close4 用・run_read_variant_20261015.sh の一般化・既存は無変更）。
# ジョブ表 KV_JOBS（TSV：name<TAB>帯動画<TAB>roi.json<TAB>出力ディレクトリ<TAB>読取器ファイル名）を mkdir で先取りして順に読む。
# CPU＝taskset 4コア（runner0=4-7・1=8-11・2=12-15）・nice 10・OMP_NUM_THREADS=4（同時に別の実験が走っているため）。
# 使い方: KV_JOBS=<jobs.tsv> bash run_read_close4_20261030.sh <runner_id 0-2>
export LANG=C.UTF-8 OMP_NUM_THREADS=4
ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"; KV=~/karaoke-vision
RID=$1; CORES=$((RID*4+4))-$((RID*4+7)); JOBS="$KV_JOBS"; CLAIM=/tmp/claims_close4_20261030; mkdir -p "$CLAIM"
LOG="$(dirname "$JOBS")/_runner.log"
echo "[$(date +%H:%M:%S)] runner$RID start cores=$CORES jobs=$JOBS" >> "$LOG"
while IFS=$'\t' read -r name band roi out reader; do
  [ -z "$name" ] && continue
  key=$(echo "$out|$name" | md5sum | cut -c1-16)
  [ -f "$out/$name/counter_events.csv" ] && continue
  mkdir "$CLAIM/$key" 2>/dev/null || continue
  mkdir -p "$out/$name"; cp "$roi" "$out/$name/roi.json" 2>/dev/null
  echo "[$(date +%H:%M:%S)] runner$RID START $name out=$out reader=$reader" >> "$LOG"
  nice -n 10 taskset -c $CORES $KV/.venv/bin/python $KV/$reader "$band" "$out/$name" "$ROOT/templates/wao_20260816/cluster_reps_wao_ext.npy" "$roi" > "$out/$name/read_log.txt" 2>&1
  rc=$?; echo "[$(date +%H:%M:%S)] runner$RID END $name rc=$rc" >> "$LOG"
  [ -f "$out/$name/counter_events.csv" ] || rmdir "$CLAIM/$key" 2>/dev/null
done < "$JOBS"
echo "[$(date +%H:%M:%S)] runner$RID DONE" >> "$LOG"
