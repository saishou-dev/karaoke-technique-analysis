#!/bin/bash
# 読取器の候補版を比較するための汎用ランナー（2026-10-15）。run_read_v08e_20261015.sh の一般化。
#  (1) 読取器＝KV_READER（~/karaoke-vision 内のファイル名）  (2) 出力先＝results_wao_<S>_<KV_OUTSUF>（既存の results_wao_<S> は上書きしない）
#  (3) ROI＝既存 results_wao_<S>/<video>/roi.json をそのまま使う（較正は再実行しない＝読取器の差だけを見る）
# 使い方: KV_SESSION=20261014 KV_READER=read_counters_wao_v09_20261015.py KV_OUTSUF=v09 KV_TARGET=30 bash run_read_variant_20261015.sh <runner_id 0-3>
export LANG=C.UTF-8
S="${KV_SESSION:-20261014}"
READER="${KV_READER:-read_counters_wao_v09_20261015.py}"
SUF="${KV_OUTSUF:-v09}"
ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"; KV=~/karaoke-vision
RID=$1; CORES=$((RID*3+4))-$((RID*3+6))
SRCRES="$ROOT/results_wao_$S"; RES="$ROOT/results_wao_${S}_${SUF}"; LOG="$RES/_runner.log"; mkdir -p "$RES"
TARGET="${KV_TARGET:-0}"
if [ "$TARGET" -le 0 ]; then TARGET=$(( $(wc -l < "$ROOT/dataset/band_probe_v2_$S.csv") - 1 )); fi
idle=0; CLAIM=/tmp/claims_${S}_${SUF}; mkdir -p "$CLAIM"
echo "[$(date +%H:%M:%S)] runner$RID start target=$TARGET cores=$CORES reader=$READER out=$RES" >> "$LOG"
while true; do
  did=0
  for band in "$ROOT"/bands_wao/*_${S}_*.mp4; do case "$band" in *.part.mp4) continue;; esac
    [ -f "$band" ] || continue
    name=$(basename "$band" .mp4); out="$RES/$name"
    [ -f "$out/counter_events.csv" ] && continue
    mkdir "$CLAIM/$name" 2>/dev/null || continue
    mkdir -p "$out"; did=1
    roi="$ROOT/templates/wao_20260816/roi_wao.json"; [ -f "$SRCRES/$name/roi.json" ] && roi="$SRCRES/$name/roi.json"
    cp "$roi" "$out/roi.json" 2>/dev/null
    echo "[$(date +%H:%M:%S)] runner$RID START $name" >> "$LOG"
    taskset -c $CORES $KV/.venv/bin/python $KV/$READER "$band" "$out" "$ROOT/templates/wao_20260816/cluster_reps_wao_ext.npy" "$roi" > "$out/read_log.txt" 2>&1
    echo "[$(date +%H:%M:%S)] runner$RID END $name rc=$?" >> "$LOG"
    true
  done
  done_n=$(ls "$RES"/*/counter_events.csv 2>/dev/null | wc -l)
  [ "$done_n" -ge "$TARGET" ] && break
  if [ $did -eq 0 ]; then idle=$((idle+1)); [ $idle -ge 20 ] && break; sleep 30; else idle=0; fi
done
echo "[$(date +%H:%M:%S)] runner$RID DONE (results=$done_n)" >> "$LOG"
