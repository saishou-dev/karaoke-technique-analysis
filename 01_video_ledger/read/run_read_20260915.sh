#!/bin/bash
# 読取ランナー v3（2026-09-15版・run_read_20260911.sh の一般化・9/15 23:2x calib の roi.json 対応）。使い方: KV_SESSION=20260915 KV_TARGET=30 bash run_read_20260915.sh <runner_id 0-3>
# taskset 3コア固定・mkdir先取りで重複回避・完成済みはスキップ・クロップが増えるのを60秒ごとに待って TARGET 本まで回す
# 帯動画＝bands_wao/*_<session>_*.mp4（.part は除外・ゲート落ちは bands_wao/_gatefail_<session>/ にあるので拾わない）
export LANG=C.UTF-8
S="${KV_SESSION:-20260915}"
ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"; KV=~/karaoke-vision
RID=$1; CORES=$((RID*3+4))-$((RID*3+6)); RES="$ROOT/results_wao_$S"; LOG="$RES/_runner.log"; mkdir -p "$RES"
TARGET="${KV_TARGET:-0}"
if [ "$TARGET" -le 0 ]; then TARGET=$(( $(wc -l < "$ROOT/dataset/band_probe_v2_$S.csv") - 1 )); fi
idle=0; CLAIM=/tmp/claims_$S; mkdir -p "$CLAIM"
echo "[$(date +%H:%M:%S)] runner$RID start target=$TARGET cores=$CORES" >> "$LOG"
while true; do
  did=0
  for band in "$ROOT"/bands_wao/*_${S}_*.mp4; do case "$band" in *.part.mp4) continue;; esac
    [ -f "$band" ] || continue
    name=$(basename "$band" .mp4); out="$RES/$name"
    [ -f "$out/counter_events.csv" ] && continue
    mkdir "$CLAIM/$name" 2>/dev/null || continue
    mkdir -p "$out"; did=1
    # ROI＝calib 工程の曲別 roi.json（results_wao_<S>/<video>/roi.json・2026-09-15 追加）。無ければ既定 roi_wao.json
    roi="$ROOT/templates/wao_20260816/roi_wao.json"; [ -f "$out/roi.json" ] && roi="$out/roi.json"
    echo "[$(date +%H:%M:%S)] runner$RID START $name roi=$(basename "$(dirname "$roi")")/$(basename "$roi")" >> "$LOG"
    taskset -c $CORES $KV/.venv/bin/python $KV/read_counters_wao.py "$band" "$out" "$ROOT/templates/wao_20260816/cluster_reps_wao_ext.npy" "$roi" > "$out/read_log.txt" 2>&1
    echo "[$(date +%H:%M:%S)] runner$RID END $name rc=$?" >> "$LOG"
    true
  done
  done_n=$(ls "$RES"/*/counter_events.csv 2>/dev/null | wc -l)
  [ "$done_n" -ge "$TARGET" ] && break
  if [ $did -eq 0 ]; then idle=$((idle+1)); [ $idle -ge 90 ] && break; sleep 60; else idle=0; fi
done
echo "[$(date +%H:%M:%S)] runner$RID DONE (results=$done_n)" >> "$LOG"
