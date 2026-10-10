#!/bin/bash
# 読取ランナー v2（2026-09-12）。使い方: bash run_read_20260911.sh <runner_id 0-3>
# taskset 3コア固定・mkdir先取りで重複回避・完成済みはスキップ・クロップが増えるのを60秒ごとに待って全30本まで回す
export LANG=C.UTF-8
ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"; KV=~/karaoke-vision
RID=$1; CORES=$((RID*3+4))-$((RID*3+6)); LOG="$ROOT/results_wao_20260911/_runner.log"
TARGET=30; idle=0; CLAIM=/tmp/claims_20260911; mkdir -p "$CLAIM"
while true; do
  did=0
  for band in "$ROOT"/bands_wao/*_20260911_*.mp4; do case "$band" in *.part.mp4) continue;; esac
    name=$(basename "$band" .mp4); out="$ROOT/results_wao_20260911/$name"
    [ -f "$out/counter_events.csv" ] && continue
    mkdir "$CLAIM/$name" 2>/dev/null || continue
    mkdir -p "$out"; did=1
    echo "[$(date +%H:%M:%S)] runner$RID START $name" >> "$LOG"
    taskset -c $CORES $KV/.venv/bin/python $KV/read_counters_wao.py "$band" "$out" "$ROOT/templates/wao_20260816/cluster_reps_wao_ext.npy" "$ROOT/templates/wao_20260816/roi_wao.json" > "$out/read_log.txt" 2>&1
    echo "[$(date +%H:%M:%S)] runner$RID END $name rc=$?" >> "$LOG"
    true
  done
  done_n=$(ls "$ROOT"/results_wao_20260911/*/counter_events.csv 2>/dev/null | wc -l)
  [ "$done_n" -ge "$TARGET" ] && break
  if [ $did -eq 0 ]; then idle=$((idle+1)); [ $idle -ge 90 ] && break; sleep 60; else idle=0; fi
done
echo "[$(date +%H:%M:%S)] runner$RID DONE (results=$done_n)" >> "$LOG"
