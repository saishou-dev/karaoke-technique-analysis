#!/bin/bash
# 読取ランナー DAM Ai 版（2026-09-15・run_read_20260915.sh の派生）。使い方: KV_SESSION=20260915 KV_TARGET=30 bash run_read_20260915ai.sh <runner_id 0-3>
# 読取器＝read_counters_ai_20260915ai.py（WSL read_counters.py v0.8 の複製・4ROI・ラベル内蔵）＋ templates/ai_20260915ai/cluster_reps.npy（40テンプレ）
# 帯動画＝<base>/bands_ai/*_<session>_*.mp4（.part は除外・ゲート落ちは bands_ai/_gatefail_<session>/ にあるので拾わない）
# KV_AI_BASE（WSLパス）を与えると bands_ai／results_ai_<S> をその配下で探す（セルフテスト用）
export LANG=C.UTF-8
S="${KV_SESSION:-20260915}"
ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"; KV=~/karaoke-vision; BASE="${KV_AI_BASE:-$ROOT}"
RID=$1; CORES=$((RID*3+4))-$((RID*3+6)); RES="$BASE/results_ai_$S"; LOG="$RES/_runner.log"; mkdir -p "$RES"
REPS="$ROOT/templates/ai_20260915ai/cluster_reps.npy"
TARGET="${KV_TARGET:-0}"
if [ "$TARGET" -le 0 ]; then
  PB="$BASE/band_probe_ai_$S.csv"; [ "$BASE" = "$ROOT" ] && PB="$ROOT/dataset/band_probe_ai_$S.csv"
  TARGET=$(( $(wc -l < "$PB") - 1 ))
fi
idle=0; CLAIM=/tmp/claims_ai_$S; mkdir -p "$CLAIM"
echo "[$(date +%H:%M:%S)] runner$RID start target=$TARGET cores=$CORES base=$BASE" >> "$LOG"
while true; do
  did=0
  for band in "$BASE"/bands_ai/*_${S}_*.mp4; do case "$band" in *.part.mp4) continue;; esac
    [ -f "$band" ] || continue
    name=$(basename "$band" .mp4); out="$RES/$name"
    [ -f "$out/counter_events.csv" ] && continue
    mkdir "$CLAIM/$name" 2>/dev/null || continue
    mkdir -p "$out"; did=1
    echo "[$(date +%H:%M:%S)] runner$RID START $name" >> "$LOG"
    taskset -c $CORES $KV/.venv/bin/python $KV/read_counters_ai_20260915ai.py "$band" "$out" "$REPS" > "$out/read_log.txt" 2>&1
    echo "[$(date +%H:%M:%S)] runner$RID END $name rc=$?" >> "$LOG"
    true
  done
  done_n=$(ls "$RES"/*/counter_events.csv 2>/dev/null | wc -l)
  [ "$done_n" -ge "$TARGET" ] && break
  if [ $did -eq 0 ]; then idle=$((idle+1)); [ $idle -ge 90 ] && break; sleep 60; else idle=0; fi
done
echo "[$(date +%H:%M:%S)] runner$RID DONE (results=$done_n)" >> "$LOG"
