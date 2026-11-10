#!/bin/bash
# 読取ランナー v3 の 2026-11-10 版（run_read_20260915.sh の複製・コア割り当てだけ変更・既存は無変更）。元＝（2026-09-15版・run_read_20260911.sh の一般化・9/15 23:2x calib の roi.json 対応）。使い方: KV_SESSION=20260915 KV_TARGET=30 bash run_read_20260915.sh <runner_id 0-3>
# taskset 3コア固定・mkdir先取りで重複回避・完成済みはスキップ・クロップが増えるのを60秒ごとに待って TARGET 本まで回す
# 帯動画＝bands_wao/*_<session>_*.mp4（.part は除外・ゲート落ちは bands_wao/_gatefail_<session>/ にあるので拾わない）
export LANG=C.UTF-8
S="${KV_SESSION:-20260915}"
ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"; KV=~/karaoke-vision
RID=$1
# 2026-11-10 Opus：コア割り当てを WSL の CPU 数に合わせる（11/4 の実測＝processors=12 で runner3 の 13-15 が taskset 無効→全曲を空振りで先取り・read 76分）。
# 16 以上＝従来どおり 4-6/7-9/10-12/13-15。16 未満＝0-2/3-5/6-8/9-11（12 CPU で 4 ランナー）。それ以外の中身は run_read_20260915.sh と同じ。
NP=$(nproc); BASE=4; [ "$NP" -lt 16 ] && BASE=0; C0=$((RID*3+BASE)); C1=$((C0+2)); [ "$C1" -ge "$NP" ] && C1=$((NP-1)); [ "$C0" -gt "$C1" ] && C0=$C1
CORES=$C0-$C1; RES="$ROOT/results_wao_$S"; LOG="$RES/_runner.log"; mkdir -p "$RES"
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
