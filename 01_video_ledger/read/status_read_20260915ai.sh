#!/bin/bash
# 読取の進捗表示 DAM Ai 版。使い方: KV_SESSION=20260915 [KV_AI_BASE=<wslパス>] bash status_read_20260915ai.sh
S="${KV_SESSION:-20260915}"; ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"; BASE="${KV_AI_BASE:-$ROOT}"; R="$BASE/results_ai_$S"
echo "session=$S (ai) runners: $(pgrep -fa "run_read_20260915ai.sh [0-3]" | wc -l)  readers: $(pgrep -fa "read_counters_a[i]" | wc -l)"
pgrep -fa "read_counters_a[i]" | sed "s|.*bands_ai/||" | cut -c1-70
echo "results done: $(ls "$R"/*/counter_events.csv 2>/dev/null | wc -l)   bands: $(ls "$BASE"/bands_ai/*_${S}_*.mp4 2>/dev/null | grep -v part | wc -l)   gatefail: $(ls "$BASE"/bands_ai/_gatefail_$S/*.mp4 2>/dev/null | wc -l)"
[ -f "$R/_runner.log" ] && tail -n 6 "$R/_runner.log"
