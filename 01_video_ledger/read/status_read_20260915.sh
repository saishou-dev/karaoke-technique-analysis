#!/bin/bash
# 読取の進捗表示。使い方: KV_SESSION=20260915 bash status_read_20260915.sh
S="${KV_SESSION:-20260915}"; ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"; R="$ROOT/results_wao_$S"
echo "session=$S runners: $(pgrep -fa "run_read_20260915.sh [0-3]" | wc -l)  readers: $(pgrep -fa "read_counters_wa[o]" | wc -l)"
pgrep -fa "read_counters_wa[o]" | sed "s|.*bands_wao/||" | cut -c1-70
echo "results done: $(ls "$R"/*/counter_events.csv 2>/dev/null | wc -l)   bands: $(ls "$ROOT"/bands_wao/*_${S}_*.mp4 2>/dev/null | grep -v part | wc -l)   gatefail: $(ls "$ROOT"/bands_wao/_gatefail_$S/*.mp4 2>/dev/null | wc -l)"
[ -f "$R/_runner.log" ] && tail -n 6 "$R/_runner.log"
