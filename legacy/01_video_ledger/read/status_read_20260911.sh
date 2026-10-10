#!/bin/bash
R="/mnt/g/研究アーカイブ/成果物/動画解析/results_wao_20260911"
echo "runners: $(pgrep -fa "run_read_2026091[1].sh [0-3]" | wc -l)  readers: $(pgrep -fa "read_counters_wa[o]" | wc -l)"
pgrep -fa "read_counters_wa[o]" | sed "s|/home/toshiaki/karaoke-vision/.venv/bin/python /home/toshiaki/karaoke-vision/read_counters_wao.py /mnt/g/研究アーカイブ/成果物/動画解析/bands_wao/||" | cut -c1-70
echo "results done: $(ls "$R"/*/counter_events.csv 2>/dev/null | grep -v ".mp4.part" | wc -l)   bands: $(ls /mnt/g/研究アーカイブ/成果物/動画解析/bands_wao/*_20260911_*.mp4 | grep -v part | wc -l)"
tail -n 6 "$R/_runner.log"
