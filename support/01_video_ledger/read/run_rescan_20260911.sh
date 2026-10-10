#!/bin/bash
export LANG=C.UTF-8; cd ~/karaoke-vision
taskset -c 4-9 .venv/bin/python "/mnt/g/研究アーカイブ/成果物/動画解析/scripts/rescan_missing_20260911.py" > "/mnt/g/研究アーカイブ/成果物/動画解析/dataset/rescan_20260911.log" 2>&1
echo RESCAN_DONE >> "/mnt/g/研究アーカイブ/成果物/動画解析/dataset/rescan_20260911.log"
