#!/bin/bash
# 11/10 D-2 の自動候補（rescan→exact）を曲ごとに並列で回す（2026-11-10 Opus・11/9 版の複製・10並列）
cd "/g/研究アーカイブ/成果物/動画解析/dataset/found_20261110"
for i in 0 1 2 3 4 5 6 7 8 9; do
  ( python d2_driver_20261110.py rescan p$i found_20261110/_verify_p$i.csv > _rescan_p$i.log 2>&1 ;
    python d2_driver_20261110.py exact p$i --verify=found_20261110/_verify_p$i.csv --ev=3 > _exact_p$i.log 2>&1 ;
    echo DONE >> _exact_p$i.log ) &
done
wait
echo ALLDONE > _d2_alldone.txt
