export LANG=C.UTF-8; cd ~/karaoke-vision
for v in "CHEERS_20260911_105621" "Answer_20260911_130234" "lovin'_20260911_113934"; do out='/mnt/g/研究アーカイブ/成果物/動画解析/dataset/_selftest_calib_20260915/results_0911/'"$v"; mkdir -p "$out"
  ( .venv/bin/python read_counters_wao.py '/mnt/g/研究アーカイブ/成果物/動画解析/bands_wao/'"$v".mp4 "$out" '/mnt/g/研究アーカイブ/成果物/動画解析/templates/wao_20260816/cluster_reps_wao_ext.npy' "$out/roi.json" > "$out/read_log.txt" 2>&1 ) &
done; wait; ls '/mnt/g/研究アーカイブ/成果物/動画解析/dataset/_selftest_calib_20260915/results_0911'/*/counter_events.csv | wc -l