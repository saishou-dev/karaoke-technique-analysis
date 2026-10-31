#!/bin/bash
# a4_apply_all.sh ＝ a4_rules.py（規則(c)＝CONDS の "k=3+(c)" を含む7条件）を v0.8f の読み直し（reread_v08f_20261031）の
#   6技法の全収録へ当てる薄いラッパー（2026-10-31）。toshiaki で実行（nice 10）。
#   入力は grade_v08f_all.py が書いた grade_c/<収録>/_inputs/verify_norm.csv・G1_frames.csv
#   （収録ごとの verify と G1 の差＝9/11 bandpts_rerun・9/15 rescan・8/18 と 9/2 の verify は正式結果フォルダ・
#    8/12・8/18・9/2 の G1 はフレーム列が無いので G1_events_all の idx 時刻×fps で換算＝を吸収した形）。
#   Ai（4技法）は a4_rules.py が6技法前提のため対象外＝grade_v08f_all.py が同じ関数で採点する。
#   出力＝grade_c/<収録>/a4/A4_table.csv・A4_per_counter.csv・a4_rules.log
D=/mnt/g/研究アーカイブ/成果物/動画解析/unread_fix_validate_20261030
O=/mnt/g/研究アーカイブ/成果物/動画解析/reread_v08f_20261031
PY=~/ocr_compare_20261018/.venv312/bin/python
source ~/ocr_compare_20261018/env.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$D"
for k in 20261014 20260924 20261006 20260911 20260915 20261021 20260902 20260818 20260812; do
  I="$O/grade_c/$k/_inputs"
  mkdir -p "$O/grade_c/$k/a4"
  nice -n 10 $PY -u a4_rules.py --scan "$O/fullscan/$k" --verify "$I/verify_norm.csv" --g1 "$I/G1_frames.csv" \
    --out "$O/grade_c/$k/a4" > "$O/grade_c/$k/a4/a4_rules.log" 2>&1
  echo "$k exit=$? $(grep -c '' "$O/grade_c/$k/a4/A4_per_counter.csv" 2>/dev/null) rows"
done
nice -n 10 $PY -u grade_v08f_all.py --check-a4
