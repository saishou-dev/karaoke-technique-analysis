#!/usr/bin/env python3
"""grade_v08f_roi.py ＝ grade_v08f_all.py を出力先だけ grade_c_roi に替えて回す薄いラッパー（2026-10-31）

背景＝v3_fullscan_all.py の roi_of を「較正 ROI（dataset/rescan_<収録>/roi）→正式結果 roi.json→既定」に直し、
  9/15 を較正 ROI で読み直した（旧走査＝reread_v08f_20261031/fullscan_pre_roi_20261031/20260915）。
  旧版の採点結果 grade_c/ は残し、直した後の採点を grade_c_roi/ に分けて書く（grade_v08f_all.py 本体は変えない）。
使い方＝grade_v08f_all.py と同じ引数（--only <収録>・--check-a4）。--only を付けないと全収録＋summary_by_session.csv・
  diff_vs_G1.csv・G1_events_all_v08f_candidate.csv を grade_c_roi/ に書く。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grade_v08f_all as g

g.OUT = g.ROOT + "/reread_v08f_20261031/grade_c_roi"

if __name__ == "__main__":
    sys.exit(g.main())
