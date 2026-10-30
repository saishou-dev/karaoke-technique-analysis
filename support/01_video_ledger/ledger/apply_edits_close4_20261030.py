# -*- coding: utf-8 -*-
"""close4（2026-10-30）＝作り直した G1_events_20261021_wao.csv に 10/22 の目視追記（編集リスト4本・91件）を当て直す駆動役。
   dataset/found_20261021/apply_edits_20261022.py と同じ注入（SESS["20261021"]・統合台帳は写しに差し替え＝本物に書かない）に加え、
   note のタグを元の追記日 20261022 に固定する（g1_edits_v2 の DATETAG は実行日が既定のため）。scripts の既存ファイルは無変更。
   使い方: python apply_edits_close4_20261030.py <edits.csv> --tag=<退避名> [--note-tag=YYYYMMDD（既定 20261022）] [--dry]
"""
import os, sys, shutil
ROOT = r"G:\研究アーカイブ\成果物\動画解析"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rescan_missing_20261015 as RM
RM.SESS["20261021"] = dict(reader="wao", verify="verify_20261021_wao.csv", results="results_wao_20261021",
                           bands="bands_wao", pts="pts_bands_20261021", roi="results",
                           g1=next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--g1=")), "G1_events_20261021_wao.csv"),  # --g1＝dataset からの相対（試走用）
                           session=20261021, pts_source="pts_bands_20261021")
import g1_edits_v2_20261020 as GE
GE.DATETAG = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--note-tag=")), "20261022")
REAL = os.path.join(ROOT, "dataset", "G1_events_all.csv")
COPY = os.path.join(ROOT, "results_wao_20261021", "_close4_20261030", "_G1_events_all_sandbox.csv")
shutil.copy(REAL, COPY); GE.ALL = COPY
print("[sandbox] G1_events_all の写し＝", COPY, " note タグ＝found_visual_" + GE.DATETAG)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
tag = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--tag=")), "edits")
GE.main(args[0], tag, "--dry" in sys.argv)
