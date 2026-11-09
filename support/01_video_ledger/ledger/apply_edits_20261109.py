# -*- coding: utf-8 -*-
r"""g1_edits_v2_20261020.py を 20261104 セッションに当てる駆動役（scripts は変更しない）。2026-11-09＝found_20261021\apply_edits_20261022.py の 11/4 版（セッション名だけ変更）。

(1) rescan_missing_20261015.SESS に 20261104 のキーを実行時に注入する。
(2) 統合台帳 G1_events_all.csv を**絶対に触らない**ため、ALL を読み取り専用の写しに差し替える。
    20261104 は統合台帳に未結合（0 行）なので、g1_edits 側は写しに1行も足さない。
    それでも最後に A.to_csv(ALL) が無条件で走るので、本物ではなく写しに書かせる。

使い方: python apply_edits_20261109.py <edits.csv> --tag=<退避名> [--dry]
"""
import os, sys, shutil
ROOT = r"G:\研究アーカイブ\成果物\動画解析"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rescan_missing_20261015 as RM
RM.SESS["20261104"] = dict(reader="wao", verify="verify_20261104_wao.csv", results="results_wao_20261104",
                           bands="bands_wao", pts="pts_bands_20261104", roi="results",
                           g1="G1_events_20261104_wao.csv", session=20261104,
                           pts_source="pts_bands_20261104")
import g1_edits_v2_20261020 as GE

REAL = os.path.join(ROOT, "dataset", "G1_events_all.csv")
COPY = os.path.join(ROOT, "dataset", "found_20261104", "_G1_events_all_sandbox.csv")
shutil.copy(REAL, COPY)
GE.ALL = COPY                       # 本物には触らせない
print("[sandbox] G1_events_all の写し＝", COPY)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
tag = "edits"
for a in sys.argv[1:]:
    if a.startswith("--tag="):
        tag = a.split("=", 1)[1]
GE.main(args[0], tag, "--dry" in sys.argv)
