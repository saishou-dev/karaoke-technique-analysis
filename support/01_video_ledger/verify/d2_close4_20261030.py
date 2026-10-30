# -*- coding: utf-8 -*-
"""close4（2026-10-30）＝付け直した帯動画（results_wao_20261021/_close4_20261030/bands）に D-2 の走査（scan_runs_20261015.scan）を当てる駆動役。
   scripts の既存ファイルは無変更。rescan_missing_20261015.SESS に close4 用のキーを実行時に注入し、出力先 OUTROOT を
   results_wao_20261021/_close4_20261030/d2/ に差し替える（dataset/found_20261015 には書かない）。優先度は BelowNormal。
   使い方: python d2_close4_20261030.py "<video>" --counters=shakuri,vibrato [--from=] [--to=] [--strip=c:a-b:step ...] [--results=<results の相対パス>] [--bands=<帯動画ディレクトリの相対パス>]
"""
import os, sys, ctypes
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
except Exception: pass
ROOT = r"G:\研究アーカイブ\成果物\動画解析"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("KV_FOUND_INVERT", "1")
import rescan_missing_20261015 as RM
res = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--results=")), r"results_wao_20261021\_close4_20261030\R2_v08e")
bands = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--bands=")), os.path.join("results_wao_20261021", "_close4_20261030", "bands"))
KEY = "20261021c4"
RM.SESS[KEY] = dict(reader="wao", verify="verify_20261021_wao.csv", results=res, bands=bands,
                    pts="pts_bands_20261021", roi="results", g1="G1_events_20261021_wao.csv", session=20261021, pts_source="pts_bands_20261021")
import scan_runs_20261015 as SR
SR.OUTROOT = os.path.join(ROOT, "results_wao_20261021", "_close4_20261030", "d2")
sys.argv = [sys.argv[0], KEY] + [a for a in sys.argv[1:] if not a.startswith("--results=") and not a.startswith("--bands=")]
name = sys.argv[2]; counters = None; f0 = 0; f1 = -1; ncc = 0.60; strips = []
for a in sys.argv[3:]:
    if a.startswith("--counters="): counters = a.split("=", 1)[1].split(",")
    elif a.startswith("--from="): f0 = int(a.split("=")[1])
    elif a.startswith("--to="): f1 = int(a.split("=")[1])
    elif a.startswith("--ncc="): ncc = float(a.split("=")[1])
    elif a.startswith("--strip="):
        s = a.split("=", 1)[1].split(":"); rng = s[1].split("-"); strips.append((s[0], int(rng[0]), int(rng[1]), int(s[2]) if len(s) > 2 else 1))
SR.scan(KEY, name, counters or RM.WAO_COUNTERS, f0, f1, ncc, strips)
