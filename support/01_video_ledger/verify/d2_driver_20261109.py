# -*- coding: utf-8 -*-
"""11/4 セッション（session=20261104・machine=wao）に D-2（目視追記）の既存ツールを当てる駆動役。

scripts は変更禁止（本指示・2026-11-09 kaishu20261104-merge）なので、rescan_missing_20261015.SESS へ 20261104 のキーを**実行時に注入**して
既存の run() をそのまま呼ぶ。found_exact_20261020 は同じ SESS オブジェクトを import しているので注入が効く。

使い方:
  python d2_driver_20261022.py rescan <keysuffix> <verify_subset_rel>
  python d2_driver_20261022.py exact  <keysuffix> [--ev=N] [--ncc=0.60] [--run=3] [--gap2=5]
    keysuffix          ... 出力先 dataset/found_20261015/<key> を分けるための接尾辞（p0 等・空なら 20261104）
    verify_subset_rel  ... dataset/ 配下の相対パス（曲を絞った verify の写し）
"""
import os, sys
ROOT = r"G:\研究アーカイブ\成果物\動画解析"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("KV_FOUND_INVERT", "1")   # D-2 の 2（明背景＝反転二値化で読み直す）
import rescan_missing_20261015 as RM

def inject(key, verify_rel):
    RM.SESS[key] = dict(reader="wao", verify=verify_rel, results="results_wao_20261104",
                        bands="bands_wao", pts="pts_bands_20261104", roi="results",
                        g1="G1_events_20261104_wao.csv", session=20261104,
                        pts_source="pts_bands_20261104")
    return key

mode = sys.argv[1]
suf = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else ""
key = "20261104" + (("_" + suf) if suf else "")

if mode == "rescan":
    inject(key, sys.argv[3])
    RM.run(key)
elif mode == "exact":
    vr = None
    for a in sys.argv[3:]:
        if a.startswith("--verify="): vr = a.split("=", 1)[1]
    inject(key, vr or "verify_20261104_wao.csv")
    import found_exact_20261020 as FE
    kw = dict(nccmin=0.60, RUN=3, nev=3, gap2=5)
    for a in sys.argv[3:]:
        if a.startswith("--ncc="): kw["nccmin"] = float(a.split("=")[1])
        elif a.startswith("--run="): kw["RUN"] = int(a.split("=")[1])
        elif a.startswith("--ev="): kw["nev"] = int(a.split("=")[1])
        elif a.startswith("--gap2="): kw["gap2"] = int(a.split("=")[1])
    FE.run(key, **kw)
else:
    raise SystemExit("mode は rescan / exact")
