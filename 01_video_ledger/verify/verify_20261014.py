# -*- coding: utf-8 -*-
"""読取結果の最終値を DAM★とも CSV と突合（2026-10-14 版・verify_20261006.py の派生・既存は無変更）。
   10/14 は全曲 WAO（Hearts）の1機種＝--machine wao のみ（既定）。Hearts CSV＝kv_config_20261014.HEARTS_CSV 固定・6カウンター・results_wao_<S>。
   対象＝rename_mapping.csv の machine=wao かつ how=TIME_WINDOW の動画だけ。出力＝dataset/verify_<S>_wao.csv。
   使い方: python scripts/verify_20261014.py [20261014] [--machine wao]
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20261014 import Cfg1014

machine = sys.argv[sys.argv.index("--machine") + 1] if "--machine" in sys.argv else "wao"
if machine != "wao":
    print("STOP: 10/14 は WAO のみ（--machine wao）"); raise SystemExit(2)
C = Cfg1014(); S = C.S
src = C.dam_csv
if not os.path.exists(src): C.log(f"VERIFY STOP: CSV が無い {src}"); raise SystemExit(2)
if not os.path.exists(C.rename_csv): C.log(f"VERIFY STOP: rename_mapping.csv が無い {C.rename_csv}"); raise SystemExit(2)
d = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
d["dt"] = pd.to_datetime(d["scoringDateTime"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
ren = {r["renamed"][:-4]: r["played_at"] for r in C.mapping_rows() if r.get("played_at") and r.get("machine") == machine and r.get("how") == "TIME_WINDOW"}
M = {"shakuri": "shakuriCount", "kobushi": "kobushiCount", "fall": "fallCount", "vibrato": "vibratoCount",
     "accent": "accentCount", "hammering": "hammeringOnCount"}
rows = []
for name, pa in sorted(ren.items(), key=lambda kv: kv[1]):
    ce = os.path.join(C.results_dir, name, "counter_events.csv")
    if not os.path.exists(ce): rows.append(dict(video=name, status="NOT_READ")); continue
    hit = d[d["dt"] == pd.Timestamp(pa)]
    if not len(hit): rows.append(dict(video=name, status="NO_CSV_ROW")); continue
    ev = pd.read_csv(ce); fin = {c: int(ev[ev["counter"] == c]["value"].max()) if (ev["counter"] == c).any() else 0 for c in M}
    row = hit.iloc[0]; diff = {c: fin[c] - int(row[M[c]]) for c in M}
    ok = all(v == 0 for v in diff.values())
    rows.append(dict(video=name, status="PERFECT" if ok else "DIFF", **{f"read_{c}": fin[c] for c in M}, **{f"csv_{c}": int(row[M[c]]) for c in M},
                     diff=";".join(f"{c}{v:+d}" for c, v in diff.items() if v)))
df = pd.DataFrame(rows); os.makedirs(os.path.dirname(C.verify_csv), exist_ok=True); df.to_csv(C.verify_csv, index=False, encoding="utf-8-sig")
C.log(f"VERIFY({machine}) csv={os.path.basename(src)} PERFECT {(df['status']=='PERFECT').sum()}/{len(df)} → {C.verify_csv}\n"
      + (df[["video", "status"] + (["diff"] if "diff" in df else [])].to_string(index=False) if len(df) else "(対象なし)"))
