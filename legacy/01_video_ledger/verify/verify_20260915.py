# -*- coding: utf-8 -*-
"""読取結果の最終値を DAM★とも CSV（Hearts）と突合（2026-09-15版・verify_20260911.py の一般化）。
   対応付け＝<video dir>/rename_mapping.csv の played_at（採点時刻）と Hearts CSV の scoringDateTime。
   Hearts CSV＝dam_api/ の最新 dam_ai-hearts_full_*.csv（環境変数 KV_HEARTS_CSV で明示可）。
   出力: dataset/verify_<session>.csv   使い方: python scripts/verify_20260915.py [session]
"""
import os, sys, csv, glob
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915 import Cfg, DAM_DIR

C = Cfg(); S = C.S
hearts = os.environ.get("KV_HEARTS_CSV") or (sorted(glob.glob(os.path.join(DAM_DIR, "dam_ai-hearts_full_*.csv")))[-1:] or [None])[0]
if not hearts or not os.path.exists(hearts): C.log("VERIFY STOP: Hearts CSV が無い（dam_api/dam_ai-hearts_full_*.csv）"); raise SystemExit(2)
if not os.path.exists(C.rename_csv): C.log(f"VERIFY STOP: rename_mapping.csv が無い {C.rename_csv}"); raise SystemExit(2)
d = pd.read_csv(hearts, encoding="utf-8-sig")
d["dt"] = pd.to_datetime(d["scoringDateTime"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
ren = {r["renamed"][:-4]: r["played_at"] for r in csv.DictReader(open(C.rename_csv, encoding="utf-8-sig")) if r.get("played_at")}
M = {"shakuri": "shakuriCount", "kobushi": "kobushiCount", "fall": "fallCount", "vibrato": "vibratoCount", "accent": "accentCount", "hammering": "hammeringOnCount"}
rows = []
for name, pa in ren.items():
    ce = os.path.join(C.results_dir, name, "counter_events.csv")
    if not os.path.exists(ce): rows.append(dict(video=name, status="NOT_READ")); continue
    hit = d[d["dt"] == pd.Timestamp(pa)]
    if not len(hit): rows.append(dict(video=name, status="NO_CSV_ROW")); continue
    ev = pd.read_csv(ce); fin = {c: int(ev[ev["counter"] == c]["value"].max()) if (ev["counter"] == c).any() else 0 for c in M}
    row = hit.iloc[0]; diff = {c: fin[c] - int(row[M[c]]) for c in M}
    ok = all(v == 0 for v in diff.values())
    rows.append(dict(video=name, status="PERFECT" if ok else "DIFF", **{f"read_{c}": fin[c] for c in M}, **{f"csv_{c}": int(row[M[c]]) for c in M},
                     diff=";".join(f"{c}{v:+d}" for c, v in diff.items() if v)))
df = pd.DataFrame(rows); df.to_csv(C.verify_csv, index=False, encoding="utf-8-sig")
C.log(f"VERIFY hearts={os.path.basename(hearts)} PERFECT {(df['status']=='PERFECT').sum()}/{len(df)}\n" + df[["video", "status"] + (["diff"] if "diff" in df else [])].to_string(index=False))
