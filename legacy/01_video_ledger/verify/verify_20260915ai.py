# -*- coding: utf-8 -*-
"""読取結果の最終値を DAM★とも CSV（**Ai**）と突合（2026-09-15・verify_20260915.py の Ai 派生）。
   対応付け＝<video dir>/rename_mapping.csv の played_at（採点時刻）と Ai CSV の scoringDateTime。
   Ai CSV＝dam_api/ の最新 dam_ai_full_*.csv（環境変数 KV_DAM_CSV で明示可）。技法カウンターは4つ（しゃくり・こぶし・フォール・ビブラート）。
   9/11 の Hearts 混入除外リスト（exclude_others_20260911.txt）は Ai には不要（04_research 66行「Ai機種への混入なし」）。
   出力: dataset/verify_<session>.csv   使い方: python scripts/verify_20260915ai.py [session]
"""
import os, sys, csv
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915ai import CfgAi

C = CfgAi(); S = C.S
dam = C.latest_dam_csv()
if not dam or not os.path.exists(dam): C.log(f"VERIFY STOP: Ai CSV が無い（dam_api/{C.dam_glob}）"); raise SystemExit(2)
if not os.path.exists(C.rename_csv): C.log(f"VERIFY STOP: rename_mapping.csv が無い {C.rename_csv}"); raise SystemExit(2)
d = pd.read_csv(dam, encoding="utf-8-sig", low_memory=False)
d["dt"] = pd.to_datetime(d["scoringDateTime"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
ren = {r["renamed"][:-4]: r["played_at"] for r in csv.DictReader(open(C.rename_csv, encoding="utf-8-sig")) if r.get("played_at")}
M = C.csv_map
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
df = pd.DataFrame(rows); os.makedirs(os.path.dirname(C.verify_csv), exist_ok=True); df.to_csv(C.verify_csv, index=False, encoding="utf-8-sig")
C.log(f"VERIFY(ai) csv={os.path.basename(dam)} PERFECT {(df['status']=='PERFECT').sum()}/{len(df)}\n" + df[["video", "status"] + (["diff"] if "diff" in df else [])].to_string(index=False))
