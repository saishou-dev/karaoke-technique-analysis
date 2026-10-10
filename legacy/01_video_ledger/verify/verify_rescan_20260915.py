# -*- coding: utf-8 -*-
"""9/15 再走査の verify（verify_20260915.py と同じ突合を results_wao_20260915_rescan に対して行い、旧 verify_20260915.csv と比較する）。
   出力: dataset/verify_20260915_rescan.csv（同じ列）・dataset/rescan_20260915/verify_compare.csv（曲×カウンターの 前/後/CSV・一致）・
   ログ＝dataset/rescan_20260915/_pipeline.log（既存 pipeline.log には書かない）。使い方: python scripts/verify_rescan_20260915.py"""
import os, sys, csv, glob
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915 import Cfg, DAM_DIR
C = Cfg("20260915"); S = C.S
RD = os.path.join(C.root, "dataset", f"rescan_{S}"); C.pipe_log = os.path.join(RD, "_pipeline.log")
C.results_dir = os.path.join(C.root, f"results_wao_{S}_rescan" + os.environ.get("KV_SUFFIX", ""))
OLD = C.verify_csv; C.verify_csv = os.path.join(C.root, "dataset", f"verify_{S}_rescan{os.environ.get('KV_SUFFIX','')}.csv")
hearts = os.environ.get("KV_HEARTS_CSV") or (sorted(glob.glob(os.path.join(DAM_DIR, "dam_ai-hearts_full_*.csv")))[-1:] or [None])[0]
d = pd.read_csv(hearts, encoding="utf-8-sig"); d["dt"] = pd.to_datetime(d["scoringDateTime"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
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
    rows.append(dict(video=name, status="PERFECT" if all(v == 0 for v in diff.values()) else "DIFF", **{f"read_{c}": fin[c] for c in M}, **{f"csv_{c}": int(row[M[c]]) for c in M},
                     diff=";".join(f"{c}{v:+d}" for c, v in diff.items() if v)))
df = pd.DataFrame(rows); df.to_csv(C.verify_csv, index=False, encoding="utf-8-sig")
C.log(f"VERIFY(rescan) hearts={os.path.basename(hearts)} PERFECT {(df['status']=='PERFECT').sum()}/{len(df)}\n" + df[["video", "status"] + (["diff"] if "diff" in df else [])].to_string(index=False))
# 旧 verify との比較（カウンター別一致数）
old = pd.read_csv(OLD, encoding="utf-8-sig").set_index("video") if os.path.exists(OLD) else None
cmp_rows = []; agree_old = {c: 0 for c in M}; agree_new = {c: 0 for c in M}; n = 0
for _, r in df.iterrows():
    if r["status"] not in ("PERFECT", "DIFF"): continue
    n += 1
    for c in M:
        o = int(old.loc[r["video"], f"read_{c}"]) if old is not None and r["video"] in old.index else None
        cmp_rows.append(dict(video=r["video"], counter=c, csv=int(r[f"csv_{c}"]), read_before=o, read_after=int(r[f"read_{c}"]),
                             match_before=(o == int(r[f"csv_{c}"])) if o is not None else "", match_after=int(r[f"read_{c}"]) == int(r[f"csv_{c}"])))
        agree_new[c] += int(int(r[f"read_{c}"]) == int(r[f"csv_{c}"]))
        if o is not None: agree_old[c] += int(o == int(r[f"csv_{c}"]))
pd.DataFrame(cmp_rows).to_csv(os.path.join(RD, f"verify_compare{os.environ.get('KV_SUFFIX','')}.csv"), index=False, encoding="utf-8-sig")
C.log("VERIFY(rescan) カウンター別一致 前→後（n=%d）: " % n + "  ".join(f"{c} {agree_old[c]}→{agree_new[c]}" for c in M))
