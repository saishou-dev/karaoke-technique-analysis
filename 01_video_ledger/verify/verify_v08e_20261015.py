# -*- coding: utf-8 -*-
"""verify_20261014.py と同じ突合を、読取器 v0.8e の別出力（results_wao_<S>_v08e）に対して行う検証版（2026-10-15）。
   既存の results_wao_<S> と dataset/verify_<S>_wao.csv は読むだけで一切上書きしない。
   出力＝dataset/verify_<S>_wao_v08e.csv ＋ 標準出力に v0.8d との比較表。
   使い方: python scripts/verify_v08e_20261015.py [20261014]
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20261014 import Cfg1014

if sys.stdout is not None: sys.stdout.reconfigure(encoding="utf-8")
C = Cfg1014(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].isdigit() else None); S = C.S
RES = C.results_dir + "_v08e"
OUT = os.path.join(os.path.dirname(C.verify_csv), f"verify_{S}_wao_v08e.csv")
d = pd.read_csv(C.dam_csv, encoding="utf-8-sig", low_memory=False)
d["dt"] = pd.to_datetime(d["scoringDateTime"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
ren = {r["renamed"][:-4]: r["played_at"] for r in C.mapping_rows()
       if r.get("played_at") and r.get("machine") == "wao" and r.get("how") == "TIME_WINDOW"}
M = {"shakuri": "shakuriCount", "kobushi": "kobushiCount", "fall": "fallCount", "vibrato": "vibratoCount",
     "accent": "accentCount", "hammering": "hammeringOnCount"}
rows = []
for name, pa in sorted(ren.items(), key=lambda kv: kv[1]):
    ce = os.path.join(RES, name, "counter_events.csv")
    if not os.path.exists(ce): rows.append(dict(video=name, status="NOT_READ")); continue
    hit = d[d["dt"] == pd.Timestamp(pa)]
    if not len(hit): rows.append(dict(video=name, status="NO_CSV_ROW")); continue
    ev = pd.read_csv(ce)
    fin = {c: int(ev[ev["counter"] == c]["value"].max()) if (ev["counter"] == c).any() else 0 for c in M}
    row = hit.iloc[0]; diff = {c: fin[c] - int(row[M[c]]) for c in M}
    rows.append(dict(video=name, status="PERFECT" if all(v == 0 for v in diff.values()) else "DIFF",
                     **{f"read_{c}": fin[c] for c in M}, **{f"csv_{c}": int(row[M[c]]) for c in M},
                     diff=";".join(f"{c}{v:+d}" for c, v in diff.items() if v)))
df = pd.DataFrame(rows); df.to_csv(OUT, index=False, encoding="utf-8-sig")
n_read = (df["status"] != "NOT_READ").sum()
print(f"v0.8e  PERFECT {(df['status']=='PERFECT').sum()}/{n_read}  → {OUT}")

old = pd.read_csv(C.verify_csv, encoding="utf-8-sig")
o = {r["video"]: r for _, r in old.iterrows()}
print("\n| 曲 | v0.8d | v0.8e | 変化 |")
print("|---|---|---|---|")
better = worse = same = 0
for _, r in df.iterrows():
    if r["status"] == "NOT_READ": continue
    po = o.get(r["video"])
    if po is None: continue
    do = sum(abs(int(po[f"read_{c}"]) - int(po[f"csv_{c}"])) for c in M)
    dn = sum(abs(int(r[f"read_{c}"]) - int(r[f"csv_{c}"])) for c in M)
    mark = "改善" if dn < do else ("悪化" if dn > do else "同じ")
    better += dn < do; worse += dn > do; same += dn == do
    print("| %s | %s(誤差計%d) | %s(誤差計%d) | %s |"
          % (r["video"].split("_2026")[0], po["status"], do, r["status"], dn, mark))
print(f"\n改善 {better} / 悪化 {worse} / 同じ {same}")
print("\n| 技法 | v0.8d read/csv | v0.8e read/csv | v0.8d 過剰 | v0.8e 過剰 | v0.8d 誤差絶対値計 | v0.8e 誤差絶対値計 |")
print("|---|---|---|---|---|---|---|")
sub = df[df["status"] != "NOT_READ"]
osub = old[old["video"].isin(set(sub["video"]))]
for c in M:
    rn, cn = sub[f"read_{c}"].sum(), sub[f"csv_{c}"].sum()
    ro, co = osub[f"read_{c}"].sum(), osub[f"csv_{c}"].sum()
    ovn = (sub[f"read_{c}"] - sub[f"csv_{c}"]).clip(lower=0).sum()
    ovo = (osub[f"read_{c}"] - osub[f"csv_{c}"]).clip(lower=0).sum()
    abn = (sub[f"read_{c}"] - sub[f"csv_{c}"]).abs().sum()
    abo = (osub[f"read_{c}"] - osub[f"csv_{c}"]).abs().sum()
    print("| %s | %d/%d=%.3f | %d/%d=%.3f | +%d | +%d | %d | %d |" % (c, ro, co, ro / co if co else 0, rn, cn, rn / cn if cn else 0, ovo, ovn, abo, abn))
