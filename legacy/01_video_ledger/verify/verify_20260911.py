# -*- coding: utf-8 -*-
"""読取結果の最終値をDAM★ともCSV（Hearts・本人30曲）と突合（2026-09-12）。出力: dataset/verify_20260911.csv"""
import os, csv, pandas as pd
ROOT=r"G:\研究アーカイブ\成果物\動画解析"; VID=r"G:\研究アーカイブ\ビデオ素材\20260911"
d=pd.read_csv(os.path.join(ROOT.replace("動画解析","dam_api"),"dam_ai-hearts_full_2026-09-11T14-39-21.csv"),encoding="utf-8-sig")
d["dt"]=pd.to_datetime(d["scoringDateTime"].astype(str),format="%Y%m%d%H%M%S",errors="coerce"); d=d[(d["dt"]>="2026-09-11 10:00")&(d["dt"]<="2026-09-11 13:40")]
ren={r["renamed"][:-4]:r["played_at"] for r in csv.DictReader(open(os.path.join(VID,"rename_mapping.csv"),encoding="utf-8-sig")) if r["played_at"]}
M={"shakuri":"shakuriCount","kobushi":"kobushiCount","fall":"fallCount","vibrato":"vibratoCount","accent":"accentCount","hammering":"hammeringOnCount"}
rows=[]
for name,pa in ren.items():
    ce=os.path.join(ROOT,"results_wao_20260911",name,"counter_events.csv")
    if not os.path.exists(ce): rows.append(dict(video=name,status="NOT_READ")); continue
    ev=pd.read_csv(ce); fin={c:int(ev[ev["counter"]==c]["value"].max()) if (ev["counter"]==c).any() else 0 for c in M}
    row=d[d["dt"]==pd.Timestamp(pa)].iloc[0]; diff={c:fin[c]-int(row[M[c]]) for c in M}
    ok=all(v==0 for v in diff.values())
    rows.append(dict(video=name,status="PERFECT" if ok else "DIFF",**{f"read_{c}":fin[c] for c in M},**{f"csv_{c}":int(row[M[c]]) for c in M},diff=";".join(f"{c}{v:+d}" for c,v in diff.items() if v)))
df=pd.DataFrame(rows); df.to_csv(os.path.join(ROOT,"dataset","verify_20260911.csv"),index=False,encoding="utf-8-sig")
print(df[["video","status","diff"]].to_string(index=False)); print("PERFECT",(df["status"]=="PERFECT").sum(),"/",len(df))
