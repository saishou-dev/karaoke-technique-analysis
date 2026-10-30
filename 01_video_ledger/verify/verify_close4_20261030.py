# -*- coding: utf-8 -*-
"""close4（2026-10-30）用の突合＝verify_20261021.py と同じ規則（Hearts CSV の採点時刻で動画を引き、counter_events.csv の技法別 max と比べる）を
   任意の読取結果ディレクトリに当て、比較表を出す（既存ファイルは読むだけ）。可読率は read_log.txt の readable 行から取る。
   使い方: python verify_close4_20261030.py <S> <名前=ディレクトリ> [<名前=ディレクトリ> ...] [--videos=a,b,...] [--out=<csv>]
   出力（標準出力）＝候補ごとの PERFECT 数・読み過ぎ（read>csv）の件数と合計・欠落（read<csv）の合計・誤差絶対値計・技法別可読率の中央値・曲ごとの diff。
"""
import os, sys, re, csv
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20261021 import Cfg1014
S = sys.argv[1]; C = Cfg1014(S)
dirs = [a.split("=", 1) for a in sys.argv[2:] if not a.startswith("--") and "=" in a]
vf = next((a.split("=", 1)[1].split(",") for a in sys.argv[2:] if a.startswith("--videos=")), None)
outp = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--out=")), None)
d = pd.read_csv(C.dam_csv, encoding="utf-8-sig", low_memory=False)
d["dt"] = pd.to_datetime(d["scoringDateTime"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
ren = {r["renamed"][:-4]: r["played_at"] for r in C.mapping_rows() if r.get("played_at") and r.get("machine") == "wao" and r.get("how") == "TIME_WINDOW"}
M = {"shakuri": "shakuriCount", "kobushi": "kobushiCount", "fall": "fallCount", "vibrato": "vibratoCount", "accent": "accentCount", "hammering": "hammeringOnCount"}
rows = []
for lab, dr in dirs:
    for name, pa in sorted(ren.items(), key=lambda kv: kv[1]):
        if vf and name not in vf: continue
        ce = os.path.join(dr, name, "counter_events.csv")
        if not os.path.exists(ce): rows.append(dict(cand=lab, video=name, status="NOT_READ")); continue
        hit = d[d["dt"] == pd.Timestamp(pa)]
        ev = pd.read_csv(ce); fin = {c: int(ev[ev["counter"] == c]["value"].max()) if (ev["counter"] == c).any() else 0 for c in M}
        csvv = {c: int(hit.iloc[0][M[c]]) for c in M}
        rd = {}
        lp = os.path.join(dr, name, "read_log.txt")
        if os.path.exists(lp):
            for m in re.finditer(r"(\w+): readable (\d+)/(\d+)", open(lp, encoding="utf-8", errors="replace").read()): rd[m.group(1)] = int(m.group(2)) / max(1, int(m.group(3)))
        diff = {c: fin[c] - csvv[c] for c in M}
        r = dict(cand=lab, video=name, status="PERFECT" if all(v == 0 for v in diff.values()) else "DIFF",
                 over=sum(v for v in diff.values() if v > 0), n_over=sum(1 for v in diff.values() if v > 0), miss=-sum(v for v in diff.values() if v < 0),
                 abserr=sum(abs(v) for v in diff.values()), read_med=round(float(pd.Series(list(rd.values())).median()), 3) if rd else None,
                 diff=";".join(f"{c}{v:+d}" for c, v in diff.items() if v))
        for c in M: r[f"rd_{c}"] = round(rd.get(c, float("nan")), 3); r[f"read_{c}"] = fin[c]; r[f"csv_{c}"] = csvv[c]
        rows.append(r)
R = pd.DataFrame(rows)
for lab, _ in dirs:
    x = R[R.cand == lab]
    print(f"[{lab}] 曲={len(x)} PERFECT={int((x.status=='PERFECT').sum())} 読み過ぎの曲={int((x.n_over>0).sum())} 読み過ぎ計={int(x.over.sum())} 欠落計={int(x.miss.sum())} 誤差絶対値計={int(x.abserr.sum())} 可読率中央値(曲の中央値)={x.read_med.median():.3f}")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(R[["cand", "video", "status", "abserr", "over", "miss", "read_med", "diff"]].assign(video=R.video.str[:22]).to_string(index=False))
if outp: R.to_csv(outp, index=False, encoding="utf-8-sig"); print("WROTE", outp)
