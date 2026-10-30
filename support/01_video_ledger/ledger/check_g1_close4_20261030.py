# -*- coding: utf-8 -*-
"""close4（2026-10-30）の台帳検算＝RUNBOOK 5-c の 5 と 10/15 の「曲単位の完全一致（台帳 max(count_value) vs 採点CSV）」。
   使い方: python check_g1_close4_20261030.py <S> <G1.csv> [<results_dir>]
   出力＝行数・技法別件数・空欄・重複・非単調・時刻逆行・pts_source・曲単位の完全一致／読み過ぎ／欠落・G1 最終値と読取値の不一致・note 別件数。
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20261021 import Cfg1014
S, g1p = sys.argv[1], sys.argv[2]; C = Cfg1014(S); resd = sys.argv[3] if len(sys.argv) > 3 else C.results_dir
G = pd.read_csv(g1p, encoding="utf-8-sig")
d = pd.read_csv(C.dam_csv, encoding="utf-8-sig", low_memory=False); d["dt"] = pd.to_datetime(d["scoringDateTime"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
ren = {r["renamed"][:-4]: r["played_at"] for r in C.mapping_rows() if r.get("played_at") and r.get("machine") == "wao" and r.get("how") == "TIME_WINDOW"}
M = {"shakuri": "shakuriCount", "kobushi": "kobushiCount", "fall": "fallCount", "vibrato": "vibratoCount", "accent": "accentCount", "hammering": "hammeringOnCount"}
print("rows", len(G), "videos", G.video.nunique(), G.counter.value_counts().to_dict())
print("空欄 wav/audio/video_time =", int(G.wav.isna().sum()), int(G.audio_time_sec.isna().sum()), int(G.video_time_sec.isna().sum()))
print("pts_source =", G.pts_source.unique().tolist())
dup = G.duplicated(["video", "counter", "count_value"]).sum(); bad = []
for (v, c), g in G.groupby(["video", "counter"]):
    g2 = g.sort_values("count_value")
    if not g2["video_time_sec"].is_monotonic_increasing: bad.append((v[:20], c))
print("重複 =", int(dup), " 時刻逆行 =", bad)
if "note" in G.columns: print("note 別 =", G.note.fillna("(reader)").value_counts().to_dict())
pm = 0; over = []; miss = 0; mism = 0; songs_ok = []; rows = []
for v, pa in sorted(ren.items(), key=lambda kv: kv[1]):
    hit = d[d.dt == pd.Timestamp(pa)].iloc[0]; gv = G[G.video == v]
    ce = os.path.join(resd, v, "counter_events.csv"); ev = pd.read_csv(ce) if os.path.exists(ce) else None
    df = {}
    for c in M:
        mx = int(gv[gv.counter == c].count_value.max()) if (gv.counter == c).any() else 0
        df[c] = mx - int(hit[M[c]])
        if ev is not None:
            rv = int(ev[ev.counter == c].value.max()) if (ev.counter == c).any() else 0
            if rv != mx and not (rv == 0 and mx == 0): mism += 1
    ok = all(x == 0 for x in df.values()); pm += ok
    if any(x > 0 for x in df.values()): over.append((v[:24], {c: x for c, x in df.items() if x > 0}))
    miss += -sum(x for x in df.values() if x < 0)
    rows.append((v[:26], "OK" if ok else ";".join(f"{c}{x:+d}" for c, x in df.items() if x)))
print(f"曲単位の完全一致 = {pm}/{len(ren)}  読み過ぎの曲 = {len(over)} {over}  欠落計 = {miss}  G1最終値と読取値の不一致（カウンター単位） = {mism}")
for r in rows:
    if r[1] != "OK": print("   ", r[0], r[1])
