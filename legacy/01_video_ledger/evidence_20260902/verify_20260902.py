#!/usr/bin/env python3
"""9/2 29本 x DAM Hearts CSV 突合(時刻窓+6カウンター一致)
usage: python verify_20260902.py <hearts_csv> <out_wao_dir> [--g1 <out_g1_csv>]
"""
import csv, os, sys, json, glob
from datetime import datetime, timedelta

CSV_PATH, OUT_WAO = sys.argv[1], sys.argv[2]
G1_OUT = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--g1" else None

COUNTERS = ["shakuri", "kobushi", "fall", "vibrato", "accent", "hammering"]
CSV_COLS = {"shakuri": "shakuriCount", "kobushi": "kobushiCount", "fall": "fallCount",
            "vibrato": "vibratoCount", "accent": "accentCount", "hammering": "hammeringOnCount"}

# DAM行 (8/12のみ)
dam = [r for r in csv.DictReader(open(CSV_PATH, encoding="utf-8-sig"))
       if r["playDate"].startswith("2026-09-02")]
for r in dam:
    r["_dt"] = datetime.strptime(r["playDate"], "%Y-%m-%d %H:%M:%S")

rows_out = []
g1_rows = []
n_perfect = 0
videos = sorted(glob.glob(os.path.join(OUT_WAO, "*_20260902_*")))
for d in videos:
    video = os.path.basename(d)
    jpath = os.path.join(d, "counter_events.json")
    if not os.path.exists(jpath):
        continue
    acc = json.load(open(jpath))
    finals = {k: (acc[k][-1][1] if acc.get(k) else 0) for k in COUNTERS}
    # 動画開始時刻 (ファイル名 曲名_20260902_HHMMSS)
    hhmmss = video.rsplit("_", 1)[1]
    vstart = datetime.strptime("2026-09-02 " + hhmmss, "%Y-%m-%d %H%M%S")
    # 時刻窓: playDate(演奏終了)は動画開始の +1分〜+8分 に入るはず
    cands = [r for r in dam if timedelta(minutes=0) < (r["_dt"] - vstart) < timedelta(minutes=8)]
    # 最も近い終了時刻を採用
    best = min(cands, key=lambda r: r["_dt"] - vstart) if cands else None
    if best is None:
        rows_out.append({"video": video, "dam_song": "NO_TIME_MATCH", "verdict": "NO_MATCH"})
        continue
    diffs = {k: finals[k] - int(best[CSV_COLS[k]] or 0) for k in COUNTERS}
    perfect = all(v == 0 for v in diffs.values())
    if perfect:
        n_perfect += 1
    rows_out.append({
        "video": video, "dam_song": best["songName"], "dam_playDate": best["playDate"],
        "verdict": "PERFECT" if perfect else "DIFF",
        **{f"read_{k}": finals[k] for k in COUNTERS},
        **{f"csv_{k}": best[CSV_COLS[k]] for k in COUNTERS},
        **{f"diff_{k}": diffs[k] for k in COUNTERS},
    })
    # G1行 (counter_events.csv の delta>0 行)
    with open(os.path.join(d, "counter_events.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["delta"] == "0":
                continue
            g1_rows.append({
                "session": "20260902", "video": video, "counter": r["counter"],
                "count_value": r["value"], "delta": r["delta"],
                "video_time_sec": f'{float(r.get("event_time_est") or r["time_sec"]):.3f}',
                "stable_read_time_sec": r["time_sec"], "take": "", "audio_time_sec": "",
            })

# レポート
w = csv.DictWriter(open(os.path.join(OUT_WAO, "verify_20260902.csv"), "w", newline="", encoding="utf-8-sig"),
                   fieldnames=list(rows_out[0].keys()))
w.writeheader(); w.writerows(rows_out)
print(f"videos={len(rows_out)} PERFECT={n_perfect}")
for r in rows_out:
    if r["verdict"] != "PERFECT":
        print("DIFF:", r["video"], r.get("dam_song"),
              {k: r[f"diff_{k}"] for k in COUNTERS if f"diff_{k}" in r and r[f"diff_{k}"] != 0} if r["verdict"] == "DIFF" else "")

if G1_OUT and g1_rows:
    w = csv.DictWriter(open(G1_OUT, "w", newline="", encoding="utf-8-sig"),
                       fieldnames=list(g1_rows[0].keys()))
    w.writeheader(); w.writerows(g1_rows)
    from collections import Counter
    print(f"G1 rows={len(g1_rows)} -> {G1_OUT}")
    print(Counter(r["counter"] for r in g1_rows))
