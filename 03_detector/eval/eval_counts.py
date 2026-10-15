# -*- coding: utf-8 -*-
"""件数ベースの整合評価：曲×技法の予測件数 vs 正解件数（DAMカウンター最終値と同じ）。
   使い方: python eval_counts.py --pred out/v2a/pred_test.csv [--win 30]
   出力: 技法別に 一致率＝1−Σ|pred−true|/Σtrue、曲ごとの完全一致率、窓（--win 秒）単位の一致率"""
import sys, json, argparse, csv, collections
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, "G:/研究アーカイブ/成果物/動画解析/scripts")
from song_valid_range_20261015 import clip_span  # 観測できた区間（2026-10-15 追加）

ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--win", type=float, default=30.0); ap.add_argument("--valid_range", default="exclude", choices=["exclude", "keep"])
a = ap.parse_args()
index = {d["video"]: d for d in json.load(open(f"{ROOT}/index.json", encoding="utf-8"))}
pred = collections.defaultdict(list)
for r in csv.DictReader(open(a.pred, encoding="utf-8-sig")):
    pred[(r["video"], r["type"])].append(float(r["wav_time_sec"]))
videos = sorted({v for v, _ in pred} | set())
if not videos: sys.exit("no predictions")
rows = []; agg = {c: [0, 0, 0, 0] for c in TYPES}  # sum|d|, sum true, n songs, exact
wagg = {c: [0, 0, 0] for c in TYPES}  # sum|d| per window, sum true, n windows exact
wtot = {c: 0 for c in TYPES}
for v in videos:
    d = index[v]; lo, hi = clip_span(v, d["span"], float(d.get("offset", 0.0)), a.valid_range)[0]
    for c in TYPES:
        tr = [t for cc, t in d["events"] if cc == c and lo <= t <= hi]
        pr = [t for t in pred.get((v, c), []) if lo <= t <= hi]
        agg[c][0] += abs(len(pr) - len(tr)); agg[c][1] += len(tr); agg[c][2] += 1; agg[c][3] += int(len(pr) == len(tr))
        rows.append(dict(video=v, type=c, true=len(tr), pred=len(pr), diff=len(pr) - len(tr)))
        edges = np.arange(lo, hi + a.win, a.win)
        ht, _ = np.histogram(tr, edges); hp, _ = np.histogram(pr, edges)
        wagg[c][0] += np.abs(hp - ht).sum(); wagg[c][1] += ht.sum(); wagg[c][2] += int((hp == ht).sum()); wtot[c] += len(ht)
print(f"曲数 {len(videos)}　件数整合＝1−Σ|予測−正解|/Σ正解")
print("| 技法 | Σ正解 | Σ|差| | 件数整合 | 曲で完全一致 | " + f"{int(a.win)}秒窓の整合 | 窓の完全一致 |")
print("|---|---|---|---|---|---|---|")
T0 = T1 = W0 = W1 = 0
for c in TYPES:
    s, t, n, ex = agg[c]; ws, wt, wex = wagg[c]
    print(f"| {c} | {t} | {s} | {1 - s / t if t else float('nan'):.3f} | {ex}/{n} | {1 - ws / wt if wt else float('nan'):.3f} | {wex}/{wtot[c]} |")
    T0 += s; T1 += t; W0 += ws; W1 += wt
print(f"| ALL | {T1} | {T0} | {1 - T0 / T1:.3f} |  | {1 - W0 / W1:.3f} |  |")
out = a.pred.replace("pred_", "counts_")
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("saved", out)
