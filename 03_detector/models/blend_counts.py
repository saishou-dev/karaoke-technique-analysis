# -*- coding: utf-8 -*-
"""曲単位の件数：曲特徴の回帰（Ridge）と CRNN 積分（counts_test_integral.csv）を混ぜて件数整合を出す。
   使い方: python blend_counts.py --tag v2b [--test 20260924]"""
import sys, json, argparse, csv
import numpy as np
from sklearn.linear_model import RidgeCV
sys.stdout.reconfigure(encoding='utf-8')
ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"; FR = 100
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--test", default="20260924"); ap.add_argument("--w", type=float, default=0.5, help="CRNN積分の重み")
a = ap.parse_args()
index = json.load(open(f"{ROOT}/index.json", encoding="utf-8"))
def songfeat(d):
    z = np.load(f"{ROOT}/feats/{d['video']}.npz"); v = z["voiced"]; st = z["st"]; rms = z["rms"]; T = len(v)
    lo, hi = d["span"]; i0, i1 = max(0, int(lo * FR)), min(T, int(hi * FR)); v = v[i0:i1]; st = st[i0:i1]; rms = rms[i0:i1]
    on = ((v[1:] > 0) & (v[:-1] == 0)).sum(); jump = ((np.abs(np.diff(st)) > 1.0) & (v[1:] > 0) & (v[:-1] > 0)).sum()
    runs = []; c = 0
    for x in v:
        if x > 0: c += 1
        elif c: runs.append(c); c = 0
    runs = np.array(runs) if runs else np.array([0]); dst = np.diff(st) * (v[1:] > 0) * (v[:-1] > 0)
    return [(i1 - i0) / FR, v.sum() / FR, on, jump, (runs >= 50).sum(), (runs >= 100).sum(), (runs >= 200).sum(), rms.mean(), (np.abs(dst) > 0.5).sum(), (dst > 0.3).sum(), (dst < -0.3).sum(), np.nanstd(st[v > 0]) if (v > 0).sum() > 10 else 0]
X = np.array([songfeat(d) for d in index], float); S = np.array([d["session"] for d in index]); V = [d["video"] for d in index]
Y = np.array([[sum(1 for c, t in d["events"] if c == tt and d["span"][0] <= t <= d["span"][1]) for tt in TYPES] for d in index], float)
tr = S != a.test; te = ~tr
crnn = {}
for r in csv.DictReader(open(f"{ROOT}/out/{a.tag}/counts_test_integral.csv", encoding="utf-8-sig")): crnn[(r["video"], r["type"])] = float(r["pred"])
print(f"test={a.test} 曲数={te.sum()}  CRNN重み={a.w}")
print("| 技法 | Σ正解 | 回帰のみ | CRNN積分のみ | 混合 |"); print("|---|---|---|---|---|")
tot = np.zeros(4)
rows = []
for k, c in enumerate(TYPES):
    m = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(X[tr], Y[tr, k]); pr = np.clip(m.predict(X[te]), 0, None)
    pc = np.array([crnn.get((V[i], c), np.nan) for i in np.where(te)[0]]); pc = np.where(np.isnan(pc), pr, pc)
    pb = a.w * pc + (1 - a.w) * pr; yt = Y[te, k]
    e = [np.abs(np.round(p) - yt).sum() for p in (pr, pc, pb)]; t = yt.sum(); tot += [t] + e
    print(f"| {c} | {int(t)} | {1 - e[0] / t if t else float('nan'):.3f} | {1 - e[1] / t if t else float('nan'):.3f} | {1 - e[2] / t if t else float('nan'):.3f} |")
    for i, vi in enumerate(np.where(te)[0]): rows.append(dict(video=V[vi], type=c, true=int(yt[i]), ridge=round(float(pr[i]), 1), crnn=round(float(pc[i]), 1), blend=int(round(pb[i]))))
print(f"| ALL | {int(tot[0])} | {1 - tot[1] / tot[0]:.3f} | {1 - tot[2] / tot[0]:.3f} | {1 - tot[3] / tot[0]:.3f} |")
with open(f"{ROOT}/out/{a.tag}/counts_test_blend.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
