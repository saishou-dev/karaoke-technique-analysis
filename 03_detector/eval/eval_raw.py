# -*- coding: utf-8 -*-
"""吸着ラベルで学習したモデルの予測（out/<tag>/pred_test.csv・予測時刻＝音の終わり）を、G1 生時刻（カウンター増分）に対して評価する。
   予測時刻に lag（既定0.30s）を足して ±0.5/1/2s で照合。件数整合も出す。
   使い方: python eval_raw.py --tag v2d [--lag 0.3] [--truth_index index_v2_20260924.json]
   --truth_index ＝正解の時刻表を差し替える（既定 index.json ＝正解 v1。正解 v2 との比較用・2026-10-23 追加）。
   TP/FP/FN も出す（TP＝一致・FP＝予測−一致・FN＝正解−一致）。"""
import sys, json, argparse, csv, collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, "G:/研究アーカイブ/成果物/動画解析/scripts")
from song_valid_range_20261015 import clip_span  # 観測できた区間（2026-10-15 追加）

ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--lag", type=float, default=0.30); ap.add_argument("--split", default="test"); ap.add_argument("--valid_range", default="exclude_band", choices=["exclude", "keep", "exclude_band"])  # 2026-10-20 既定を exclude_band へ（build_ledgers_20261008.py と揃える。帯の列が空の曲は exclude と同じ挙動へ自動で戻る）
ap.add_argument("--truth_index", default="index.json", help="正解の時刻表（既定＝index.json＝正解v1。正解v2 は index_v2_<session>.json）")
a = ap.parse_args()
index = {d["video"]: d for d in json.load(open(f"{ROOT}/{a.truth_index}", encoding="utf-8"))}
pred = collections.defaultdict(list)
for r in csv.DictReader(open(f"{ROOT}/out/{a.tag}/pred_{a.split}.csv", encoding="utf-8-sig")): pred[(r["video"], r["type"])].append(float(r["wav_time_sec"]) + a.lag)
vids = sorted({v for v, _ in pred})
def match(pred, truth, tol):
    used = set(); hit = 0
    for p in sorted(pred):
        best = None
        for k, t in enumerate(truth):
            if k in used or abs(t - p) > tol: continue
            if best is None or abs(t - p) < abs(truth[best] - p): best = k
        if best is not None: used.add(best); hit += 1
    return hit
print(f"# {a.tag} → G1生時刻（lag +{a.lag}s）{a.split} {len(vids)}曲　正解＝{a.truth_index}　valid_range={a.valid_range}")
for tol in (0.5, 1.0, 2.0):
    print(f"## 許容±{tol}s"); print("| 技法 | 正解 | 予測 | 一致 | TP | FP | FN | 再現率 | 適合率 | F1 | 件数整合 |"); print("|---|---|---|---|---|---|---|---|---|---|---|"); H = NP = NT = CE = 0
    for c in TYPES:
        h = npd = nt = ce = 0
        for v in vids:
            d = index[v]; lo, hi = clip_span(v, d["span"], float(d.get("offset", 0.0)), a.valid_range)[0]; tr = [t for cc, t in d["events"] if cc == c and lo <= t <= hi]; pr = [t for t in pred.get((v, c), []) if lo <= t <= hi]
            h += match(pr, tr, tol); npd += len(pr); nt += len(tr); ce += abs(len(pr) - len(tr))
        H += h; NP += npd; NT += nt; CE += ce; p = h / npd if npd else 0; r = h / nt if nt else 0
        print(f"| {c} | {nt} | {npd} | {h} | {h} | {npd - h} | {nt - h} | {r:.3f} | {p:.3f} | {2*p*r/(p+r) if p+r else 0:.3f} | {1 - ce / nt if nt else float('nan'):.3f} |")
    p = H / NP if NP else 0; r = H / NT if NT else 0; print(f"| ALL | {NT} | {NP} | {H} | {H} | {NP - H} | {NT - H} | {r:.3f} | {p:.3f} | {2*p*r/(p+r) if p+r else 0:.3f} | {1 - CE / NT:.3f} |")
