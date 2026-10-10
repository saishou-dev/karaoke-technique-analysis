# -*- coding: utf-8 -*-
"""ノート単位の分類：各ノート（pYIN有声ラン分割）に「加点あり/なし」を技法別に付け（snap_events.py の吸着ラベル）、
   ノート特徴量＋前後文脈で HistGradientBoosting を学習。test（9/24）でノート単位F1、G1生時刻に対する±1s F1、件数整合を出す。
   使い方: python note_classifier.py [--test 20260924 --dev 20260911]"""
import sys, json, argparse, csv
import numpy as np
from scipy.ndimage import median_filter
from sklearn.ensemble import HistGradientBoostingClassifier
sys.stdout.reconfigure(encoding='utf-8')
ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"; FR = 100
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
ap = argparse.ArgumentParser(); ap.add_argument("--test", default="20260924"); ap.add_argument("--dev", default="20260911"); ap.add_argument("--suffix", default="")
a = ap.parse_args()
index = json.load(open(f"{ROOT}/index_snapped{a.suffix}.json", encoding="utf-8")); notes = json.load(open(f"{ROOT}/notes{a.suffix}.json", encoding="utf-8"))
def note_feats(d):
    z = np.load(f"{ROOT}/feats/{d['video']}.npz"); st = z["st"].astype(float); v = z["voiced"] > 0; rms = z["rms"].astype(float); conf = z["conf"].astype(float)
    st[~v] = np.nan; sm = st.copy(); idx = np.where(v)[0]
    if len(idx) > 10: sm[idx] = median_filter(sm[idx], size=5, mode="nearest")
    N = notes[d["video"]]; S = np.array(N["start"]); E = np.array(N["end"]); L = np.array(N["labels"]) if N["labels"] else np.zeros((0, 6))
    X = []; keep = []
    for k in range(len(S)):
        i, j = int(round(S[k] * FR)), int(round(E[k] * FR)) - 1; n = j - i + 1
        if n < 3: continue
        s = sm[i:j + 1]; s = s[~np.isnan(s)]
        if len(s) < 3: continue
        med = np.median(s); head = s[:min(len(s), 15)]; tail = s[-min(len(s), 15):]
        body = s[min(len(s) - 1, 15):max(min(len(s) - 1, 15) + 1, len(s) - 15)] if len(s) > 30 else s
        tgt = np.median(body)
        # ビブラート系
        if len(s) >= 40:
            dd = s - median_filter(s, size=31, mode="nearest"); dd -= dd.mean(); den = np.dot(dd, dd)
            ac = np.correlate(dd, dd, "full")[len(dd) - 1:] / den if den > 1e-9 else np.zeros(len(dd)); r = float(ac[12:26].max()) if len(ac) > 26 else 0.0; depth = float(np.std(dd) * 283)
        else: r = 0.0; depth = 0.0
        r_ = rms[i:j + 1]; gap_prev = S[k] - E[k - 1] if k > 0 else 5.0; gap_next = S[k + 1] - E[k] if k + 1 < len(S) else 5.0
        prev_med = np.nanmedian(sm[int(round(S[k - 1] * FR)):int(round(E[k - 1] * FR))]) if k > 0 else med
        next_med = np.nanmedian(sm[int(round(S[k + 1] * FR)):int(round(E[k + 1] * FR))]) if k + 1 < len(S) else med
        X.append([n / FR, med, tgt - head.min(), tgt - head[0], head.max() - tgt, tgt - tail[-1], tail.max() - tgt, tail.min() - tgt, np.std(s), s.max() - s.min(), r, depth,
                  r_.mean(), r_.max(), r_[:min(len(r_), 10)].mean() - r_.mean(), r_[-min(len(r_), 10):].mean() - r_.mean(), conf[i:j + 1].mean(), gap_prev, gap_next, med - prev_med, next_med - med,
                  (np.diff(s).max() if len(s) > 1 else 0), (np.diff(s).min() if len(s) > 1 else 0), np.argmin(head) / max(1, len(head)), np.argmax(tail) / max(1, len(tail))])
        keep.append(k)
    X = np.array(X, float).reshape(-1, 25); Y = L[keep] if len(keep) else np.zeros((0, 6)); return X, (Y > 0).astype(int), S[keep], E[keep]
data = {d["video"]: (d,) + note_feats(d) for d in index}
train = [v for v, (d, *_) in data.items() if d["session"] not in (a.test, a.dev)]; dev = [v for v, (d, *_) in data.items() if d["session"] == a.dev]; test = [v for v, (d, *_) in data.items() if d["session"] == a.test]
print(f"train {len(train)} dev {len(dev)} test {len(test)} notes(train) {sum(len(data[v][1]) for v in train)}", flush=True)
def match(pred, truth, tol):
    used = set(); hit = 0
    for p in sorted(pred):
        best = None
        for k, t in enumerate(truth):
            if k in used or abs(t - p) > tol: continue
            if best is None or abs(t - p) < abs(truth[best] - p): best = k
        if best is not None: used.add(best); hit += 1
    return hit
rep = ["# ノート単位分類（HistGradientBoosting）", f"train={len(train)}曲 dev={a.dev} test={a.test}", ""]
rows_out = []; summary = {}
for c_i, c in enumerate(TYPES):
    Xtr = np.vstack([data[v][1] for v in train]); Ytr = np.concatenate([data[v][2][:, c_i] for v in train])
    if Ytr.sum() < 20: rep.append(f"## {c}: 正例不足"); continue
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, class_weight="balanced", random_state=0).fit(Xtr, Ytr)
    # dev でしきい値（ノート単位F1最大）
    Pd = np.concatenate([clf.predict_proba(data[v][1])[:, 1] for v in dev]); Yd = np.concatenate([data[v][2][:, c_i] for v in dev])
    best = max(((th, (lambda p: 2 * (p & (Yd == 1)).sum() / max(1, p.sum() + Yd.sum()))(Pd >= th)) for th in np.arange(0.1, 0.96, 0.02)), key=lambda x: x[1]); th = best[0]
    # test：ノート単位
    Pt = np.concatenate([clf.predict_proba(data[v][1])[:, 1] for v in test]); Yt = np.concatenate([data[v][2][:, c_i] for v in test]); pt = Pt >= th
    tp = int((pt & (Yt == 1)).sum()); prec = tp / max(1, pt.sum()); rec = tp / max(1, Yt.sum()); f1n = 2 * prec * rec / max(1e-9, prec + rec)
    # test：G1生時刻に対する事象F1（予測時刻＝ノート終端＋0.26s）と件数整合
    H = {0.5: 0, 1.0: 0, 2.0: 0}; NP = NT = 0; cerr = 0; ctrue = 0
    for v in test:
        d, X, Y, S, E = data[v]; p = clf.predict_proba(X)[:, 1] >= th; lo, hi = d["span"]
        pred_t = [float(E[k]) + 0.26 for k in np.where(p)[0] if lo <= E[k] + 0.26 <= hi]; truth = [t for cc, t in d["events_raw"] if cc == c and lo <= t <= hi]
        for tol in H: H[tol] += match(pred_t, truth, tol)
        NP += len(pred_t); NT += len(truth); cerr += abs(len(pred_t) - len(truth)); ctrue += len(truth)
        rows_out.append(dict(video=v, type=c, true=len(truth), pred=len(pred_t)))
    f1 = {tol: (lambda P, R: 2 * P * R / max(1e-9, P + R))(H[tol] / max(1, NP), H[tol] / max(1, NT)) for tol in H}
    summary[c] = dict(note_f1=f1n, note_p=prec, note_r=rec, ev_f1=f1, count=1 - cerr / max(1, ctrue), th=th, ntrue=NT, npred=NP)
    rep.append(f"## {c}（しきい値{th:.2f}）"); rep.append(f"- ノート単位：適合率{prec:.3f} 再現率{rec:.3f} F1 {f1n:.3f}（正例{int(Yt.sum())}／予測{int(pt.sum())}）")
    rep.append(f"- G1生時刻に対する事象F1：±0.5s {f1[0.5]:.3f}／±1.0s {f1[1.0]:.3f}／±2.0s {f1[2.0]:.3f}（正解{NT}・予測{NP}）"); rep.append(f"- 件数整合 {1 - cerr / max(1, ctrue):.3f}"); rep.append("")
    print("\n".join(rep[-4:]), flush=True)
NT = sum(s["ntrue"] for s in summary.values());
rep.append("## 全体（技法合算・件数重み）");
for tol in (0.5, 1.0, 2.0): rep.append(f"- 事象F1 ±{tol}s（正解数で加重平均）: {sum(s['ev_f1'][tol] * s['ntrue'] for s in summary.values()) / NT:.3f}")
rep.append(f"- ノート単位F1（加重平均）: {sum(s['note_f1'] * s['ntrue'] for s in summary.values()) / NT:.3f}")
rep.append(f"- 件数整合（加重平均）: {sum(s['count'] * s['ntrue'] for s in summary.values()) / NT:.3f}")
txt = "\n".join(rep); print(txt); open(f"{ROOT}/out/note_classifier_report{a.suffix}.md", "w", encoding="utf-8").write(txt)
with open(f"{ROOT}/out/note_counts_test.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["video", "type", "true", "pred"]); w.writeheader(); w.writerows(rows_out)
