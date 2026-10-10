# -*- coding: utf-8 -*-
"""ノート分類＋過去演奏の基準（pairs/*.npz）：自分の音の特徴に「同曲の過去演奏（DTW位置合わせ）との差」と「過去の加点の有無」を足す。
   対象＝過去演奏がある曲だけ。学習＝train セッション、しきい値＝dev、評価＝test（G1生時刻 ±0.5/1/2s F1・件数整合）。
   使い方: python note_classifier_pairs.py --suffix 2"""
import os, sys, json, argparse, csv
import numpy as np
from scipy.ndimage import median_filter
from sklearn.ensemble import HistGradientBoostingClassifier
sys.stdout.reconfigure(encoding='utf-8')
ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"; FR = 100
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
ap = argparse.ArgumentParser(); ap.add_argument("--test", default="20260924"); ap.add_argument("--dev", default="20260911"); ap.add_argument("--suffix", default="2"); ap.add_argument("--lag", type=float, default=0.38)
a = ap.parse_args()
index = json.load(open(f"{ROOT}/index_snapped{a.suffix}.json", encoding="utf-8")); notes = json.load(open(f"{ROOT}/notes{a.suffix}.json", encoding="utf-8"))
def feats(d):
    pr = f"{ROOT}/pairs/{d['video']}.npz"
    if not os.path.exists(pr): return None
    P = np.load(pr); pst = P["past_st"].astype(float); pv = P["past_voiced"] > 0; pst[~pv] = np.nan; ev_t = P["ev_type"]; ev_time = P["ev_time"]
    z = np.load(f"{ROOT}/feats/{d['video']}.npz"); st = z["st"].astype(float); v = z["voiced"] > 0; rms = z["rms"].astype(float); st[~v] = np.nan
    sm = st.copy(); idx = np.where(v)[0]
    if len(idx) > 10: sm[idx] = median_filter(sm[idx], size=5, mode="nearest")
    N = notes[d["video"]]; S = np.array(N["start"]); E = np.array(N["end"]); L = np.array(N["labels"]) if N["labels"] else np.zeros((0, 6))
    X = []; keep = []
    for k in range(len(S)):
        i, j = int(round(S[k] * FR)), int(round(E[k] * FR)) - 1
        s = sm[i:j + 1]; ok = ~np.isnan(s)
        if ok.sum() < 3: continue
        s2 = s[ok]; med = np.median(s2); head = s2[:min(len(s2), 15)]; tail = s2[-min(len(s2), 15):]
        body = s2[min(len(s2) - 1, 15):max(min(len(s2) - 1, 15) + 1, len(s2) - 15)] if len(s2) > 30 else s2; tgt = np.median(body)
        if len(s2) >= 40:
            dd = s2 - median_filter(s2, size=31, mode="nearest"); dd -= dd.mean(); den = np.dot(dd, dd)
            ac = np.correlate(dd, dd, "full")[len(dd) - 1:] / den if den > 1e-9 else np.zeros(len(dd)); r = float(ac[12:26].max()) if len(ac) > 26 else 0.0; depth = float(np.std(dd) * 283)
        else: r = 0.0; depth = 0.0
        r_ = rms[i:j + 1]; gap_prev = S[k] - E[k - 1] if k > 0 else 5.0; gap_next = S[k + 1] - E[k] if k + 1 < len(S) else 5.0
        # 過去演奏との差
        ps = pst[i:j + 1]; pok = ~np.isnan(ps)
        if pok.sum() >= 3:
            pmed = np.nanmedian(ps); d_med = med - pmed; d_head = (head[0] - ps[pok][0]); d_min = head.min() - np.nanmin(ps[:min(len(ps), 15)]); pvf = pok.mean()
            phead = ps[:min(len(ps), 15)]; p_rise = (np.nanmedian(ps) - np.nanmin(phead)) if np.isfinite(np.nanmin(phead)) else 0.0
        else: d_med = d_head = d_min = 0.0; pvf = 0.0; p_rise = 0.0
        # 過去の加点（技法別）：この音の [start-0.3, end+lag+0.7] に過去イベント（今回時刻へ転写済み）があるか
        lo_, hi_ = S[k] - 0.3, E[k] + a.lag + 0.7
        prior = [int(((ev_t == t_) & (ev_time >= lo_) & (ev_time <= hi_)).any()) for t_ in range(6)]
        near = [float(np.min(np.abs(ev_time[ev_t == t_] - (E[k] + a.lag)))) if (ev_t == t_).any() else 9.0 for t_ in range(6)]
        X.append([(j - i + 1) / FR, med, tgt - head.min(), tgt - head[0], head.max() - tgt, tgt - tail[-1], tail.max() - tgt, np.std(s2), s2.max() - s2.min(), r, depth, r_.mean(), r_.max(), gap_prev, gap_next,
                  d_med, d_head, d_min, pvf, p_rise] + prior + near)
        keep.append(k)
    X = np.array(X, float).reshape(-1, 32); Y = (L[keep] > 0).astype(int) if len(keep) else np.zeros((0, 6), int)
    return X, Y, S[keep], E[keep]
data = {}
for d in index:
    f = feats(d)
    if f is not None: data[d["video"]] = (d,) + f
train = [v for v, (d, *_) in data.items() if d["session"] not in (a.test, a.dev)]; dev = [v for v, (d, *_) in data.items() if d["session"] == a.dev]; test = [v for v, (d, *_) in data.items() if d["session"] == a.test]
print(f"pairsあり: train {len(train)} dev {len(dev)} test {len(test)}", flush=True)
def match(pred, truth, tol):
    used = set(); hit = 0
    for p in sorted(pred):
        best = None
        for k, t in enumerate(truth):
            if k in used or abs(t - p) > tol: continue
            if best is None or abs(t - p) < abs(truth[best] - p): best = k
        if best is not None: used.add(best); hit += 1
    return hit
def ev_eval(vids, c, predfn):
    H = {0.5: 0, 1.0: 0, 2.0: 0}; NP = NT = cerr = 0
    for v in vids:
        d = data[v][0]; lo, hi = d["span"]; pred_t = [t for t in predfn(v) if lo <= t <= hi]; truth = [t for cc, t in d["events_raw"] if cc == c and lo <= t <= hi]
        for tol in H: H[tol] += match(pred_t, truth, tol)
        NP += len(pred_t); NT += len(truth); cerr += abs(len(pred_t) - len(truth))
    f1 = {tol: (lambda P, R: 2 * P * R / max(1e-9, P + R))(H[tol] / max(1, NP), H[tol] / max(1, NT)) for tol in H}
    return f1, 1 - cerr / max(1, NT), NT, NP
rep = [f"# ノート分類＋過去演奏の基準（pairs）　train {len(train)} / dev {len(dev)} / test {len(test)} 曲", ""]
agg = {"f1": {0.5: 0, 1.0: 0, 2.0: 0}, "f1_prior": {0.5: 0, 1.0: 0, 2.0: 0}, "cnt": 0, "cnt_prior": 0, "nt": 0}
for c_i, c in enumerate(TYPES):
    Xtr = np.vstack([data[v][1] for v in train]); Ytr = np.concatenate([data[v][2][:, c_i] for v in train])
    if Ytr.sum() < 20: rep.append(f"## {c}: 正例不足"); continue
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, class_weight="balanced", random_state=0).fit(Xtr, Ytr)
    Pd = np.concatenate([clf.predict_proba(data[v][1])[:, 1] for v in dev]); Yd = np.concatenate([data[v][2][:, c_i] for v in dev])
    th = max(((th_, (lambda p: 2 * (p & (Yd == 1)).sum() / max(1, p.sum() + Yd.sum()))(Pd >= th_)) for th_ in np.arange(0.1, 0.96, 0.02)), key=lambda x: x[1])[0]
    prob = {v: clf.predict_proba(data[v][1])[:, 1] for v in test}
    f1, cnt, NT, NP = ev_eval(test, c, lambda v: [float(data[v][4][k]) + a.lag for k in np.where(prob[v] >= th)[0]])
    # 比較＝過去の加点をそのまま転写（prior 列だけ）
    f1p, cntp, _, NPp = ev_eval(test, c, lambda v: [float(data[v][4][k]) + a.lag for k in np.where(data[v][1][:, 20 + c_i] > 0)[0]])
    # 適合率重視の動作点（±1s）：しきい値を上げたときの 適合率/再現率
    pr_lines = []
    for th2 in (0.6, 0.7, 0.8, 0.9):
        f1x, _, NTx, NPx = ev_eval(test, c, lambda v: [float(data[v][4][k]) + a.lag for k in np.where(prob[v] >= th2)[0]])
        hits = 0
        for v in test:
            d = data[v][0]; lo, hi = d["span"]; pt = [float(data[v][4][k]) + a.lag for k in np.where(prob[v] >= th2)[0] if lo <= data[v][4][k] + a.lag <= hi]; tr = [t for cc, t in d["events_raw"] if cc == c and lo <= t <= hi]; hits += match(pt, tr, 1.0)
        pr_lines.append(f"th{th2:.1f}: 予測{NPx} 適合率{hits / max(1, NPx):.2f} 再現率{hits / max(1, NTx):.2f}")
    rep.append(f"- ±1s 適合率重視: " + "／".join(pr_lines))
    rep.append(f"## {c}（しきい値{th:.2f}・正解{NT}）"); rep.append(f"- 分類器：F1 ±0.5s {f1[0.5]:.3f}／±1s {f1[1.0]:.3f}／±2s {f1[2.0]:.3f}・件数整合 {cnt:.3f}（予測{NP}）")
    rep.append(f"- 転写のみ：F1 ±0.5s {f1p[0.5]:.3f}／±1s {f1p[1.0]:.3f}／±2s {f1p[2.0]:.3f}・件数整合 {cntp:.3f}（予測{NPp}）"); rep.append("")
    for tol in f1: agg["f1"][tol] += f1[tol] * NT; agg["f1_prior"][tol] += f1p[tol] * NT
    agg["cnt"] += cnt * NT; agg["cnt_prior"] += cntp * NT; agg["nt"] += NT
    print("\n".join(rep[-3:]), flush=True)
nt = max(1, agg["nt"]); rep.append("## 全体（正解数で加重平均）")
rep.append("- 分類器：" + "／".join(f"±{t}s {agg['f1'][t] / nt:.3f}" for t in (0.5, 1.0, 2.0)) + f"・件数整合 {agg['cnt'] / nt:.3f}")
rep.append("- 転写のみ：" + "／".join(f"±{t}s {agg['f1_prior'][t] / nt:.3f}" for t in (0.5, 1.0, 2.0)) + f"・件数整合 {agg['cnt_prior'] / nt:.3f}")
txt = "\n".join(rep); print(txt); open(f"{ROOT}/out/note_pairs_report.md", "w", encoding="utf-8").write(txt)
