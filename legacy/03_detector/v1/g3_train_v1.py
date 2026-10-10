# -*- coding: utf-8 -*-
"""G3 技法検出器 v1（2026-09-10・初版）
   教師＝G1_events_synced.csv（pts補正済み・multi_increment=0・has_pitch=1）、音声側＝dataset/pitch_pyin/<video>_f0.csv（10ms）。
   候補生成→特徴量→HistGradientBoosting。学習＝0628/0807/0812/0818、検証＝0902（未見）。
   出力: models/g3_v1/<type>.joblib, models/g3_v1/thresholds.json, dataset/g3_v1_report.md, dataset/g3_v1_val_predictions.csv
"""
import os, csv, json, math, time
import numpy as np, pandas as pd
from scipy.ndimage import median_filter
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
ROOT = r"G:\研究アーカイブ\成果物\動画解析"; PDIR = os.path.join(ROOT, "dataset", "pitch_pyin"); MDIR = os.path.join(ROOT, "models", "g3_v1")
os.makedirs(MDIR, exist_ok=True)
HOP = 0.01; VAL_SESSION = "20260902"; TYPES = ["shakuri", "kobushi", "fall", "vibrato"]
CONF_MIN = float(os.environ.get("G3_CONF", "0.0")); MIN_DUR = float(os.environ.get("G3_MINDUR", "0.08")); CW = os.environ.get("G3_CW", "balanced"); TAG = os.environ.get("G3_TAG", "")
if TAG: MDIR = os.path.join(ROOT, "models", "g3_v1_" + TAG); os.makedirs(MDIR, exist_ok=True)
LAB_WIN = {"shakuri": (-0.3, 0.9), "kobushi": (-0.3, 0.9), "fall": (-0.3, 0.9), "vibrato": (-0.2, 1.0)}  # 候補アンカーから見た加点時刻の許容

ev = pd.read_csv(os.path.join(ROOT, "dataset", "G1_events_synced.csv"), encoding="utf-8-sig", dtype={"session": str})
ev = ev[(ev["has_pitch"] == 1) & (ev["multi_increment"] == 0) & ev["pitch_time_sec"].notna()].copy()
videos = sorted(ev["video"].unique())
print("videos", len(videos), "events", len(ev), flush=True)

PF = dict(zip(ev["video"], ev["pitch_file"]))  # 0807はVID_→曲名にリネーム済みなので pitch_file 列で引く
def load_pitch(video):
    p = os.path.join(PDIR, str(PF.get(video) or (video + "_f0.csv")))
    if not os.path.exists(p): return None
    d = pd.read_csv(p, encoding="utf-8-sig")
    t = d["audio_time_sec"].values.astype(float); f0 = d["f0_hz"].values.astype(float); conf = d["conf"].values.astype(float)
    v = (f0 > 40) & (conf >= CONF_MIN)
    st = np.full(len(f0), np.nan); st[v] = 69 + 12 * np.log2(f0[v] / 440.0)
    return t, st, v

def segments(t, st, v):
    """有声ランを音符に分割：無声で切る／隣接フレームの中央値差が0.8半音超で切る。80ms未満は捨てる"""
    sm = st.copy(); ok = ~np.isnan(sm)
    if ok.sum() < 10: return []
    # 有声区間内だけ中央値フィルタ
    idx = np.where(ok)[0]; sm[idx] = median_filter(sm[idx], size=5, mode="nearest")
    segs = []; i = 0; n = len(t)
    while i < n:
        if not ok[i]: i += 1; continue
        j = i
        while j + 1 < n and ok[j + 1]:
            a = np.nanmedian(sm[max(i, j - 4):j + 1]); b = np.nanmedian(sm[j + 1:j + 6])
            if abs(b - a) > 0.8 and (j - i) >= 8: break
            j += 1
        if (j - i + 1) * HOP >= MIN_DUR: segs.append((i, j))
        i = j + 1
    return segs, sm

def feats_onset(t, sm, segs, k):
    i, j = segs[k]; n = j - i + 1
    w = lambda a, b: sm[i + max(0, int(a / HOP)): i + min(n, int(b / HOP))]
    p0 = np.nanmedian(sm[i:i + 3]); tgt = np.nanmedian(w(0.2, 0.45)) if n > 25 else np.nanmedian(sm[i:j + 1])
    rise = tgt - p0; first = w(0.0, 0.12); mn = np.nanmin(first) if len(first) else p0
    reach = next((q for q in range(n) if sm[i + q] >= tgt - 0.3), n) * HOP
    slope = np.nanmax(np.diff(sm[i:i + min(n, 30)])) / HOP if n > 2 else 0
    gap = (t[i] - t[segs[k - 1][1]]) if k > 0 else 5.0
    prevp = np.nanmedian(sm[segs[k - 1][0]:segs[k - 1][1] + 1]) if k > 0 else tgt
    return [rise, tgt - mn, reach, slope, gap, tgt - prevp, n * HOP, p0]

def feats_note(t, sm, segs, k):
    i, j = segs[k]; s = sm[i:j + 1]; n = len(s)
    base = median_filter(s, size=min(31, n if n % 2 else n - 1) if n >= 3 else 1, mode="nearest") if n >= 3 else s
    ex = s - base; up = np.nanmax(ex) if n else 0; dn = np.nanmin(ex) if n else 0
    above = ex > 0.5; runs = np.diff(np.concatenate([[0], above.astype(int), [0]]))
    nb = int((runs == 1).sum()); width = 0
    if nb:
        starts = np.where(runs == 1)[0]; ends = np.where(runs == -1)[0]; width = float(np.max(ends - starts)) * HOP
    return [up, dn, nb, width, n * HOP, float(np.nanstd(s)), float(np.nanmax(s) - np.nanmin(s))]

def feats_offset(t, sm, segs, k):
    i, j = segs[k]; n = j - i + 1
    stab = np.nanmedian(sm[max(i, j - 40):max(i + 1, j - 20)]); pend = np.nanmedian(sm[max(i, j - 2):j + 1])
    drop = pend - stab; q = j
    while q > i and sm[q] < stab - 0.3: q -= 1
    ddur = (j - q) * HOP; slope = np.nanmin(np.diff(sm[max(i, j - 30):j + 1])) / HOP if n > 2 else 0
    gap = (t[segs[k + 1][0]] - t[j]) if k + 1 < len(segs) else 5.0
    return [drop, ddur, slope, gap, n * HOP, pend]

def feats_vib(t, sm, i0, i1):
    s = sm[i0:i1]; ok = ~np.isnan(s)
    if ok.sum() < 30: return None
    s = np.interp(np.arange(len(s)), np.where(ok)[0], s[ok]); d = s - median_filter(s, size=31, mode="nearest")
    d = d - d.mean(); den = np.dot(d, d)
    if den <= 1e-9: return None
    ac = np.correlate(d, d, "full")[len(d) - 1:] / den; lags = np.arange(14, 31)  # 140〜300ms
    r = ac[lags]; b = int(np.argmax(r)); rate = 1.0 / (lags[b] * HOP)
    depth = float(np.std(d) * 2 * math.sqrt(2) * 100)  # cents
    zc = np.where(np.diff(np.sign(d)) != 0)[0]; reg = float(np.std(np.diff(zc)) * HOP) if len(zc) > 3 else 1.0
    return [float(r[b]), rate, depth, reg, float(np.nanmean(s)), float(ac[1])]

def build(video):
    lp = load_pitch(video)
    if lp is None: return []
    t, st, v = lp; sg = segments(t, st, v)
    if not sg: return []
    segs, sm = sg; e = ev[ev["video"] == video]; rows = []
    def label(typ, anchor):
        lo, hi = LAB_WIN[typ]; tt = e[e["counter"] == typ]["pitch_time_sec"].values
        return int(np.any((tt >= anchor + lo) & (tt <= anchor + hi)))
    for k, (i, j) in enumerate(segs):
        a = t[i]; rows.append(dict(video=video, type="shakuri", anchor=a, label=label("shakuri", a), f=feats_onset(t, sm, segs, k)))
        rows.append(dict(video=video, type="kobushi", anchor=a + (j - i) * HOP / 2, label=label("kobushi", t[i] + (j - i) * HOP / 2), f=feats_note(t, sm, segs, k)))
        rows.append(dict(video=video, type="fall", anchor=t[j], label=label("fall", t[j]), f=feats_offset(t, sm, segs, k)))
    win = 50; step = 10
    for i0 in range(0, len(t) - win, step):
        f = feats_vib(t, sm, i0, i0 + win)
        if f is None: continue
        a = t[i0 + win // 2]; rows.append(dict(video=video, type="vibrato", anchor=a, label=label("vibrato", a), f=f))
    return rows

t0 = time.time(); allrows = []
for n, vid in enumerate(videos):
    allrows += build(vid)
    if n % 10 == 0: print(f"{n}/{len(videos)} {vid[:20]} rows={len(allrows)} {time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(allrows); df["session"] = df["video"].map(dict(zip(ev["video"], ev["session"])))
print("candidates", df.groupby("type")["label"].agg(["count", "sum"]), flush=True)

report = ["# G3 検出器 v1 レポート（%s）TAG=%s conf=%.2f mindur=%.2f cw=%s" % (time.strftime("%Y-%m-%d %H:%M"), TAG, CONF_MIN, MIN_DUR, CW), "", "学習＝0628/0807/0812/0818・検証＝0902（未見）。候補→HistGradientBoosting。事象レベル評価＝許容±1.0s（副次±0.5s）。", ""]
th = {}; valpred = []
for typ in TYPES:
    d = df[df["type"] == typ]; X = np.array(d["f"].tolist(), dtype=float); y = d["label"].values
    tr = d["session"] != VAL_SESSION; va = ~tr
    if y[tr].sum() < 20 or y[va].sum() < 5:
        report.append(f"## {typ}: 例が少なく学習不可（train pos={int(y[tr].sum())}, val pos={int(y[va].sum())}）"); continue
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, class_weight=(None if CW == "none" else "balanced"), random_state=0)
    clf.fit(X[tr], y[tr]); p = clf.predict_proba(X[va])[:, 1]
    dv = d[va].copy(); dv["p"] = p
    # 事象レベル評価：しきい値を振って、検証セッションの真イベント（type）に対する再現率・適合率
    evv = ev[(ev["session"] == VAL_SESSION) & (ev["counter"] == typ)]
    best = None; lines = []
    for thr in np.arange(0.2, 0.95, 0.05):
        pos = dv[dv["p"] >= thr]
        # 同一動画内で0.5s以内の陽性候補をまとめる（最大pの1件）
        preds = []
        for vid, g in pos.groupby("video"):
            g = g.sort_values("anchor"); cur = []
            for _, r in g.iterrows():
                if cur and r["anchor"] - cur[-1]["anchor"] > 0.5: preds.append(max(cur, key=lambda z: z["p"])); cur = []
                cur.append(r)
            if cur: preds.append(max(cur, key=lambda z: z["p"]))
        for tol in (1.0, 0.5):
            hit_ev = 0
            for _, e in evv.iterrows():
                lo, hi = LAB_WIN[typ]
                if any(pr["video"] == e["video"] and (e["pitch_time_sec"] - pr["anchor"]) >= lo - (tol - 0.5) and (e["pitch_time_sec"] - pr["anchor"]) <= hi + (tol - 0.5) for pr in preds): hit_ev += 1
            hit_pr = sum(1 for pr in preds if any((evv["video"] == pr["video"]) & (evv["pitch_time_sec"] - pr["anchor"] >= LAB_WIN[typ][0] - (tol - 0.5)) & (evv["pitch_time_sec"] - pr["anchor"] <= LAB_WIN[typ][1] + (tol - 0.5))))
            rec = hit_ev / max(1, len(evv)); prec = hit_pr / max(1, len(preds)); f1 = 2 * rec * prec / max(1e-9, rec + prec)
            lines.append((thr, tol, len(preds), rec, prec, f1))
            if tol == 1.0 and (best is None or f1 > best[5]): best = (thr, tol, len(preds), rec, prec, f1)
    joblib.dump(clf, os.path.join(MDIR, typ + ".joblib")); th[typ] = float(best[0])
    report.append(f"## {typ}\n- 学習例 pos/neg={int(y[tr].sum())}/{int((~y[tr].astype(bool)).sum())}・検証の真イベント={len(evv)}\n- 最良（±1.0s・F1）: thr={best[0]:.2f} 予測数={best[2]} 再現率={best[3]:.2f} 適合率={best[4]:.2f} F1={best[5]:.2f}")
    report.append("- しきい値別（thr, tol, 予測数, 再現率, 適合率, F1）:")
    report += ["  - " + ", ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in l) for l in lines if abs(l[0] - best[0]) < 0.026 or l[0] in (0.3, 0.5, 0.7)]
    dv["type"] = typ; valpred.append(dv[["video", "type", "anchor", "label", "p"]])
    print(typ, "done", best, flush=True)
json.dump(th, open(os.path.join(MDIR, "thresholds.json"), "w"), indent=1)
if valpred: pd.concat(valpred).to_csv(os.path.join(ROOT, "dataset", "g3_v1_val_predictions" + ("_" + TAG if TAG else "") + ".csv"), index=False, encoding="utf-8-sig")
report += ["", "特徴量：しゃくり＝[上昇幅, 最低点からの上昇, 到達時間, 最大傾き, 直前の無音長, 前ノートとの差, ノート長, 開始音高]／こぶし＝[上ぶれ, 下ぶれ, 山の数, 山の幅, ノート長, 標準偏差, 音域]／フォール＝[下降幅, 下降時間, 最小傾き, 直後の無音長, ノート長, 終端音高]／ビブラート＝[自己相関ピーク, 速さHz, 深さcent, 周期の不規則さ, 平均音高, 隣接相関]", "アクセント＝音量（RMS）が要るので今回未実装。ハンマリング＝見送り。"]
open(os.path.join(ROOT, "dataset", "g3_v1_report" + ("_" + TAG if TAG else "") + ".md"), "w", encoding="utf-8").write("\n".join(report))
print("REPORT WRITTEN", time.time() - t0, flush=True)
