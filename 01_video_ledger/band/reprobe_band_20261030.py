# -*- coding: utf-8 -*-
"""帯の付け直し（再プローブ）2026-10-30・WSL cv2・既存スクリプトは無変更。
   目的＝帯プローブの位置・傾き・幅のずれで、帯クロップの下端で数字が切れている曲（10/21 前半5曲＋Mela!＋ダンスホール）の
         プローブ行（xL,yL,w,h,angle）を作り直す。
   方式＝元動画から曲の 20〜80% の区間で12フレームを取り、wincrop_20260915.py と同じ変換（crop 余白20/48→rotate(-angle)→crop→scale 2480x120）を
         cv2 で再現し、calib_roi_20260915.py と同じラベル照合（9/11 CHEERS のラベル・Laplacian NCC・scale 0.94〜1.06）で
         6カウンターの (dx,dy) を測る。dx＝k·x＋b、dy＝a·x＋c の直線を当て、幅 w を (1+k) 倍・中心を (b,c) ぶん・傾きを a ぶん直す。
         これを最大6回繰り返し、|k|<0.004・|a|<0.003・中心ずれ<3px で止める。傾きの符号は両方試して残差が小さい方を採る。
   出力＝dataset/close4_20261030/reprobe_<video>.json（反復の記録）と reprobe_rows.csv（band_probe_v2 と同じ列＋note）。
   使い方：python reprobe_band_20261030.py <session> <video> [<video> ...]
"""
import os, sys, csv, json, math, numpy as np, cv2
ROOT = "/mnt/g/研究アーカイブ/成果物/動画解析"; S = sys.argv[1]; names = sys.argv[2:]
VID = f"/mnt/g/研究アーカイブ/ビデオ素材/{S}"
OUT = os.path.join(ROOT, "dataset", "close4_20261030"); os.makedirs(OUT, exist_ok=True)
LAB = {"shakuri": (70, 260), "kobushi": (410, 640), "fall": (815, 1030), "vibrato": (1215, 1460), "accent": (1640, 1890), "hammering": (2075, 2340)}
TY0, TY1 = 28, 96; PADX, PADY = 120, 40; SX = 220; SCALES = (0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06)
def prep(g): return cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F, ksize=3)
def padded(g): return cv2.copyMakeBorder(g, PADY, PADY, PADX, PADX, cv2.BORDER_REPLICATE)
def gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cap = cv2.VideoCapture(os.path.join(ROOT, "bands_wao", "CHEERS_20260911_105621.mp4")); refs = []
for f in (2000, 3500, 6000):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
    if ok: refs.append(gray(img))
cap.release(); ref = np.median(np.stack(refs), axis=0).astype(np.uint8)
T = {c: ref[TY0:TY1, a:b] for c, (a, b) in LAB.items()}
def match(P, c):
    a, b = LAB[c]; X0 = max(0, a - SX + PADX); X1 = min(P.shape[1], b + SX + PADX); reg = P[:, X0:X1]; best = None
    for s in SCALES:
        t = prep(cv2.resize(T[c], (int((b - a) * s), int((TY1 - TY0) * s))))
        if t.shape[0] > reg.shape[0] or t.shape[1] > reg.shape[1]: continue
        r = cv2.matchTemplate(reg, t, cv2.TM_CCOEFF_NORMED); _, m, _, loc = cv2.minMaxLoc(r)
        if best is None or m > best[0]: best = (float(m), X0 + loc[0] - PADX - a, loc[1] - PADY - TY0, s)
    return best
def crop(img, p):
    xL, yL, w, h, ang = int(round(p["xL"])), int(round(p["yL"])), int(round(p["w"])), int(round(p["h"])), float(p["angle_deg"]); mx, my = 20, 48
    W, H = w + 2 * mx, h + 2 * my; x0, y0 = xL - mx, yL - my
    sub = np.zeros((H, W, 3), np.uint8); sx0, sy0 = max(0, x0), max(0, y0); sx1, sy1 = min(img.shape[1], x0 + W), min(img.shape[0], y0 + H)
    sub[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    M = cv2.getRotationMatrix2D(((W - 1) / 2, (H - 1) / 2), ang, 1.0)  # ffmpeg rotate(-θ rad)＝反時計回り θ
    rot = cv2.warpAffine(sub, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return cv2.resize(rot[my:my + h, mx:mx + w], (2480, 120), interpolation=cv2.INTER_CUBIC)
def measure(frames, p):
    rows = {c: [] for c in LAB}
    for img in frames:
        P = prep(padded(gray(crop(img, p))))
        for c in LAB:
            m = match(P, c)
            if m: rows[c].append(m)
    res = {}
    for c in LAB:
        good = [r for r in rows[c] if r[0] >= 0.25]
        res[c] = dict(n_good=len(good), score=round(float(np.median([r[0] for r in rows[c]])), 3) if rows[c] else 0,
                      dx=float(np.median([r[1] for r in good])) if good else None, dy=float(np.median([r[2] for r in good])) if good else None)
    return res
def fit(res, thr=0.30):
    pts = [((LAB[c][0] + LAB[c][1]) / 2, r["dx"], r["dy"], r["score"]) for c, r in res.items() if r["dx"] is not None and r["score"] >= thr and r["n_good"] >= 4]
    if len(pts) < 2: return None  # 2点でも直線を当てる（反復で残りのカウンターが照合域に入ってくる）
    X = np.array([q[0] for q in pts]); k, b = np.polyfit(X, [q[1] for q in pts], 1); a, c = np.polyfit(X, [q[2] for q in pts], 1)
    return dict(k=float(k), b=float(b), a=float(a), c=float(c), n=len(pts), cx=float(k * 1240 + b), cy=float(a * 1240 + c))
def score_of(res): return float(np.mean([r["score"] for r in res.values()]))
probe = {r["video"]: r for r in csv.DictReader(open(os.path.join(ROOT, "dataset", f"band_probe_v2_{S}.csv"), encoding="utf-8-sig"))}
out_rows = []
for name in names:
    r0 = probe[name]; p = {k: float(r0[k]) for k in ("xL", "yL", "w", "h", "angle_deg")}
    cap = cv2.VideoCapture(os.path.join(VID, name + ".mp4")); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); frames = []
    for f in np.linspace(N * 0.2, N * 0.8, 12).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, img = cap.read()
        if ok: frames.append(img)
    cap.release()
    log = []; res = measure(frames, p); log.append(dict(it=0, p=dict(p), fit=fit(res), mean_score=score_of(res), res=res))
    print(name, "it0", {k: round(v, 3) for k, v in p.items()}, "score", round(score_of(res), 3), "fit", log[-1]["fit"], flush=True)
    for it in range(1, 7):
        F = log[-1]["fit"]
        if F is None: break
        if abs(F["k"]) < 0.004 and abs(F["a"]) < 0.003 and abs(F["cx"]) < 3 and abs(F["cy"]) < 3: break
        sx, sy = 2480 / p["w"], 120 / p["h"]
        cxs, cys = p["xL"] + p["w"] / 2 + F["cx"] / sx, p["yL"] + p["h"] / 2 + F["cy"] / sy
        w1 = p["w"] * (1 + F["k"]); h1 = p["h"] * (1 + F["k"]); dth = math.degrees(math.atan(F["a"] * sx / sy))
        cands = []
        for sgn in (+1, -1):
            q = dict(xL=cxs - w1 / 2, yL=cys - h1 / 2, w=w1, h=h1, angle_deg=p["angle_deg"] + sgn * dth)
            rq = measure(frames, q); fq = fit(rq)
            cands.append((abs(fq["a"]) if fq else 9, sgn, q, rq, fq))
        cands.sort(key=lambda z: z[0]); _, sgn, p, res, F1 = cands[0]
        log.append(dict(it=it, sign=sgn, p=dict(p), fit=F1, mean_score=score_of(res), res=res))
        print(name, f"it{it}", {k: round(v, 3) for k, v in p.items()}, "sign", sgn, "score", round(score_of(res), 3), "fit", F1, flush=True)
    best = max(log, key=lambda z: z["mean_score"])
    json.dump(dict(video=name, log=log, best_it=best["it"]), open(os.path.join(OUT, f"reprobe_{name}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    bp = best["p"]
    out_rows.append(dict(video=name, xL=int(round(bp["xL"])), yL=int(round(bp["yL"])), w=int(round(bp["w"])), h=int(round(bp["h"])), scale=r0["scale"],
                         angle_deg=round(bp["angle_deg"], 3), score=r0["score"], n_frames=len(frames),
                         note=f"再プローブ 2026-10-30（reprobe_band_20261030.py・反復{best['it']}・ラベル照合の平均 {best['mean_score']:.3f}・旧 xL{r0['xL']}/yL{r0['yL']}/w{r0['w']}/h{r0['h']}/angle{r0['angle_deg']}）"))
fp = os.path.join(OUT, "reprobe_rows.csv"); old = []
if os.path.exists(fp): old = [r for r in csv.DictReader(open(fp, encoding="utf-8-sig")) if r["video"] not in names]
with open(fp, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(old + out_rows)
print("REPROBE DONE", len(out_rows), fp, flush=True)
