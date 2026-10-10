# -*- coding: utf-8 -*-
"""9/15 再走査 第1段＝帯内 ROI 位置の再較正（2026-09-15・WSL cv2）。
   9/15 の帯動画はプローブの左端照合が外れた曲で数字の位置が roi_wao.json からずれる（Soranji＝傾き残り・天国＝左へ約35px）。
   9/11 CHEERS の帯フレームからカウンター名ラベル（アイコン〜文字）のテンプレを作り、各曲の帯フレーム（約10秒ごと）で
   Laplacian 上の NCC 照合（scale 0.94〜1.06）を行い、カウンター別の (dx,dy) を測る。
   出力: dataset/rescan_20260915/calib_samples.csv（全サンプル）・calib_summary.csv（曲×カウンターの中央値等）・
         roi/<video>.json（中央値で補正した ROI・読取器の argv[4] にそのまま渡す）・debug/<video>.png（補正 ROI を描いた帯フレーム）
   使い方: python rescan_calib_20260915.py [video_name ...]   （省略＝bands_wao/*_20260915_*.mp4 全部）"""
import os, sys, csv, json, glob, numpy as np, cv2
ROOT = "/mnt/g/研究アーカイブ/成果物/動画解析"; S = "20260915"
OUT = os.path.join(ROOT, "dataset", f"rescan_{S}"); os.makedirs(os.path.join(OUT, "roi"), exist_ok=True); os.makedirs(os.path.join(OUT, "debug"), exist_ok=True)
ROI = json.load(open(os.path.join(ROOT, "templates/wao_20260816/roi_wao.json")))
# ラベル領域（帯 2480x120 座標・9/11 CHEERS で目視）＝アイコン左端〜数字の手前
LAB = {"shakuri": (70, 260), "kobushi": (410, 640), "fall": (815, 1030), "vibrato": (1215, 1460), "accent": (1640, 1890), "hammering": (2075, 2340)}
TY0, TY1 = 28, 96; PADX, PADY = 120, 40  # 帯の外へはみ出す位置も探せるよう replicate で余白を付ける
REF_VIDEO = os.path.join(ROOT, "bands_wao", "CHEERS_20260911_105621.mp4"); REF_FRAMES = (2000, 3500, 6000)
def prep(g): return cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F, ksize=3)
def padded(g): return cv2.copyMakeBorder(g, PADY, PADY, PADX, PADX, cv2.BORDER_REPLICATE)
def gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cap = cv2.VideoCapture(REF_VIDEO); refs = []
for f in REF_FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
    if ok: refs.append(gray(img))
cap.release()
ref = np.median(np.stack(refs), axis=0).astype(np.uint8)
T = {c: ref[TY0:TY1, a:b] for c, (a, b) in LAB.items()}
SCALES = (0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06); SX = 220
def match(P, c):
    a, b = LAB[c]; X0 = max(0, a - SX + PADX); X1 = min(P.shape[1], b + SX + PADX); reg = P[:, X0:X1]; best = None
    for s in SCALES:
        t = prep(cv2.resize(T[c], (int((b - a) * s), int((TY1 - TY0) * s))))
        if t.shape[0] > reg.shape[0] or t.shape[1] > reg.shape[1]: continue
        r = cv2.matchTemplate(reg, t, cv2.TM_CCOEFF_NORMED); _, m, _, loc = cv2.minMaxLoc(r)
        if best is None or m > best[0]: best = (float(m), X0 + loc[0] - PADX - a, loc[1] - PADY - TY0, s)
    return best
names = [a for a in sys.argv[1:]] or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "bands_wao", f"*_{S}_*.mp4")))
samples = []; summary = []
for name in names:
    path = os.path.join(ROOT, "bands_wao", name + ".mp4"); cap = cv2.VideoCapture(path); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(300, N // 40); rows = {c: [] for c in LAB}; mid = None
    for f in range(300, N - 300, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
        if not ok: continue
        if mid is None or abs(f - N // 2) < abs(mid[0] - N // 2): mid = (f, img.copy())
        P = prep(padded(gray(img)))
        for c in LAB:
            m = match(P, c)
            if m is None: continue
            rows[c].append(m); samples.append(dict(video=name, frame=f, t=round(f / fps, 1), counter=c, score=round(m[0], 3), dx=m[1], dy=m[2], scale=m[3]))
    cap.release(); roi = {}; dbg = mid[1].copy() if mid else None
    for c in LAB:
        good = [r for r in rows[c] if r[0] >= 0.25] or rows[c]
        if not good: summary.append(dict(video=name, counter=c, n=0)); roi[c] = ROI[c]; continue
        dx = int(np.median([r[1] for r in good])); dy = int(np.median([r[2] for r in good]))
        sp = dict(video=name, counter=c, n=len(rows[c]), n_good=len(good), score_med=round(float(np.median([r[0] for r in good])), 3),
                  dx=dx, dy=dy, dx_min=min(r[1] for r in good), dx_max=max(r[1] for r in good), dy_min=min(r[2] for r in good), dy_max=max(r[2] for r in good),
                  scale_med=round(float(np.median([r[3] for r in good])), 3))
        summary.append(sp); x1, y1, x2, y2 = ROI[c]
        # 帯の縦 120px に収める（ROI 高さは維持）
        y1n = y1 + dy; y2n = y2 + dy
        if y1n < 0: y2n -= y1n; y1n = 0
        if y2n > 120: y1n -= (y2n - 120); y2n = 120
        roi[c] = [max(0, x1 + dx), y1n, min(2480, x2 + dx), y2n]
        if dbg is not None:
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 1); cv2.rectangle(dbg, (roi[c][0], roi[c][1]), (roi[c][2], roi[c][3]), (0, 255, 0), 1)
    json.dump(roi, open(os.path.join(OUT, "roi", name + ".json"), "w"), ensure_ascii=False)
    if dbg is not None: cv2.imwrite(os.path.join(OUT, "debug", name + ".png"), dbg)
    print(name, {c: (roi[c][0] - ROI[c][0], roi[c][1] - ROI[c][1]) for c in LAB}, flush=True)
    with open(os.path.join(OUT, "calib_samples.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["video", "frame", "t", "counter", "score", "dx", "dy", "scale"]); w.writeheader(); w.writerows(samples)
    with open(os.path.join(OUT, "calib_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        keys = ["video", "counter", "n", "n_good", "score_med", "dx", "dy", "dx_min", "dx_max", "dy_min", "dy_max", "scale_med"]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows([{k: r.get(k, "") for k in keys} for r in summary])
print("CALIB DONE", len(names))
