# -*- coding: utf-8 -*-
"""【2026-10-30 派生・案1（10/22 報告 4(3) の案文）】calib_roi_20260915.py の写し＋次の2点だけを変えたもの（元ファイルは無変更）。
   (1) 照合スコア中央値が KV_CALIB_MINSCORE（既定 0.25）未満のカウンターは較正を捨てて既定 ROI（applied=0）。
   (2) ROI の幅・高さを保ってクランプ（負幅・はみ出しを作らない）。帯動画の実幅を使う（2480 の決め打ちをやめる）。
   比較用（close4_20261030 の候補 R1）。出力先は環境変数 KV_RESULTS_DIR／KV_CALIB_DIR で別ディレクトリに出す。
"""
import os, sys, csv, json, glob, numpy as np, cv2
ROOT = "/mnt/g/研究アーカイブ/成果物/動画解析"
S = next((a for a in sys.argv[1:] if a.isdigit() and len(a) == 8), None) or os.environ.get("KV_SESSION", "20260915")
names = [a for a in sys.argv[1:] if a != S]
BANDS = os.environ.get("KV_BANDS_DIR") or os.path.join(ROOT, "bands_wao")
RES = os.environ.get("KV_RESULTS_DIR") or os.path.join(ROOT, f"results_wao_{S}")
OUT = os.environ.get("KV_CALIB_DIR") or os.path.join(ROOT, "dataset", f"calib_{S}")
FORCE = os.environ.get("KV_FORCE", "") == "1"
DEADBAND = int(os.environ.get("KV_CALIB_DEADBAND", "10"))
MINSCORE = float(os.environ.get("KV_CALIB_MINSCORE", "0.25"))
os.makedirs(os.path.join(OUT, "debug"), exist_ok=True)
ROI = json.load(open(os.path.join(ROOT, "templates/wao_20260816/roi_wao.json")))
LAB = {"shakuri": (70, 260), "kobushi": (410, 640), "fall": (815, 1030), "vibrato": (1215, 1460), "accent": (1640, 1890), "hammering": (2075, 2340)}
TY0, TY1 = 28, 96; PADX, PADY = 120, 40
REF_VIDEO = os.path.join(ROOT, "bands_wao", "CHEERS_20260911_105621.mp4"); REF_FRAMES = (2000, 3500, 6000)
def prep(g): return cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F, ksize=3)
def padded(g): return cv2.copyMakeBorder(g, PADY, PADY, PADX, PADX, cv2.BORDER_REPLICATE)
def gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cap = cv2.VideoCapture(REF_VIDEO); refs = []
for f in REF_FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
    if ok: refs.append(gray(img))
cap.release()
if len(refs) < 3: print("CALIB STOP: 参照帯（9/11 CHEERS）が読めない", REF_VIDEO, flush=True); raise SystemExit(2)
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
if not names: names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(BANDS, f"*_{S}_*.mp4")) if ".part" not in p)
samples = []; summary = []; n_done = 0; n_skip = 0
sp = os.path.join(OUT, "calib_samples.csv"); su = os.path.join(OUT, "calib_summary.csv")
if os.path.exists(sp): samples = [r for r in csv.DictReader(open(sp, encoding="utf-8-sig")) if r["video"] not in names or (not FORCE and os.path.exists(os.path.join(RES, r["video"], "roi.json")))]
if os.path.exists(su): summary = [r for r in csv.DictReader(open(su, encoding="utf-8-sig")) if r["video"] not in names or (not FORCE and os.path.exists(os.path.join(RES, r["video"], "roi.json")))]
for name in names:
    outj = os.path.join(RES, name, "roi.json")
    if os.path.exists(outj) and not FORCE: n_skip += 1; continue
    path = os.path.join(BANDS, name + ".mp4"); cap = cv2.VideoCapture(path); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 59.0
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
        dx = int(np.median([r[1] for r in good])); dy = int(np.median([r[2] for r in good])); applied = int(max(abs(dx), abs(dy)) > DEADBAND)
        if float(np.median([r[0] for r in good])) < MINSCORE: applied = 0  # 案1：低スコアは較正を捨てて既定 ROI
        summary.append(dict(video=name, counter=c, n=len(rows[c]), n_good=len(good), score_med=round(float(np.median([r[0] for r in good])), 3),
                            dx=dx, dy=dy, applied=applied, dx_min=min(r[1] for r in good), dx_max=max(r[1] for r in good), dy_min=min(r[2] for r in good), dy_max=max(r[2] for r in good),
                            scale_med=round(float(np.median([r[3] for r in good])), 3)))
        if not applied: roi[c] = ROI[c]; continue  # 不感帯内＝既定 ROI
        x1, y1, x2, y2 = ROI[c]; y1n = y1 + dy; y2n = y2 + dy; Wb = P.shape[1] - 2 * PADX; Hb = P.shape[0] - 2 * PADY
        if y1n < 0: y2n -= y1n; y1n = 0
        if y2n > Hb: y1n -= (y2n - Hb); y2n = Hb
        x1n, x2n = x1 + dx, x2 + dx; w0 = x2 - x1
        if x1n < 0: x1n, x2n = 0, w0
        if x2n > Wb: x1n, x2n = Wb - w0, Wb
        roi[c] = [x1n, y1n, x2n, y2n]  # 案1：幅を保ってクランプ
        if dbg is not None:
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 1); cv2.rectangle(dbg, (roi[c][0], roi[c][1]), (roi[c][2], roi[c][3]), (0, 255, 0), 1)
    os.makedirs(os.path.join(RES, name), exist_ok=True)
    json.dump(roi, open(outj + ".part", "w"), ensure_ascii=False); os.replace(outj + ".part", outj)
    if dbg is not None: cv2.imwrite(os.path.join(OUT, "debug", name + ".png"), dbg)
    n_done += 1
    print("CALIB", name, {c: (roi[c][0] - ROI[c][0], roi[c][1] - ROI[c][1]) for c in LAB}, f"deadband={DEADBAND}", flush=True)
    with open(sp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["video", "frame", "t", "counter", "score", "dx", "dy", "scale"]); w.writeheader(); w.writerows(samples)
    with open(su, "w", newline="", encoding="utf-8-sig") as f:
        keys = ["video", "counter", "n", "n_good", "score_med", "dx", "dy", "applied", "dx_min", "dx_max", "dy_min", "dy_max", "scale_med"]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows([{k: r.get(k, "") for k in keys} for r in summary])
print(f"CALIB DONE session={S} calibrated={n_done} skipped={n_skip} total={len(names)}", flush=True)
