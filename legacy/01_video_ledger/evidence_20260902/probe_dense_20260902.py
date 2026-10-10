#!/usr/bin/env python3
"""不安定3本の高密度再プローブ: 10フレームで2点照合→スケール中央値と読める数投票"""
import cv2, numpy as np, os, sys, json

SRC = "/mnt/d/研究アーカイブ/ビデオ素材/20260902"
REF_VIDEO = "/mnt/d/研究アーカイブ/ビデオ素材/20260812/エルダーフラワー_20260812_090419.mp4"
L_X0, L_X1, Y0, Y1 = 40, 255, 600, 720
R_X0, R_X1 = 2100, 2350
ROI = json.load(open("/home/toshiaki/karaoke-vision/roi_wao.json"))

def grab(video, t):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    cap.release()
    return img if ok else None

def edges(g):
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))

def readable_rois(img, xL, yL, scale):
    w = int(round(2480 * scale)); h = int(round(120 * scale))
    x0 = max(0, int(xL)); y0 = max(0, int(yL))
    sub = img[y0:y0+h, x0:x0+w]
    if sub.size == 0: return 0
    band = cv2.resize(sub, (2480, 120), interpolation=cv2.INTER_CUBIC)
    found = 0
    for name, (x1, y1, x2, y2) in ROI.items():
        r = band[y1:y2, x1:x2]
        g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        mx = r.max(axis=2)
        ok = False
        for s in (g, mx):
            for T in (215, 200, 185, 170, 150):
                m = (s > T).astype(np.uint8)
                n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
                boxes = [b for b in stats[1:] if 28 <= b[3] <= 58 and 4 <= b[2] <= 44 and b[4] >= 60]
                if 1 <= len(boxes) <= 2:
                    ok = True; break
            if ok: break
        found += ok
    return found

ref = grab(REF_VIDEO, 60.0)
tplL = edges(cv2.cvtColor(ref[Y0:Y1, L_X0:L_X1], cv2.COLOR_BGR2GRAY))
tplR = edges(cv2.cvtColor(ref[Y0:Y1, R_X0:R_X1], cv2.COLOR_BGR2GRAY))

video = sys.argv[1]
v = os.path.join(SRC, video + ".mp4")
cands = []
for t in range(20, 190, 15):
    img = grab(v, float(t))
    if img is None: continue
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    winL = edges(g[400:950, 0:700])
    _, sL, _, locL = cv2.minMaxLoc(cv2.matchTemplate(winL, tplL, cv2.TM_CCOEFF_NORMED))
    winR = edges(g[400:950, 1800:3200])
    _, sR, _, locR = cv2.minMaxLoc(cv2.matchTemplate(winR, tplR, cv2.TM_CCOEFF_NORMED))
    xL = locL[0]; yL = 400 + locL[1]; xR = 1800 + locR[0]
    scale = (xR - xL) / (R_X0 - L_X0)
    if not (0.90 <= scale <= 1.05):
        print(f"  t={t} xL={xL} yL={yL} scale={scale:.4f} REJECT(range)")
        continue
    k = readable_rois(img, xL, yL, scale)
    cands.append((xL, yL, scale, sL, sR, k, t))
    print(f"  t={t} xL={xL} yL={yL} scale={scale:.4f} sL={sL:.2f} sR={sR:.2f} readable={k}/6")
if cands:
    best = sorted(cands, key=lambda c: (-c[5], -(c[3] + c[4])))
    good = [c for c in cands if c[5] >= 5]
    pool = good if good else cands
    xs = sorted(c[0] for c in pool); ys = sorted(c[1] for c in pool); ss = sorted(c[2] for c in pool)
    n = len(pool)
    print(f"MEDIAN(readable>=5 n={n}): xL={xs[n//2]} yL={ys[n//2]} scale={ss[n//2]:.4f}")
