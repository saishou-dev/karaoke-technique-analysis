#!/usr/bin/env python3
"""カウンター帯の自動位置特定 v2
- 複数スキンの基準テンプレ(金枠/暗色)で多スケール探索
- 上位候補を「実際に数字グリフが読めるか」で自己検証して採用
"""
import cv2
import numpy as np
import sys

VIDEO = sys.argv[1]
TEMPLATES = sys.argv[2:]  # 基準帯画像(1500x180)を複数

PROBE_ROIS = [(245, 85, 350, 148), (565, 85, 690, 148)]  # しゃくり・こぶし数字
THRESHOLDS = (215, 200, 185, 228, 170)

def glyph_readable(roi_gray):
    for T in THRESHOLDS:
        m = (roi_gray > T).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        boxes = 0
        junk = 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if 28 <= h <= 58 and 4 <= w <= 44 and area >= 60:
                boxes += 1
            elif area > 30:
                junk += area
        if 1 <= boxes <= 2 and junk <= 1200:
            return True
    return False

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
dur = nframes / fps

# 1) 候補収集
sample_ts = [t for t in (30, 60, 90, 120, 150) if t < dur - 5]
frames = {}
for t in sample_ts:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, img = cap.read()
    if ok:
        frames[t] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cands = {}  # (x20,y20,s) -> [score,...]
for tp in TEMPLATES:
    tpl_full = cv2.imread(tp, cv2.IMREAD_GRAYSCALE)
    tpl = tpl_full[:, 0:560]
    for t, g in frames.items():
        region = g[250:1300, 0:2400]
        for s in np.arange(0.6, 1.4, 0.05):
            tw, th = int(tpl.shape[1] * s), int(tpl.shape[0] * s)
            if tw < 60 or th < 24 or tw > region.shape[1] or th > region.shape[0]:
                continue
            tt = cv2.resize(tpl, (tw, th))
            res = cv2.matchTemplate(region, tt, cv2.TM_CCOEFF_NORMED)
            mn, mx, mnl, mxl = cv2.minMaxLoc(res)
            if mx < 0.30:
                continue
            key = (round(mxl[0] / 25), round((mxl[1] + 250) / 25), round(s, 2))
            cands.setdefault(key, []).append((mx, mxl[0], mxl[1] + 250, s))

# 2) 候補を集約し上位を自己検証
agg = []
for key, hits in cands.items():
    score = max(h[0] for h in hits) + 0.05 * len(hits)
    x = int(np.median([h[1] for h in hits]))
    y = int(np.median([h[2] for h in hits]))
    s = float(np.median([h[3] for h in hits]))
    agg.append((score, x, y, s))
agg.sort(key=lambda a: -a[0])

best = None
for score, x, y, s in agg[:6]:
    w, h = int(1500 * s), int(180 * s)
    readable = 0
    for t, g in frames.items():
        if y + h > g.shape[0] or x + w > g.shape[1]:
            continue
        band = cv2.resize(g[y:y+h, x:x+w], (1500, 180))
        for roi in PROBE_ROIS:
            x1, y1, x2, y2 = roi
            if glyph_readable(band[y1:y2, x1:x2]):
                readable += 1
    print(f"cand x={x} y={y} s={s:.2f} score={score:.3f} readable={readable}/{2*len(frames)}")
    if best is None or readable > best[0]:
        best = (readable, x, y, s)

if best is None or best[0] == 0:
    print("LOCATE_FAIL")
    sys.exit(1)
r, x, y, s = best
w, h = int(1500 * s), int(180 * s)
print(f"BAND {x} {y} {w} {h} {s:.3f} {r}")
