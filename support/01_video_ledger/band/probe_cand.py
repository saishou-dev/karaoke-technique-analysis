#!/usr/bin/env python3
"""候補矩形の実読みテスト: 多フレームでしゃくり・こぶし数字が読める率を出す"""
import cv2
import numpy as np
import sys

VIDEO = sys.argv[1]
PROBE_ROIS = [(245, 85, 350, 148), (565, 85, 690, 148)]
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

CANDS = []
for y in (600, 620, 640, 660, 680, 700, 720):
    for s in (1.0, 1.05, 1.1):
        CANDS.append((40, y, s))

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ts = [20, 45, 75, 110, 140, 170, 200, 230]
frames = []
for t in ts:
    fr = int(t * fps)
    if fr >= nframes:
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
    ok, img = cap.read()
    if ok:
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

best = None
for x, y, s in CANDS:
    w, h = int(1500 * s), int(180 * s)
    cnt = 0
    for g in frames:
        if y + h > g.shape[0] or x + w > g.shape[1]:
            continue
        band = cv2.resize(g[y:y+h, x:x+w], (1500, 180))
        for roi in PROBE_ROIS:
            x1, y1, x2, y2 = roi
            if glyph_readable(band[y1:y2, x1:x2]):
                cnt += 1
    if best is None or cnt > best[0]:
        best = (cnt, x, y, s)
    if cnt > 0:
        print(f"x={x} y={y} s={s} readable={cnt}/{2*len(frames)}")
c, x, y, s = best
print(f"BEST {x} {y} {int(1500*s)} {int(180*s)} {s}")
