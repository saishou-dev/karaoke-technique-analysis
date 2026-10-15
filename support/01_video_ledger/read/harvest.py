#!/usr/bin/env python3
"""v0.7 数字テンプレート収穫: 1秒ごとにグリフのグレースケール切片を収集し
粗クラスタリング→代表を1枚のモンタージュに出力(人手ラベル用)
"""
import cv2
import numpy as np
import sys
import os

VIDEO = sys.argv[1]
OUTDIR = sys.argv[2]

DIGIT_ROIS = {
    "shakuri": (245, 85, 350, 148),
    "kobushi": (565, 85, 660, 148),
    "fall":    (925, 85, 1020, 148),
    "vibrato": (1293, 85, 1392, 148),
}
THRESHOLDS = (215, 200, 185, 228, 170)
NORM = (28, 40)

def glyph_boxes(roi_gray):
    for T in THRESHOLDS:
        m = (roi_gray > T).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        boxes = []
        junk = 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if 28 <= h <= 52 and 4 <= w <= 40 and area >= 60:
                boxes.append((x, y, w, h))
            elif area > 30:
                junk += area
        if 1 <= len(boxes) <= 2 and junk <= 250:
            boxes.sort(key=lambda b: b[0])
            return boxes
    return None

def norm_glyph(roi_gray, box):
    x, y, w, h = box
    pad = 2
    y0, y1 = max(0, y - pad), min(roi_gray.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(roi_gray.shape[1], x + w + pad)
    g = roi_gray[y0:y1, x0:x1].astype(np.float32)
    g = (g - g.min()) / max(1.0, (g.max() - g.min()))
    return cv2.resize(g, NORM)

def ncc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-6 else 0.0

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
clusters = []  # [rep, count, src_counter]
fidx = -1
while True:
    ok, img = cap.read()
    if not ok:
        break
    fidx += 1
    if fidx % 60 != 0:
        continue
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for name, roi in DIGIT_ROIS.items():
        x1, y1, x2, y2 = roi
        rg = g[y1:y2, x1:x2]
        boxes = glyph_boxes(rg)
        if boxes is None:
            continue
        for b in boxes:
            gl = norm_glyph(rg, b)
            best, bi = -1, -1
            for i, (rep, c, src) in enumerate(clusters):
                s = ncc(gl, rep)
                if s > best:
                    best, bi = s, i
            if bi >= 0 and best > 0.85:
                clusters[bi][1] += 1
            else:
                clusters.append([gl, 1, name])

clusters.sort(key=lambda c: -c[1])
print(f"clusters: {len(clusters)}")
reps = []
for i, (rep, c, src) in enumerate(clusters[:40]):
    im = (rep * 255).astype(np.uint8)
    im = cv2.resize(im, (NORM[0] * 2, NORM[1] * 2), interpolation=cv2.INTER_NEAREST)
    im = cv2.copyMakeBorder(im, 16, 2, 3, 3, cv2.BORDER_CONSTANT, value=40)
    cv2.putText(im, f"{i}:{c}", (1, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)
    reps.append(im)
np.save(os.path.join(OUTDIR, "cluster_reps.npy"), np.stack([c[0] for c in clusters[:40]]))
cv2.imwrite(os.path.join(OUTDIR, "digit_clusters.png"), np.hstack(reps))
for i, (rep, c, src) in enumerate(clusters[:40]):
    print(i, c, src)
print("done")
