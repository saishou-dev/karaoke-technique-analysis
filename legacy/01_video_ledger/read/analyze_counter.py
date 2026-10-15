#!/usr/bin/env python3
"""カウンター増分読み取り v0.4 (2026-08-07)
v0.3からの修正:
- ROIから箱の枠線を除外(y85-148)
- 適応閾値: フレームごとに複数閾値を試し、綺麗にグリフが切れる閾値を採用
- 枠線等の非グリフ成分は「無視」(v0.3はフレームごと破棄していた=バグ)
"""
import cv2
import numpy as np
import sys
import os
import csv
from collections import Counter

VIDEO = sys.argv[1]
OUTDIR = sys.argv[2]

DIGIT_ROIS = {
    "shakuri": (245, 85, 350, 148),
    "kobushi": (565, 85, 660, 148),
    "fall":    (925, 85, 1020, 148),
    "vibrato": (1293, 85, 1392, 148),
}
THRESHOLDS = (215, 200, 185, 228, 170)
GLYPH_NORM = (28, 40)
SAME_T = 0.12
STABLE_MIN = 20

def crop(img, roi):
    x1, y1, x2, y2 = roi
    return img[y1:y2, x1:x2]

def try_extract(roi_gray, T):
    m = (roi_gray > T).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    glyphs = []
    junk_area = 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 28 <= h <= 52 and 4 <= w <= 40 and area >= 60:
            glyphs.append((x, y, w, h, i))
        else:
            if area > 30:
                junk_area += area
    if not (1 <= len(glyphs) <= 2):
        return None
    if junk_area > 250:  # 演出汚染
        return None
    glyphs.sort(key=lambda b: b[0])
    out = []
    for x, y, w, h, i in glyphs:
        gm = (lab[y:y+h, x:x+w] == i).astype(np.uint8)
        gm = cv2.resize(gm, GLYPH_NORM, interpolation=cv2.INTER_NEAREST)
        out.append(gm)
    return out

def extract_glyphs(roi_gray):
    for T in THRESHOLDS:
        r = try_extract(roi_gray, T)
        if r is not None:
            return r
    return None

K3 = np.ones((3, 3), np.uint8)

def norm_thickness(gm):
    """適応閾値による線の太さ揺れを正規化(白画素率を0.26-0.40に寄せる)"""
    for _ in range(3):
        r = float(gm.mean())
        if r < 0.26:
            gm = cv2.dilate(gm, K3)
        elif r > 0.40:
            gm = cv2.erode(gm, K3)
        else:
            break
    return gm

def soften(g):
    g = norm_thickness(g)
    return cv2.GaussianBlur(g.astype(np.float32), (5, 5), 0)

class GlyphLib:
    def __init__(self):
        self.classes = []  # [centroid(float), count]
    def classify(self, g):
        s = soften(g)
        best, bi = 1e9, -1
        for idx, (m, c) in enumerate(self.classes):
            d = float(np.mean(np.abs(s - m)))
            if d < best:
                best, bi = d, idx
        if bi >= 0 and best < SAME_T:
            m, c = self.classes[bi]
            self.classes[bi] = [(m * c + s) / (c + 1), c + 1]
            return bi
        self.classes.append([s, 1])
        return len(self.classes) - 1

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"fps={fps:.3f} frames={nframes} dur={nframes/fps:.1f}s")

lib = GlyphLib()
timeline = {k: [] for k in DIGIT_ROIS}
fidx = -1
while True:
    ok, img = cap.read()
    if not ok:
        break
    fidx += 1
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for name, roi in DIGIT_ROIS.items():
        gs = extract_glyphs(crop(g, roi))
        if gs is None:
            continue
        ids = tuple(lib.classify(x) for x in gs)
        timeline[name].append((fidx, ids))

print(f"glyph classes: {len(lib.classes)}")
for name in DIGIT_ROIS:
    r = len(timeline[name])
    print(f"{name}: readable {r}/{nframes} ({100*r/nframes:.0f}%)")

os.makedirs(OUTDIR, exist_ok=True)
reps = []
for idx, (m, c) in enumerate(lib.classes):
    im = cv2.resize((np.clip(m, 0, 1) * 255).astype(np.uint8), (GLYPH_NORM[0] * 2, GLYPH_NORM[1] * 2), interpolation=cv2.INTER_NEAREST)
    im = cv2.copyMakeBorder(im, 14, 2, 3, 3, cv2.BORDER_CONSTANT, value=60)
    cv2.putText(im, f"{idx}:{c}", (1, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, 255, 1)
    reps.append(im)
if reps:
    maxh = max(r.shape[0] for r in reps)
    reps = [cv2.copyMakeBorder(r, 0, maxh - r.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0) for r in reps]
    cv2.imwrite(os.path.join(OUTDIR, "glyph_classes.png"), np.hstack(reps))

events = {}
for name in DIGIT_ROIS:
    seq = timeline[name]
    stable_val = None
    run_val, run_n, run_start = None, 0, None
    evs = []
    for fr, ids in seq:
        if ids == run_val:
            run_n += 1
        else:
            run_val, run_n, run_start = ids, 1, fr
        if run_n == STABLE_MIN:
            if stable_val is not None and run_val != stable_val:
                evs.append((run_start, stable_val, run_val))
            stable_val = run_val
    events[name] = evs
    print(f"{name}: transitions {len(evs)}")

with open(os.path.join(OUTDIR, "transitions.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["counter", "frame", "time_sec", "from_ids", "to_ids"])
    rows = []
    for name, evs in events.items():
        for fr, a, b in evs:
            rows.append((fr, name, a, b))
    for fr, name, a, b in sorted(rows):
        w.writerow([name, fr, f"{fr/fps:.3f}", "+".join(map(str, a)), "+".join(map(str, b))])

with open(os.path.join(OUTDIR, "unreadable_segments.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["counter", "start_sec", "end_sec", "dur_sec"])
    for name in DIGIT_ROIS:
        readable = set(fr for fr, _ in timeline[name])
        seg = None
        for fr in range(nframes):
            if fr not in readable:
                if seg is None:
                    seg = fr
            else:
                if seg is not None and fr - seg >= fps * 0.5:
                    w.writerow([name, f"{seg/fps:.2f}", f"{(fr-1)/fps:.2f}", f"{(fr-seg)/fps:.2f}"])
                seg = None
        if seg is not None and nframes - seg >= fps * 0.5:
            w.writerow([name, f"{seg/fps:.2f}", f"{(nframes-1)/fps:.2f}", f"{(nframes-seg)/fps:.2f}"])
print("done")
