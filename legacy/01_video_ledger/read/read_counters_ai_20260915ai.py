#!/usr/bin/env python3
"""v0.8 カウンター読み取り 本実装
グレースケールNCC digit分類(閾値非依存) → 整数タイムライン → 単調増加制約で確定
"""
import cv2
import numpy as np
import sys
import os
import csv
import json

VIDEO = sys.argv[1]
OUTDIR = sys.argv[2]
REPS_NPY = sys.argv[3]
ROI_OVERRIDE = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else {}

# harvest.py のクラスタ index → 数字 (None=ノイズ)
LABELS = {0: 0, 1: 2, 2: 3, 3: 1, 4: 2, 5: 6, 6: 1, 7: 2, 8: 1, 9: 7,
          10: 4, 11: 5, 12: 6, 13: 9, 14: None, 15: 5, 16: 0, 17: 0, 18: None,
          19: 2, 20: 1, 21: 6, 22: 1, 23: 0, 24: 2, 25: 2, 26: 2, 27: 2,
          28: None, 29: 6, 30: 2, 31: 8, 32: 2, 33: 4, 34: 0, 35: 4,
          36: None, 37: 7, 38: None, 39: 8}

DIGIT_ROIS = {
    "shakuri": (245, 85, 350, 148),
    "kobushi": (565, 85, 690, 148),
    "fall":    (925, 85, 1020, 148),
    "vibrato": (1230, 85, 1460, 148),
}
for k, v in ROI_OVERRIDE.items():
    DIGIT_ROIS[k] = tuple(v)
THRESHOLDS = (215, 200, 185, 228, 170)
NORM = (28, 40)
NCC_MIN = 0.75
STABLE_MIN = 6  # 同一値がこの回数読めたら安定値

def glyph_boxes(roi_gray):
    for T in THRESHOLDS:
        m = (roi_gray > T).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        boxes = []
        junk = 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if 28 <= h <= 58 and 4 <= w <= 44 and area >= 60:
                boxes.append((x, y, w, h))
            elif area > 30:
                junk += area
        if 1 <= len(boxes) <= 2 and junk <= 1200:
            boxes.sort(key=lambda b: b[0])
            if len(boxes) == 2:
                gap = boxes[1][0] - (boxes[0][0] + boxes[0][2])
                hdiff = abs(boxes[0][3] - boxes[1][3])
                ydiff = abs(boxes[0][1] - boxes[1][1])
                if gap > 14 or gap < -4 or hdiff > 8 or ydiff > 6:
                    return None  # 2桁として不自然(キラキラ等の混入)
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

reps = np.load(REPS_NPY)  # (40, 40, 28)
reps_flat = []
for r in reps:
    v = r.ravel() - r.mean()
    reps_flat.append(v / max(1e-6, np.linalg.norm(v)))
reps_flat = np.stack(reps_flat)

def classify(gl):
    v = gl.ravel() - gl.mean()
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    scores = reps_flat @ (v / n)
    bi = int(np.argmax(scores))
    if scores[bi] < NCC_MIN:
        return None
    return LABELS.get(bi)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"fps={fps:.3f} frames={nframes} dur={nframes/fps:.1f}s")

values = {k: [] for k in DIGIT_ROIS}  # (frame, int)
fidx = -1
while True:
    ok, img = cap.read()
    if not ok:
        break
    fidx += 1
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for name, roi in DIGIT_ROIS.items():
        x1, y1, x2, y2 = roi
        rg = g[y1:y2, x1:x2]
        boxes = glyph_boxes(rg)
        if boxes is None:
            continue
        digits = []
        okread = True
        for b in boxes:
            d = classify(norm_glyph(rg, b))
            if d is None:
                okread = False
                break
            digits.append(d)
        if not okread:
            continue
        val = digits[0] if len(digits) == 1 else digits[0] * 10 + digits[1]
        values[name].append((fidx, val))

for name in DIGIT_ROIS:
    print(f"{name}: readable {len(values[name])}/{nframes} ({100*len(values[name])/nframes:.0f}%)")

# 安定ラン抽出(滞在フレーム数=重み) → 重み付き最長非減少部分列(WLIS)で真の軌跡を選択
results = {}
for name in DIGIT_ROIS:
    seq = values[name]
    runs = []  # [first_frame, value, weight]
    run_v = None
    for fr, v in seq:
        if run_v is not None and v == run_v:
            runs[-1][2] += 1
        else:
            runs.append([fr, v, 1])
            run_v = v
    runs = [r for r in runs if r[2] >= STABLE_MIN]
    n = len(runs)
    accepted = []
    if n:
        dp = [0.0] * n
        prev = [-1] * n
        for i in range(n):
            dp[i] = runs[i][2]
            for j in range(i):
                if runs[j][1] <= runs[i][1] and dp[j] + runs[i][2] > dp[i]:
                    dp[i] = dp[j] + runs[i][2]
                    prev[i] = j
        best = int(np.argmax(dp))
        path = []
        while best != -1:
            path.append(best)
            best = prev[best]
        path.reverse()
        cur = None
        for idx in path:
            fr, v, wgt = runs[idx]
            if cur is None:
                accepted.append((fr, v, 0))
                cur = v
            elif v > cur:
                accepted.append((fr, v, v - cur))
                cur = v
    results[name] = accepted
    ev = sum(d for _, _, d in accepted)
    ini = accepted[0][1] if accepted else None
    fin = accepted[-1][1] if accepted else None
    rej = n - len(accepted)
    print(f"{name}: initial={ini} final={fin} increments={ev} runs={n} off_path={rej}")

# 遷移窓: 真のイベント時刻はカウンターフラッシュ開始≒「旧値が最後に読めたフレーム」直後
# (較正実測 2026/8/8: 数字は加点の瞬間にフラッシュ付きで変化し、安定読取はフラッシュ明け≈+0.5s)
with open(os.path.join(OUTDIR, "counter_events.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["counter", "frame", "time_sec", "value", "delta",
                "event_frame_est", "event_time_est"])
    rows = []
    for name, acc in results.items():
        seq = values[name]
        for i, (fr, v, d) in enumerate(acc):
            ev_fr = fr
            if i > 0:
                pv = acc[i - 1][1]
                last_old = None
                for f2, v2 in seq:
                    if f2 >= fr:
                        break
                    if v2 == pv:
                        last_old = f2
                if last_old is not None:
                    ev_fr = last_old + 1
            rows.append((fr, name, v, d, ev_fr))
    for fr, name, v, d, ev_fr in sorted(rows):
        w.writerow([name, fr, f"{fr/fps:.3f}", v, d, ev_fr, f"{ev_fr/fps:.3f}"])
json.dump({k: [[fr, v, d] for fr, v, d in acc] for k, acc in results.items()},
          open(os.path.join(OUTDIR, "counter_events.json"), "w"))
print("done")
