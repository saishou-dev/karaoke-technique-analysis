#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import os

VIDEO = sys.argv[1]
OUTDIR = sys.argv[2]
ROI = (245, 60, 350, 150)  # shakuri digits

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
for t in [40, 70, 100, 130, 160]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, img = cap.read()
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = ROI
    roi = g[y1:y2, x1:x2]
    print(f"--- t={t}s roi gray: min={roi.min()} max={roi.max()} mean={roi.mean():.0f} p90={np.percentile(roi,90):.0f}")
    for T in [160, 175, 190, 205, 220]:
        m = (roi > T).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        comps = [(s[0], s[1], s[2], s[3], s[4]) for s in stats[1:]]
        comps.sort(key=lambda c: -c[4])
        top = "; ".join(f"x{c[0]},y{c[1]},w{c[2]},h{c[3]},a{c[4]}" for c in comps[:4])
        print(f"  T={T}: ncomp={n-1} white%={100*m.mean():.1f} top:[{top}]")
    cv2.imwrite(os.path.join(OUTDIR, f"probe_gray_t{t}.png"), roi)
    cv2.imwrite(os.path.join(OUTDIR, f"probe_mask190_t{t}.png"), (roi > 190).astype(np.uint8) * 255)
print("done")
