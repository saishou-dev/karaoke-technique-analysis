#!/usr/bin/env python3
"""9/2帯の縦ずれ実測: 現行クロップ枠(band_scale_final_20260902.csv)を上下+40px広げて切り出し→2480幅へ正規化し、
8/18エルダー基準帯(完全一致で読めた帯)の左ラベル/右ラベル領域をエッジ照合してdyL/dyR(帯座標px)を求める。
出力: measure_dy_20260902.csv (video, scale, dyL, dyR, dxL, dxR, per-frame)"""
import cv2, numpy as np, csv, os, sys
SRC = "/mnt/d/研究アーカイブ/ビデオ素材/20260902"
REFB = "/mnt/d/研究アーカイブ/成果物/動画解析/bands_wao/エルダーフラワー_20260818_125624.mp4"
CSV = "/home/toshiaki/karaoke-vision/band_scale_final_20260902.csv"
OUT = "/home/toshiaki/karaoke-vision/measure_dy_20260902.csv"
PAD = 40
TS = (45.0, 90.0, 150.0, 200.0)
def grab(v, t):
    cap = cv2.VideoCapture(v); cap.set(cv2.CAP_PROP_POS_MSEC, t*1000); ok, im = cap.read(); cap.release(); return im if ok else None
def edges(g):
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
ref = grab(REFB, 120.0)
rg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
# ラベル領域(数字を含まない): しゃくりラベル x20-250 / ハンマリングラベル x2130-2350, y10-110
LX0, LX1, RX0, RX1, Y0, Y1 = 20, 250, 2130, 2350, 10, 110
tL = edges(rg[Y0:Y1, LX0:LX1]); tR = edges(rg[Y0:Y1, RX0:RX1])
only = sys.argv[1] if len(sys.argv) > 1 else None
rows = []
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    name = r["video"]
    if only and name != only: continue
    xL, yL, sc = int(float(r["xL"])), int(float(r["yL"])), float(r["scale"])
    W, H = int(round(2480*sc)), int(round(120*sc))
    per = []
    for t in TS:
        im = grab(os.path.join(SRC, name + ".mp4"), t)
        if im is None: continue
        y0 = max(0, yL - PAD); x0 = max(0, xL)
        sub = im[y0:yL+H+PAD, x0:x0+W]
        if sub.shape[1] < W: continue
        s = 2480.0 / W
        tall = cv2.resize(sub, (2480, int(round(sub.shape[0]*s))), interpolation=cv2.INTER_CUBIC)
        base = int(round((yL - y0) * s))  # 現行枠の上端が来る帯座標
        g = cv2.cvtColor(tall, cv2.COLOR_BGR2GRAY)
        eL = edges(g[:, 0:600]); eR = edges(g[:, 1900:2480])
        _, sL, _, lL = cv2.minMaxLoc(cv2.matchTemplate(eL, tL, cv2.TM_CCOEFF_NORMED))
        _, sR, _, lR = cv2.minMaxLoc(cv2.matchTemplate(eR, tR, cv2.TM_CCOEFF_NORMED))
        dyL = (lL[1] - Y0) - base; dxL = lL[0] - LX0
        dyR = (lR[1] - Y0) - base; dxR = (1900 + lR[0]) - RX0
        per.append((dyL, dyR, dxL, dxR, round(sL, 2), round(sR, 2)))
    if not per:
        print(name, "NOFRAME"); continue
    med = lambda i: sorted(p[i] for p in per)[len(per)//2]
    print(f"{name} scale={sc} dyL={med(0):+d} dyR={med(1):+d} dxL={med(2):+d} dxR={med(3):+d} per={per}", flush=True)
    rows.append((name, sc, med(0), med(1), med(2), med(3), str(per)))
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["video", "scale", "dyL", "dyR", "dxL", "dxR", "per_frame"]); w.writerows(rows)
print("saved", OUT)
