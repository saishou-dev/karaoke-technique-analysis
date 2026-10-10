#!/usr/bin/env python3
"""9/2用: 左(しゃくり)・右(ハンマリング)ラベル2点照合でスケール+オフセット実測(D18方式)
probe_scale_20260818.pyの20260902適応(SRC/出力名のみ変更)。ref=エルダーフラワー_20260812(基準帯)"""
import cv2, numpy as np, glob, os, csv

SRC = "/mnt/d/研究アーカイブ/ビデオ素材/20260902"
REF_VIDEO = "/mnt/d/研究アーカイブ/ビデオ素材/20260812/エルダーフラワー_20260812_090419.mp4"
L_X0, L_X1, Y0, Y1 = 40, 255, 600, 720
R_X0, R_X1 = 2100, 2350
SXL0, SXL1 = 0, 700
SY0, SY1 = 400, 950
TS = (45.0, 90.0, 150.0)

def grab(video, t):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    cap.release()
    return img if ok else None

def edges(g):
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))

ref = grab(REF_VIDEO, 60.0)
tplL = edges(cv2.cvtColor(ref[Y0:Y1, L_X0:L_X1], cv2.COLOR_BGR2GRAY))
tplR = edges(cv2.cvtColor(ref[Y0:Y1, R_X0:R_X1], cv2.COLOR_BGR2GRAY))

rows = []
for v in sorted(glob.glob(os.path.join(SRC, "*.mp4"))):
    name = os.path.basename(v)[:-4]
    if name.startswith("VID_"):
        continue
    per = []
    for t in TS:
        img = grab(v, t)
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        winL = edges(g[SY0:SY1, SXL0:SXL1])
        resL = cv2.matchTemplate(winL, tplL, cv2.TM_CCOEFF_NORMED)
        _, sL, _, locL = cv2.minMaxLoc(resL)
        SXR0 = 1800
        winR = edges(g[SY0:SY1, SXR0:3200])
        resR = cv2.matchTemplate(winR, tplR, cv2.TM_CCOEFF_NORMED)
        _, sR, _, locR = cv2.minMaxLoc(resR)
        xL = SXL0 + locL[0]
        yL = SY0 + locL[1]
        xR = SXR0 + locR[0]
        scale = (xR - xL) / (R_X0 - L_X0)
        per.append((xL, yL, xR, scale, sL, sR))
    if not per:
        print(name, "NOFRAME", flush=True)
        rows.append((name, "", "", "", ""))
        continue
    scales = sorted(p[3] for p in per)
    med = per[[p[3] for p in per].index(scales[len(scales) // 2])]
    xL, yL, xR, scale, sL, sR = med
    print(f"{name} xL={xL} yL={yL} xR={xR} scale={scale:.4f} scoreL={sL:.2f} scoreR={sR:.2f} allxL={[p[0] for p in per]} allyL={[p[1] for p in per]} allsc={[round(p[3],4) for p in per]}", flush=True)
    rows.append((name, xL, yL, f"{scale:.4f}", f"{min(sL, sR):.2f}"))

with open("/home/toshiaki/karaoke-vision/band_scale_20260902.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["video", "xL", "yL", "scale", "score"])
    w.writerows(rows)
print("saved band_scale_20260902.csv")
