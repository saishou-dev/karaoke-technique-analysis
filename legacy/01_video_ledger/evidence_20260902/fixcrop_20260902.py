#!/usr/bin/env python3
"""9/2帯クロップの傾き補正(カメラロール): measure_dy_20260902.csv(dyL,dyR 帯座標px)から
ffmpegフィルタ列 crop(上下pad付き) -> rotate(微小角) -> crop -> scale=2480:120 を各動画に生成。
usage: fixcrop_20260902.py test <sign>      # エルダー12:11 t=120 の1フレームで補正後のdyL/dyRを実測
       fixcrop_20260902.py write <sign>     # band_fix_final_20260902.csv を出力(video,vf,note)
"""
import cv2, numpy as np, csv, os, sys, math, subprocess
KV = "/home/toshiaki/karaoke-vision"
SRC = "/mnt/d/研究アーカイブ/ビデオ素材/20260902"
SRCWIN = r"D:\研究アーカイブ\ビデオ素材\20260902"
REFB = "/mnt/d/研究アーカイブ/成果物/動画解析/bands_wao/エルダーフラワー_20260818_125624.mp4"
FF = "/mnt/c/Users/conan/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0-full_build/bin/ffmpeg.exe"
SP = "/mnt/c/Users/conan/AppData/Local/Temp/claude/C--/285c4693-3dbc-4e3b-bc86-028a075fc040/scratchpad"
SPWIN = r"C:\Users\conan\AppData\Local\Temp\claude\C--\285c4693-3dbc-4e3b-bc86-028a075fc040\scratchpad"
PAD = 48
XLBL, XRBL = 135.0, 2240.0   # 測定に使ったラベル中心(帯座標)
mode, sign = sys.argv[1], float(sys.argv[2])

base = {r["video"]: r for r in csv.DictReader(open(f"{KV}/band_scale_final_20260902.csv", encoding="utf-8"))}
meas = {r["video"]: r for r in csv.DictReader(open(f"{KV}/measure_dy_20260902.csv", encoding="utf-8"))}

def params(name):
    b, m = base[name], meas[name]
    xL, yL, sc = int(float(b["xL"])), int(float(b["yL"])), float(b["scale"])
    dyL, dyR = float(m["dyL"]), float(m["dyR"])
    W, H = int(round(2480 * sc)), int(round(120 * sc))
    slope = (dyR - dyL) / (XRBL - XLBL)           # スケール不変
    ang = math.atan(slope)                         # rad
    yT = yL - PAD
    Ht = H + 2 * PAD
    yc = PAD + dyL * sc + slope * (W / 2.0 - XLBL * sc)   # 回転後(中心不変)の帯上端
    yc_i = int(round(yc))
    vf = (f"crop={W}:{Ht}:{xL}:{yT},rotate={sign*ang:.6f}:ow=iw:oh=ih:c=black,"
          f"crop={W}:{H}:0:{yc_i},scale=2480:120")
    return vf, dict(xL=xL, yL=yL, sc=sc, dyL=dyL, dyR=dyR, ang_deg=math.degrees(ang), yc=yc_i, W=W, H=H)

def edges(g):
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))

def measure_band(img):
    cap = cv2.VideoCapture(REFB); cap.set(cv2.CAP_PROP_POS_MSEC, 120000); ok, ref = cap.read(); cap.release()
    rg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    LX0, LX1, RX0, RX1, Y0, Y1 = 20, 250, 2130, 2350, 10, 110
    tL = edges(rg[Y0:Y1, LX0:LX1]); tR = edges(rg[Y0:Y1, RX0:RX1])
    # 上下に余白を足して探索(帯は120行しかないため)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    padv = 40
    g = cv2.copyMakeBorder(g, padv, padv, 0, 0, cv2.BORDER_CONSTANT, value=0)
    eL = edges(g[:, 0:600]); eR = edges(g[:, 1900:2480])
    _, sL, _, lL = cv2.minMaxLoc(cv2.matchTemplate(eL, tL, cv2.TM_CCOEFF_NORMED))
    _, sR, _, lR = cv2.minMaxLoc(cv2.matchTemplate(eR, tR, cv2.TM_CCOEFF_NORMED))
    return (lL[1] - padv - Y0, lR[1] - padv - Y0, lL[0] - LX0, 1900 + lR[0] - RX0, round(sL, 2), round(sR, 2))

if mode == "test":
    name = sys.argv[3] if len(sys.argv) > 3 else "エルダーフラワー_20260902_121130"
    for t in (120, 200):
        vf, p = params(name)
        out = f"{SP}/fixtest_{sign:+.0f}_{t}.png"
        outwin = f"{SPWIN}\\fixtest_{sign:+.0f}_{t}.png"
        cmd = [FF, "-nostdin", "-v", "error", "-y", "-ss", str(t), "-i", f"{SRCWIN}\\{name}.mp4",
               "-frames:v", "1", "-vf", vf, outwin]
        subprocess.run(cmd, check=True)
        img = cv2.imread(out)
        print(name, "t=", t, "sign=", sign, "params=", p)
        print("  after-fix dyL,dyR,dxL,dxR,sL,sR =", measure_band(img))
elif mode == "write":
    rows = []
    for name in base:
        vf, p = params(name)
        rows.append((name, vf, f"roll={p['ang_deg']:.3f}deg dyL={p['dyL']:+.0f} dyR={p['dyR']:+.0f} yc={p['yc']} PAD={PAD}"))
        print(name, rows[-1][2])
    with open(f"{KV}/band_fix_final_20260902.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["video", "vf", "note"]); w.writerows(rows)
    with open(f"{SP}/band_fix_final_20260902.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["video", "vf", "note"]); w.writerows(rows)
    print("saved band_fix_final_20260902.csv", len(rows))
