# -*- coding: utf-8 -*-
"""案D（mark-pos-time）の下調べ・2026-11-05 Opus
指定曲・指定時刻から連続フレームを detect_marks.py と同じ ROI（4K 原画素・crop 原点を足し戻す座標）で復号し、
(1) 再生カーソル x（detect_marks.find_cursor と同じ判定）(2) 前フレームとの横ずれ（位相相関）を出す。
既存スクリプトは import して関数を使うだけ（本体は無変更）。
使い方: python probe_scroll_20261105.py SESSION VIDEO T0 NFRAMES [--save K]
"""
import os, sys, subprocess, re, argparse
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_marks as dm

def decode(path, t0, n, crop):
    cx, cy, cw, ch = crop
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats", "-ss", f"{t0:.3f}", "-copyts", "-i", path,
           "-frames:v", str(n), "-an", "-vf", f"crop={cw}:{ch}:{cx}:{cy},showinfo", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    raw = p.stdout; fs = cw * ch * 3
    frames = [np.frombuffer(raw[i*fs:(i+1)*fs], np.uint8).reshape(ch, cw, 3) for i in range(len(raw)//fs)]
    pts = [float(x) for x in re.findall(rb"pts_time:([0-9.]+)", p.stderr)]
    return frames, pts[:len(frames)]

def shift(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32); gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ga, gb, win)
    return dx, dy, resp

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("session"); ap.add_argument("video"); ap.add_argument("t0", type=float)
    ap.add_argument("n", type=int); ap.add_argument("--save", type=int, default=0); ap.add_argument("--out", default="")
    a = ap.parse_args()
    dm.configure(a.session)
    band = pd.read_csv(dm.BAND_CSV); crop = dm.roi_for(a.video, band)
    fr, pts = decode(os.path.join(dm.VID_DIR, a.video + ".mp4"), a.t0, a.n, crop)
    print("crop", crop, "frames", len(fr))
    prev = None
    for i, (f, t) in enumerate(zip(fr, pts)):
        cur = dm.find_cursor(dm.classify(f))
        s = shift(prev, f) if prev is not None else (0, 0, 0)
        print(f"{i:3d} t={t:.3f} cursor_x4k={None if cur is None else cur + crop[0]} dx={s[0]:+.2f} dy={s[1]:+.2f} r={s[2]:.2f}")
        prev = f
    if a.save and a.out:
        for i in range(0, len(fr), max(1, len(fr)//a.save)):
            cv2.imwrite(os.path.join(a.out, f"probe_{i:03d}.png"), cv2.resize(fr[i], None, fx=0.5, fy=0.5))
