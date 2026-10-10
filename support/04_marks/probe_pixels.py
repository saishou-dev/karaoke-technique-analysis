# -*- coding: utf-8 -*-
"""探索用：指定時刻のフレームから矩形を切り出し、拡大画像を保存し、HSVヒストグラム（色相の分布）を表示する。
使い方: python probe_pixels.py <video名> <t秒> <x0 y0 x1 y1> <ラベル> [拡大率]
"""
import sys, os, cv2, numpy as np

VID_DIR = r"G:\研究アーカイブ\ビデオ素材\20260924"
EV_DIR = r"G:\研究アーカイブ\成果物\動画解析\marks_20260928\evidence"

def imwrite_u(path, img):
    open(path, "wb").write(cv2.imencode(".png", img)[1].tobytes())

def get_frame(video, t):
    cap = cv2.VideoCapture(os.path.join(VID_DIR, video + ".mp4"))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, (t - 0.3) * 1000))
    fr = None
    while True:
        ok = cap.grab()
        if not ok: break
        tt = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if tt >= t:
            ok, fr = cap.retrieve(); break
    cap.release()
    return fr, tt

def main():
    video = sys.argv[1]; t = float(sys.argv[2])
    x0, y0, x1, y1 = map(int, sys.argv[3:7]); label = sys.argv[7]
    zoom = float(sys.argv[8]) if len(sys.argv) > 8 else 3.0
    fr, tt = get_frame(video, t)
    crop = fr[y0:y1, x0:x1]
    big = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    # 目盛り（50px刻み）
    for gx in range(0, x1 - x0, 50):
        cv2.line(big, (int(gx*zoom), 0), (int(gx*zoom), 12), (255, 255, 255), 1)
        cv2.putText(big, str(x0+gx), (int(gx*zoom)+2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    for gy in range(0, y1 - y0, 50):
        cv2.line(big, (0, int(gy*zoom)), (12, int(gy*zoom)), (255, 255, 255), 1)
        cv2.putText(big, str(y0+gy), (14, int(gy*zoom)+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    out = os.path.join(EV_DIR, f"{label}.png")
    imwrite_u(out, big)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sel = (S > 100) & (V > 100)
    print(f"frame t={tt:.3f}  saved {out}  size={crop.shape}")
    print("彩度>100かつ明度>100 の画素の色相ヒストグラム（10刻み）:")
    hist, _ = np.histogram(H[sel], bins=range(0, 190, 10))
    for b, c in zip(range(0, 180, 10), hist):
        print(f"  H{b:3d}-{b+9:3d}: {c}")

if __name__ == "__main__":
    main()
