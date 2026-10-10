# -*- coding: utf-8 -*-
"""探索用：G1イベント時刻 T の前後 [T+dt0, T+dt1] のフレームを指定領域で切り出し、
コンタクトシート（グリッド）を evidence フォルダに保存する。
使い方: python explore_frames.py <video名(拡張子なし)> <T秒> <ラベル> [step_frames] [x0 y0 x1 y1] [scale] [cols] [dt0 dt1]
"""
import sys, os, cv2, numpy as np

VID_DIR = r"G:\研究アーカイブ\ビデオ素材\20260924"
EV_DIR = r"G:\研究アーカイブ\成果物\動画解析\marks_20260928\evidence"

def imwrite_u(path, img):
    # cv2.imwrite は非ASCIIパス不可のため imencode 経由で書く
    open(path, "wb").write(cv2.imencode(".png", img)[1].tobytes())

def main():
    video = sys.argv[1]; T = float(sys.argv[2]); label = sys.argv[3]
    step = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    if len(sys.argv) > 8:
        x0, y0, x1, y1 = map(int, sys.argv[5:9])
    else:
        x0, y0, x1, y1 = 150, 480, 3720, 1010   # ノーツ帯（既定・4K座標）
    scale = float(sys.argv[9]) if len(sys.argv) > 9 else 0.4
    cols = int(sys.argv[10]) if len(sys.argv) > 10 else 1
    dt0 = float(sys.argv[11]) if len(sys.argv) > 11 else -1.7
    dt1 = float(sys.argv[12]) if len(sys.argv) > 12 else 0.5
    path = os.path.join(VID_DIR, video + ".mp4")
    cap = cv2.VideoCapture(path)
    t0, t1 = T + dt0, T + dt1
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t0 * 1000))
    tiles = []
    i = 0
    while True:
        ok = cap.grab()
        if not ok: break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t < t0: continue
        if t > t1: break
        if i % step == 0:
            ok, fr = cap.retrieve()
            crop = fr[y0:y1, x0:x1]
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.putText(crop, f"t={t:.3f} (T{t-T:+.3f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(crop)
        i += 1
    cap.release()
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for k, tl in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r*h:(r+1)*h, c*w:(c+1)*w] = tl
    out = os.path.join(EV_DIR, f"{label}.png")
    imwrite_u(out, sheet)
    print(out, sheet.shape, len(tiles))

if __name__ == "__main__":
    main()
