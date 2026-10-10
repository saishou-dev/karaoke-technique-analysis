# -*- coding: utf-8 -*-
"""セッション展開前の目視確認用画像を作る（最初の 1 曲）。
 1) ROI 重ね描き：しゃくりイベント T−0.6 の全画面（1/2）にカウンター帯（band probe）とノーツ帯 ROI を描く
 2) ノーツ帯ストリップ：T−1.2 / T−0.6 / T+0.0 の ROI を縦に並べる（印・カーソル・演出の見え方）
 3) 印の色相：ROI 内で赤/青/緑ファミリの画素数と、最大の赤成分の位置を印字
出力: evidence/sessions/<session>_<video>_roi.png, _strip.png"""
import os, sys, argparse, numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_marks as d

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--session", required=True); ap.add_argument("--video"); ap.add_argument("--counter", default="shakuri")
    a = ap.parse_args(); d.configure(a.session)
    g1 = pd.read_csv(d.G1_CSV); ev = g1[(g1.session == int(a.session)) & (g1.multi_increment == 0)]
    video = a.video or sorted(ev.video.unique())[0]
    e = ev[(ev.video == video) & (ev.counter == a.counter)].sort_values("video_time_sec")
    if len(e) == 0: e = ev[ev.video == video].sort_values("video_time_sec")
    r = e.iloc[min(2, len(e) - 1)]; T = float(r.video_time_sec)
    band = pd.read_csv(d.BAND_CSV); crop = d.roi_for(video, band); b = band[band.video == video].iloc[0]
    path = os.path.join(d.VID_DIR, video + ".mp4")
    out_dir = os.path.join(d.OUT_DIR, "evidence", "sessions"); os.makedirs(out_dir, exist_ok=True)
    # 1) 全画面 + ROI
    d.DIV = 2
    fr = next(iter(d.decode_segment(path, T - 0.6, 1, (0, 0, 3840, 2160), hwaccel="", threads=4)))[1]
    cv2.rectangle(fr, (int(b.xL / 2), int(b.yL / 2)), (int((b.xL + b.w) / 2), int((b.yL + b.h) / 2)), (0, 255, 0), 2)
    cv2.rectangle(fr, (crop[0] // 2, crop[1] // 2), ((crop[0] + crop[2]) // 2, (crop[1] + crop[3]) // 2), (0, 255, 255), 2)
    cv2.putText(fr, f"{a.session} {video[:30]} {r.counter}#{int(r.count_value)} T-0.6={T-0.6:.2f} band=({int(b.xL)},{int(b.yL)},s{b.scale}) ROI={crop}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    d.imwrite_u(os.path.join(out_dir, f"{a.session}_{video}_roi.png"), fr)
    # 2) ROI ストリップ（等倍 → 0.45 倍）＋ 3) 色ファミリ画素数
    d.DIV = 1
    tiles = []
    for dt in (-1.2, -0.6, 0.0):
        f = next(iter(d.decode_segment(path, T + dt, 1, crop, hwaccel="", threads=4)))[1]
        m = d.classify(f)
        cnt = {k: int(((m & v) > 0).sum()) for k, v in (("red", d.B_RED), ("blue", d.B_BLUE), ("green", d.B_GREEN))}
        t = cv2.resize(f, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"T{dt:+.1f}  red={cnt['red']} blue={cnt['blue']} green={cnt['green']}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(t)
    d.imwrite_u(os.path.join(out_dir, f"{a.session}_{video}_strip.png"), np.vstack(tiles))
    print(a.session, video, r.counter, T, crop, "saved")

if __name__ == "__main__":
    main()
