# -*- coding: utf-8 -*-
"""カウンター帯の位置・スケールを、既存の帯クロップ（bands_wao/<video>.mp4 の先頭フレーム＝2480×120 の正規化画像）を
テンプレートに全画面へ当てて求める（セッション横断・帯ファイルの有無に依らない）。
出力: marks_20260928/band_probe_marks_<session>.csv（video,xL,yL,w,h,scale,score,note）＝ band_probe_v2 と同じ列名。
検証: --validate 20260924 で band_probe_v2_20260924.csv と比較。"""
import os, sys, argparse, subprocess, numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_marks as d
BANDS = r"G:\研究アーカイブ\成果物\動画解析\bands_wao"
Q = 4      # 探索は 1/4 解像度

def first_frame(path, t=0.0, div=1, crop=(0, 0, 3840, 2160)):
    d.DIV = div
    for idx, fr in d.decode_segment(path, t, 1, crop, hwaccel="", threads=4, stall_timeout=120):
        return fr
    return None

def band_template(video):
    p = os.path.join(BANDS, video + ".mp4")
    if not os.path.exists(p): return None
    cap = None
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", "30", "-i", p, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], capture_output=True).stdout
    if len(out) < 2480 * 120 * 3: return None
    return np.frombuffer(out, np.uint8).reshape(120, 2480, 3)

def probe(video, vid_dir, times=(30.0, 60.0, 90.0)):
    tpl = band_template(video)
    if tpl is None: return None, "帯クロップ無し"
    tg = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    best = None
    for t in times:
        fr = first_frame(os.path.join(vid_dir, video + ".mp4"), t, Q)
        if fr is None: continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        for s in np.arange(0.84, 1.06, 0.01):
            w = int(round(2480 * s / Q)); h = max(8, int(round(120 * s / Q)))
            if w >= g.shape[1] or h >= g.shape[0]: continue
            tt = cv2.resize(tg, (w, h), interpolation=cv2.INTER_AREA)
            r = cv2.matchTemplate(g, tt, cv2.TM_CCOEFF_NORMED)
            _, v, _, loc = cv2.minMaxLoc(r)
            if best is None or v > best[0]: best = (v, loc[0] * Q, loc[1] * Q, int(2480 * s), int(120 * s), s, t)
    if best is None: return None, "照合失敗"
    v, x, y, w, h, s, t = best
    return dict(video=video, xL=x, yL=y, w=w, h=h, scale=round(float(s), 4), angle_deg=0.0, score=round(float(v), 3), n_frames=len(times), note=f"tpl=bands_wao t={t}"), ""

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--session", required=True); ap.add_argument("--validate", action="store_true"); ap.add_argument("--videos", nargs="*")
    a = ap.parse_args()
    vid_dir = rf"G:\研究アーカイブ\ビデオ素材\{a.session}"
    g1 = pd.read_csv(d.G1_CSV); vids = a.videos or sorted(g1[g1.session == int(a.session)].video.unique())
    rows = []
    for v in vids:
        r, err = probe(v, vid_dir)
        if r is None: d.log(f"[probe {a.session}] {v}: {err}"); rows.append(dict(video=v, note=err)); continue
        rows.append(r); print(v, r["xL"], r["yL"], r["scale"], r["score"], flush=True)
    out = os.path.join(d.OUT_DIR, f"band_probe_marks_{a.session}.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig"); print("saved", out)
    if a.validate:
        ref = pd.read_csv(os.path.join(os.path.dirname(d.BAND_CSV), f"band_probe_v2_{a.session}.csv"))
        m = pd.DataFrame(rows).merge(ref, on="video", suffixes=("", "_ref"))
        print(m[["video", "xL", "xL_ref", "yL", "yL_ref", "scale", "scale_ref", "score"]].to_string())

if __name__ == "__main__":
    main()
