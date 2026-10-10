# -*- coding: utf-8 -*-
"""WAO帯プローブ v2（2026-09-15版・WSL cv2・wao_probe_v2_20260911.py の一般化）。
   v1の粗探索（多段スケール全帯照合）の後、左端400px・右端400pxの部分帯をそれぞれ照合して実スパンから scale を再推定＋ dyL/dyR から傾き。
   入力＝/mnt/g/研究アーカイブ/ビデオ素材/<session>/ の mp4（VID_ で始まる未改名は除外・KV_INCLUDE_VID=1 で含める）。
   済みの動画（CSVに行あり）はスキップ＝再実行で続きから。出力: dataset/band_probe_v2_<session>.csv（最後に WROTE 行）
   使い方（WSL）: cd ~/karaoke-vision && .venv/bin/python /mnt/g/研究アーカイブ/成果物/動画解析/scripts/wao_probe_v2_20260915.py 20260915 [video.mp4 ...]
"""
import sys, os, csv, math, numpy as np, cv2
S = next((a for a in sys.argv[1:] if a.isdigit() and len(a) == 8), os.environ.get("KV_SESSION", "20260915"))
ROOT = "/mnt/g/研究アーカイブ/成果物/動画解析"; VID = f"/mnt/g/研究アーカイブ/ビデオ素材/{S}"
OUTCSV = os.path.join(ROOT, "dataset", f"band_probe_v2_{S}.csv")
REF = cv2.imread(os.path.join(ROOT, "templates/wao_20260911/ref_band_from_0812.png"), cv2.IMREAD_GRAYSCALE); RW, RH = REF.shape[1], REF.shape[0]
PW = 400
def prep(g): return cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F, ksize=3)
def coarse(g):
    H, W = g.shape; y0, y1 = int(H * 0.20), int(H * 0.55); d = 0.5; subs = cv2.resize(g[y0:y1], None, fx=d, fy=d); P = prep(subs); best = None
    for s in np.arange(0.78, 1.08, 0.01):
        tw, th = int(RW * s * d), int(RH * s * d)
        if tw >= subs.shape[1] or th >= subs.shape[0]: continue
        r = cv2.matchTemplate(P, prep(cv2.resize(REF, (tw, th))), cv2.TM_CCOEFF_NORMED); _, mx, _, loc = cv2.minMaxLoc(r)
        if best is None or mx > best[0]: best = (mx, loc[0] / d, loc[1] / d + y0, s)
    return best
def part(g, x, y, s, x0ref, x1ref, mx=60, my=40):
    best = None
    for ds in (-0.03, -0.015, 0.0, 0.015, 0.03):
        ss = s + ds; T = prep(cv2.resize(REF[:, x0ref:x1ref], (int((x1ref - x0ref) * ss), int(RH * ss))))
        px = int(x + x0ref * s); py = int(y); X0 = max(0, px - mx); Y0 = max(0, py - my)
        reg = prep(g[Y0:py + int(RH * ss) + my, X0:px + T.shape[1] + mx])
        if reg.shape[0] < T.shape[0] or reg.shape[1] < T.shape[1]: continue
        r = cv2.matchTemplate(reg, T, cv2.TM_CCOEFF_NORMED); _, m, _, loc = cv2.minMaxLoc(r)
        if best is None or m > best[0]: best = (m, X0 + loc[0], Y0 + loc[1], ss)
    return best
def probe(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); sc, x, y, s = coarse(g)
    L = part(g, x, y, s, 0, PW); R = part(g, x, y, s, RW - PW, RW)
    if not L or not R: return dict(score=sc, x=x, y=y, scale=s, w=RW * s, h=RH * s, angle=0.0, scL=None, scR=None, refined=0)
    xl, yl, sl = L[1], L[2], L[3]; xr, yr, sr = R[1], R[2], R[3]
    span = (xr + PW * sr) - xl; scale = span / RW; ang = math.degrees(math.atan2(yr - yl, (xr + PW * sr / 2) - (xl + PW * sl / 2)))
    return dict(score=sc, x=xl, y=yl, scale=scale, w=RW * scale, h=RH * scale, angle=ang, scL=L[0], scR=R[0], refined=1)
vids = [a for a in sys.argv[1:] if a.lower().endswith(".mp4")]
if not vids:
    vids = [os.path.join(VID, f) for f in sorted(os.listdir(VID)) if f.lower().endswith(".mp4") and (os.environ.get("KV_INCLUDE_VID") == "1" or not f.startswith("VID_"))]
done = set()
if os.path.exists(OUTCSV): done = {r["video"] for r in csv.DictReader(open(OUTCSV, encoding="utf-8"))}
fields = ["video", "xL", "yL", "w", "h", "scale", "angle_deg", "score", "n_frames", "note"]
print(f"PROBE session={S} videos={len(vids)} done={len(done)}", flush=True)
for v in vids:
    name = os.path.splitext(os.path.basename(v))[0]
    if name in done: print("skip", name, flush=True); continue
    cap = cv2.VideoCapture(v); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); res = []
    for frac in (0.25, 0.45, 0.65, 0.85):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac)); ok, img = cap.read()
        if ok: res.append(probe(img))
    cap.release()
    if not res: print("ERR no frame", name, flush=True); continue
    good = [r for r in res if r["refined"] and r["scL"] is not None and r["scL"] > 0.1 and r["scR"] > 0.1] or res
    med = lambda k: float(np.median([r[k] for r in good]))
    row = dict(video=name, xL=round(med("x")), yL=round(med("y")), w=round(med("w")), h=round(med("h")), scale=round(med("scale"), 4), angle_deg=round(med("angle"), 3),
               score=round(med("score"), 3), n_frames=len(good), note=";".join(f"L{(r['scL'] or 0):.2f}/R{(r['scR'] or 0):.2f}/s{r['scale']:.3f}/a{r['angle']:.2f}" for r in res))
    ex = os.path.exists(OUTCSV)
    with open(OUTCSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not ex: w.writeheader()
        w.writerow(row)
    print(row, flush=True)
print("WROTE", OUTCSV, flush=True)
