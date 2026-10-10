# -*- coding: utf-8 -*-
"""DAM Ai 帯プローブ（2026-09-15版・WSL cv2）＝ locate_band.py v2（2026-08-08・0628/0807 で使用）の一般化＋WAO v2 の左右2点照合。
   帯の定義（Ai）＝基準帯画像 1500x180（templates/sample_counterband_t70.png＝金枠スキン／dark_band_ref.png＝暗スキン）を多スケール照合し、
   4K 座標の (xL, yL, w, h) を出す。クロップは wincrop_20260915ai.py が crop→rotate→scale=1500:180 で行う。
   locate_band.py との違い：
     (1) 探索範囲を拡大（scale 0.55〜1.9・y 12〜70%・x 0〜75%）＝9/8以降の固定構図（TV を画面いっぱい）でも帯が収まるように。
     (2) 粗探索の後、左端400px・右端400px の部分帯を照合してスパンから scale を再推定＋傾き（9/11の教訓②「1点照合は最大26pxずれる」）。
     (3) 自己検証＝正規化した帯の数字 ROI（しゃくり・こぶし）でグリフが読めるか（locate_band と同じ規則）→ 読める候補（スキン）を採用。
   入力＝/mnt/g/研究アーカイブ/ビデオ素材/<session>/ の mp4（VID_ で始まる未改名は除外・KV_INCLUDE_VID=1 で含める）。済みはスキップ。
   出力: <dataset>/band_probe_ai_<session>.csv（列＝wincrop 互換：video,xL,yL,w,h,scale,angle_deg,score,n_frames,note）・最後に WROTE 行。
   使い方（WSL）: cd ~/karaoke-vision && .venv/bin/python /mnt/g/研究アーカイブ/成果物/動画解析/scripts/ai_probe_20260915ai.py 20260915 [video.mp4 ...]
   セルフテスト: 環境変数 KV_AI_BASE=<wslパス> で出力先を差し替え（<base>/band_probe_ai_<S>.csv）。
"""
import sys, os, csv, math, numpy as np, cv2
S = next((a for a in sys.argv[1:] if a.isdigit() and len(a) == 8), os.environ.get("KV_SESSION", "20260915"))
ROOT = "/mnt/g/研究アーカイブ/成果物/動画解析"; VID = f"/mnt/g/研究アーカイブ/ビデオ素材/{S}"
BASE = os.environ.get("KV_AI_BASE") or os.path.join(ROOT, "dataset")
OUTCSV = os.path.join(BASE, f"band_probe_ai_{S}.csv")
REFS = [("gold", cv2.imread(os.path.join(ROOT, "templates/sample_counterband_t70.png"), cv2.IMREAD_GRAYSCALE)),
        ("dark", cv2.imread(os.path.join(ROOT, "templates/dark_band_ref.png"), cv2.IMREAD_GRAYSCALE))]
RW, RH = 1500, 180; PW = 400
PROBE_ROIS = [(245, 85, 350, 148), (565, 85, 690, 148)]  # しゃくり・こぶし（read_counters.py v0.8 と同じ）
THRESHOLDS = (215, 200, 185, 228, 170)

def glyph_readable(roi_gray):  # locate_band.py と同一規則
    for T in THRESHOLDS:
        m = (roi_gray > T).astype(np.uint8); n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8); boxes = 0; junk = 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if 28 <= h <= 58 and 4 <= w <= 44 and area >= 60: boxes += 1
            elif area > 30: junk += area
        if 1 <= boxes <= 2 and junk <= 1200: return True
    return False
def prep(g): return cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F, ksize=3)
def coarse(g, REF):
    H, W = g.shape; y0, y1 = int(H * 0.12), int(H * 0.70); x1 = int(W * 0.75); d = 0.5
    subs = cv2.resize(g[y0:y1, 0:x1], None, fx=d, fy=d); P = prep(subs); best = None
    for s in np.arange(0.55, 1.90, 0.025):
        tw, th = int(RW * s * d), int(RH * s * d)
        if tw >= subs.shape[1] or th >= subs.shape[0]: continue
        r = cv2.matchTemplate(P, prep(cv2.resize(REF, (tw, th))), cv2.TM_CCOEFF_NORMED); _, mx, _, loc = cv2.minMaxLoc(r)
        if best is None or mx > best[0]: best = (mx, loc[0] / d, loc[1] / d + y0, s)
    return best
def part(g, REF, x, y, s, x0ref, x1ref, mx=60, my=40):
    best = None
    for ds in (-0.03, -0.015, 0.0, 0.015, 0.03):
        ss = s + ds; T = prep(cv2.resize(REF[:, x0ref:x1ref], (int((x1ref - x0ref) * ss), int(RH * ss))))
        px = int(x + x0ref * s); py = int(y); X0 = max(0, px - mx); Y0 = max(0, py - my)
        reg = prep(g[Y0:py + int(RH * ss) + my, X0:px + T.shape[1] + mx])
        if reg.shape[0] < T.shape[0] or reg.shape[1] < T.shape[1]: continue
        r = cv2.matchTemplate(reg, T, cv2.TM_CCOEFF_NORMED); _, m, _, loc = cv2.minMaxLoc(r)
        if best is None or m > best[0]: best = (m, X0 + loc[0], Y0 + loc[1], ss)
    return best
def readable(g, x, y, w, h):
    H, W = g.shape; x, y = int(max(0, x)), int(max(0, y)); w, h = int(w), int(h)
    if w < 100 or h < 12 or y + h > H or x + w > W: return 0
    band = cv2.resize(g[y:y + h, x:x + w], (RW, RH))
    return sum(1 for (x1, y1, x2, y2) in PROBE_ROIS if glyph_readable(band[y1:y2, x1:x2]))
def probe(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); cands = []
    for name, REF in REFS:
        c = coarse(g, REF)
        if c is None: continue
        sc, x, y, s = c; L = part(g, REF, x, y, s, 0, PW); R = part(g, REF, x, y, s, RW - PW, RW)
        if L and R and L[0] > 0.1 and R[0] > 0.1:
            xl, yl, sl = L[1], L[2], L[3]; xr, yr, sr = R[1], R[2], R[3]
            span = (xr + PW * sr) - xl; scale = span / RW; ang = math.degrees(math.atan2(yr - yl, (xr + PW * sr / 2) - (xl + PW * sl / 2)))
            d = dict(tpl=name, score=sc, x=xl, y=yl, scale=scale, w=RW * scale, h=RH * scale, angle=ang, scL=L[0], scR=R[0], refined=1)
        else:
            d = dict(tpl=name, score=sc, x=x, y=y, scale=s, w=RW * s, h=RH * s, angle=0.0, scL=(L[0] if L else None), scR=(R[0] if R else None), refined=0)
        d["readable"] = readable(g, d["x"], d["y"], d["w"], d["h"]); cands.append(d)
    if not cands: return None
    cands.sort(key=lambda d: (d["readable"], d["refined"], d["score"]), reverse=True)
    return cands[0]
vids = [a for a in sys.argv[1:] if a.lower().endswith(".mp4")]
if not vids:
    vids = [os.path.join(VID, f) for f in sorted(os.listdir(VID)) if f.lower().endswith(".mp4") and (os.environ.get("KV_INCLUDE_VID") == "1" or not f.startswith("VID_"))]
os.makedirs(BASE, exist_ok=True); done = set()
if os.path.exists(OUTCSV): done = {r["video"] for r in csv.DictReader(open(OUTCSV, encoding="utf-8"))}
fields = ["video", "xL", "yL", "w", "h", "scale", "angle_deg", "score", "n_frames", "note"]
print(f"PROBE(ai) session={S} videos={len(vids)} done={len(done)}", flush=True)
for v in vids:
    name = os.path.splitext(os.path.basename(v))[0]
    if name in done: print("skip", name, flush=True); continue
    cap = cv2.VideoCapture(v); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); res = []
    for frac in (0.25, 0.45, 0.65, 0.85):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac)); ok, img = cap.read()
        if ok:
            r = probe(img)
            if r: res.append(r)
    cap.release()
    if not res: print("ERR no frame", name, flush=True); continue
    good = [r for r in res if r["readable"] > 0 and r["refined"]] or [r for r in res if r["readable"] > 0] or res
    med = lambda k: float(np.median([r[k] for r in good]))
    tpl = max(set(r["tpl"] for r in good), key=lambda t: sum(r["tpl"] == t for r in good))
    row = dict(video=name, xL=round(med("x")), yL=round(med("y")), w=round(med("w")), h=round(med("h")), scale=round(med("scale"), 4), angle_deg=round(med("angle"), 3),
               score=round(med("score"), 3), n_frames=len(good),
               note=f"tpl={tpl};" + ";".join(f"{r['tpl'][0]}L{(r['scL'] or 0):.2f}/R{(r['scR'] or 0):.2f}/s{r['scale']:.3f}/a{r['angle']:.2f}/rd{r['readable']}" for r in res))
    ex = os.path.exists(OUTCSV)
    with open(OUTCSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not ex: w.writeheader()
        w.writerow(row)
    print(row, flush=True)
print("WROTE", OUTCSV, flush=True)
