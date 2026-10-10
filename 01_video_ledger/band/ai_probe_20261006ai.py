# -*- coding: utf-8 -*-
"""DAM Ai 帯プローブ 2026-10-06 版 v3（セッション内参照帯で照合・18:50 書き直し・既存 *_20260915ai.* は無変更）。
   経緯＝10/6 の Ai 8 曲は固定カメラで帯が 4K の x≈−49〜1549・y≈876〜1068（scale≈1.065・左端「しゃくり」箱の左縁が画面外）。
   汎用テンプレ（sample_counterband_t70／dark_band_ref）の Laplacian-NCC 粗探索は 18:30 版・18:34 版・18:45 版とも誤検出
   （score 0.09〜0.18・座標ばらつき・数字 ROI 不可読）。→ 方式＝**かくれんぼ 45% フレームから目視で確定した帯座標を基準にし、
   その帯画像（画面内の部分 x 0〜1540・y 876〜1068）をセッション内参照として各動画の 4 フレームに照合（scale 0.97〜1.03・y 25〜70%）**、
   dx・dy だけを求める。自己検証＝数字 ROI（しゃくり・こぶし）の可読（9/15ai 版と同じ規則）。出力列・済みスキップ・引数は 9/15ai 版と同じ。
   基準座標（4K）＝REF_XL,REF_YL,REF_W,REF_H（下の定数・templates/ai_20261006/ref_band_kakurenbo_45.png を初回に生成）。
   根拠＝0807 午後の「手動座標 (245,571,1530,208) を通した実績」（RUNBOOK Ai-4 E）の一般化。
"""
import sys, os, csv, json, numpy as np, cv2
S = next((a for a in sys.argv[1:] if a.isdigit() and len(a) == 8), os.environ.get("KV_SESSION", "20261006"))
ROOT = "/mnt/g/研究アーカイブ/成果物/動画解析"; VID = f"/mnt/g/研究アーカイブ/ビデオ素材/{S}"
BASE = os.environ.get("KV_AI_BASE") or os.path.join(ROOT, "dataset")
OUTCSV = os.path.join(BASE, f"band_probe_ai_{S}.csv")
RW, RH = 1500, 180
REF_VIDEO = os.path.join(VID, "かくれんぼ_20261006_101138.mp4"); REF_FRAC = 0.45
REF_XL, REF_YL, REF_W, REF_H = -49, 876, 1598, 192          # 目視確定（18:48・ROI プレビューで数字 4 箱が ROI 内）
REF_PNG = os.path.join(ROOT, "templates", "ai_20261006", "ref_band_kakurenbo_45.png")
REF_X0, REF_X1 = 0, 1540                                      # 参照に使う画面内の範囲（x）
SR = tuple(float(x) for x in os.environ.get("KV_PROBE_SR", "0.97,1.03").split(","))
YR = tuple(float(x) for x in os.environ.get("KV_PROBE_YR", "0.25,0.70").split(","))
PROBE_ROIS = [(245, 85, 350, 148), (565, 85, 690, 148)]
THRESHOLDS = (215, 200, 185, 228, 170)

def glyph_readable(roi_gray):
    for T in THRESHOLDS:
        m = (roi_gray > T).astype(np.uint8); n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8); boxes = 0; junk = 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if 28 <= h <= 58 and 4 <= w <= 44 and area >= 60: boxes += 1
            elif area > 30: junk += area
        if 1 <= boxes <= 2 and junk <= 1200: return True
    return False
def prep(g): return cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F, ksize=3)
def frame_at(path, frac):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac)); ok, img = cap.read(); cap.release()
    return img if ok else None
def ensure_ref():
    if os.path.exists(REF_PNG): return cv2.imread(REF_PNG, cv2.IMREAD_GRAYSCALE)
    img = frame_at(REF_VIDEO, REF_FRAC); g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ref = g[REF_YL:REF_YL + REF_H, REF_X0:REF_X1]; os.makedirs(os.path.dirname(REF_PNG), exist_ok=True); cv2.imwrite(REF_PNG, ref)
    json.dump(dict(video=os.path.basename(REF_VIDEO), frac=REF_FRAC, xL=REF_XL, yL=REF_YL, w=REF_W, h=REF_H, ref_x0=REF_X0, ref_x1=REF_X1),
              open(REF_PNG[:-4] + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return ref
REF = ensure_ref(); PREF = prep(REF)
def readable(g, x, y, w, h):
    H, W = g.shape; pad = max(0, -int(round(x))); gg = cv2.copyMakeBorder(g, 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=0) if pad else g
    x, y, w, h = int(round(x)) + pad, int(round(y)), int(round(w)), int(round(h))
    if y < 0 or y + h > H or x + w > gg.shape[1]: return 0
    band = cv2.resize(gg[y:y + h, x:x + w], (RW, RH))
    return sum(1 for (x1, y1, x2, y2) in PROBE_ROIS if glyph_readable(band[y1:y2, x1:x2]))
def probe(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); H, W = g.shape; y0, y1 = int(H * YR[0]), int(H * YR[1]); P = prep(g[y0:y1]); best = None
    for s in np.arange(SR[0], SR[1] + 1e-9, 0.005):
        T = cv2.resize(PREF, (int(REF.shape[1] * s), int(REF.shape[0] * s)), interpolation=cv2.INTER_AREA) if abs(s - 1) > 1e-6 else PREF
        if T.shape[0] >= P.shape[0] or T.shape[1] >= P.shape[1]: continue
        r = cv2.matchTemplate(P, T, cv2.TM_CCOEFF_NORMED); _, mx, _, loc = cv2.minMaxLoc(r)
        if best is None or mx > best[0]: best = (mx, loc[0], loc[1] + y0, s)
    sc, mx_, my_, s = best
    xL = mx_ - REF_X0 * s + REF_XL * s; yL = my_; w = REF_W * s; h = REF_H * s
    d = dict(tpl="session", score=sc, x=xL, y=yL, scale=s * (REF_W / RW), w=w, h=h, angle=0.0, scL=sc, scR=sc, refined=1)
    d["readable"] = readable(g, xL, yL, w, h); return d
vids = [a for a in sys.argv[1:] if a.lower().endswith(".mp4")]
if not vids:
    vids = [os.path.join(VID, f) for f in sorted(os.listdir(VID)) if f.lower().endswith(".mp4") and (os.environ.get("KV_INCLUDE_VID") == "1" or not f.startswith("VID_"))]
os.makedirs(BASE, exist_ok=True); done = set()
if os.path.exists(OUTCSV): done = {r["video"] for r in csv.DictReader(open(OUTCSV, encoding="utf-8"))}
fields = ["video", "xL", "yL", "w", "h", "scale", "angle_deg", "score", "n_frames", "note"]
print(f"PROBE(ai) v3 session={S} videos={len(vids)} done={len(done)} ref={os.path.basename(REF_PNG)} sr={SR} yr={YR}", flush=True)
for v in vids:
    name = os.path.splitext(os.path.basename(v))[0]
    if name in done: print("skip", name, flush=True); continue
    res = []
    for frac in (0.25, 0.45, 0.65, 0.85):
        img = frame_at(v, frac)
        if img is not None: res.append(probe(img))
    if not res: print("ERR no frame", name, flush=True); continue
    good = [r for r in res if r["readable"] > 0] or res
    med = lambda k: float(np.median([r[k] for r in good]))
    row = dict(video=name, xL=round(med("x")), yL=round(med("y")), w=round(med("w")), h=round(med("h")), scale=round(med("scale"), 4), angle_deg=0.0,
               score=round(med("score"), 3), n_frames=len(good),
               note="tpl=session;" + ";".join(f"sc{r['score']:.2f}/s{r['scale']:.3f}/x{r['x']:.0f}/y{r['y']:.0f}/rd{r['readable']}" for r in res))
    ex = os.path.exists(OUTCSV)
    with open(OUTCSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not ex: w.writeheader()
        w.writerow(row)
    print(row, flush=True)
print("WROTE", OUTCSV, flush=True)
