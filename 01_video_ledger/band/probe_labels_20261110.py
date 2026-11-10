# -*- coding: utf-8 -*-
"""帯プローブのラベル投票版（2026-11-10 Opus・新規・既存は無変更）。
   なぜ＝11/10 は曲ごとにカメラの構図が動き、帯が高さの約 0.50〜0.55（4K で y≈1080〜1160）にあり、
         しかも一部の曲で帯の左端（しゃくりのラベル）が画面の外に出ている。
         wao_probe_v2（帯全体のテンプレ照合）は帯が画面に収まる前提なので、KV_PROBE_YR=0.55,0.80 では score 中央値 0.057、
         0.40,0.70 でも 0.157・xL が 36〜1463 にばらつく（実測＝誤検出）。
   方式＝9/11 CHEERS の帯（bands_wao/CHEERS_20260911_105621.mp4・reprobe_band_20261030.py と同じ参照）から6カウンターのラベルを切り出し、
         元動画の 6 フレーム（曲の 20〜80%）で各ラベルを Laplacian-NCC 照合。
         (1) 粗＝半分の解像度・倍率 0.80〜1.02 で、各ラベルの照合図を「帯の原点」へずらして足す投票（画面外のラベルは票が無いだけ）。
         (2) 細＝原寸で各ラベルを予測位置の 横±160px・縦±60px・倍率 −4〜+10% で照合し、ラベル中心の x・y に直線を当てて 倍率・中心・傾きを出す（フレームごと→中央値）。
   出力＝dataset/band_probe_v2_<S>.csv と同じ列（video,xL,yL,w,h,scale,angle_deg,score,n_frames,note）。xL は負になりうる（帯の左端が画面外）。
         負の xL は wincrop_20261110.py が左に黒の余白を足して切り出す。
   使い方: python probe_labels_20261110.py <S> [--out=<csv>] [--workers=4] [video_stem ...]
"""
import os, sys, csv, math, glob, numpy as np, cv2
import concurrent.futures as cf

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
LAB = {"shakuri": (70, 260), "kobushi": (410, 640), "fall": (815, 1030), "vibrato": (1215, 1460), "accent": (1640, 1890), "hammering": (2075, 2340)}
TY0, TY1 = 28, 96; BW, BH = 2480, 120
SCALES = np.round(np.arange(0.80, 1.021, 0.02), 3)
FRACS = (0.20, 0.32, 0.44, 0.56, 0.68, 0.80)
Y0F, Y1F = 0.35, 0.75; PADX = 700
MX = 160  # 細の横の探索幅（11/10 実測＝構図の遠近で右端のラベルが直線の予測から 60〜90px ずれる）


def prep(g):
    return cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F, ksize=3)


def ref_templates():
    cap = cv2.VideoCapture(os.path.join(ROOT, "bands_wao", "CHEERS_20260911_105621.mp4")); refs = []
    for f in (2000, 3500, 6000):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
        if ok: refs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cap.release()
    ref = np.median(np.stack(refs), axis=0).astype(np.uint8)
    return {c: ref[TY0:TY1, a:b] for c, (a, b) in LAB.items()}


def frames(path):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out = []
    for fr in FRACS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * fr)); ok, img = cap.read()
        if ok: out.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cap.release()
    return out


def coarse(gs, T):
    """半分の解像度で投票。返り値＝(倍率, 帯原点 x, 帯原点 y)（原寸・パディング無しの座標）"""
    H, W = gs[0].shape; y0, y1 = int(H * Y0F), int(H * Y1F); d = 0.5
    Ps = [prep(cv2.resize(cv2.copyMakeBorder(g[y0:y1], 0, 0, PADX, PADX, cv2.BORDER_CONSTANT, value=0), None, fx=d, fy=d)) for g in gs]
    best = None
    for s in SCALES:
        acc = None
        for P in Ps:
            for c, (a, b) in LAB.items():
                t = prep(cv2.resize(T[c], (max(8, int((b - a) * s * d)), max(8, int((TY1 - TY0) * s * d)))))
                r = cv2.matchTemplate(P, t, cv2.TM_CCOEFF_NORMED); r = np.nan_to_num(r, nan=0.0)
                r = cv2.dilate(r, np.ones((41, 5), np.uint8))   # 傾き（±2°）と横の誤差を吸収
                ox, oy = int(round(a * s * d)), int(round(TY0 * s * d))  # ラベル左上 → 帯原点へのずらし
                sh = np.full((P.shape[0], P.shape[1]), 0, np.float32)
                hh, ww = r.shape
                # 帯原点 (X,Y) の票＝ラベル位置 (X+ox, Y+oy) の照合値
                xs0, ys0 = max(0, -ox), max(0, -oy)
                src = r[max(0, oy):, max(0, ox):]
                sh[ys0:ys0 + src.shape[0], xs0:xs0 + src.shape[1]] = src[:sh.shape[0] - ys0, :sh.shape[1] - xs0]
                acc = sh if acc is None else acc + sh
        _, mx, _, loc = cv2.minMaxLoc(acc)
        if best is None or mx > best[0]:
            best = (mx, s, loc[0] / d - PADX, loc[1] / d + y0)
    return best


def fine(gs, T, s0, X0, Y0):
    """原寸で各ラベルを予測位置 ±60px で照合し、フレームごとに直線を当てる"""
    per = []; lab_hits = {c: [] for c in LAB}
    for g in gs:
        H, W = g.shape
        gp = cv2.copyMakeBorder(g, 0, 0, PADX, PADX, cv2.BORDER_CONSTANT, value=0)
        pts = []
        for c, (a, b) in LAB.items():
            best = None
            for ds in (-0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10):  # 遠近で右ほどラベルが大きい（11/10 実測＝右端 +5〜8%）
                s = s0 + ds; tw, th = int((b - a) * s), int((TY1 - TY0) * s)
                ex = int(X0 + a * s) + PADX; ey = int(Y0 + TY0 * s)
                xa, ya = max(0, ex - MX), max(0, ey - 60); xb, yb = min(gp.shape[1], ex + tw + MX), min(H, ey + th + 60)
                if xb - xa <= tw or yb - ya <= th: continue
                reg = prep(gp[ya:yb, xa:xb]); t = prep(cv2.resize(T[c], (tw, th)))
                r = np.nan_to_num(cv2.matchTemplate(reg, t, cv2.TM_CCOEFF_NORMED), nan=0.0); _, m, _, loc = cv2.minMaxLoc(r)
                if best is None or m > best[0]: best = (m, xa + loc[0] - PADX + tw / 2, ya + loc[1] + th / 2)
            if best is None: continue
            u = (a + b) / 2.0
            inside = 0 <= best[1] - (b - a) * s0 / 2 and best[1] + (b - a) * s0 / 2 <= W
            if best[0] >= 0.25 and inside:
                pts.append((u, best[1], best[2], best[0], c))
            lab_hits[c].append(round(best[0], 2))
        if len(pts) >= 3:
            u = np.array([p[0] for p in pts]); px = np.array([p[1] for p in pts]); py = np.array([p[2] for p in pts]); wt = np.array([p[3] for p in pts])
            A = np.vstack([u - BW / 2, np.ones_like(u)]).T
            sx, cx = np.linalg.lstsq(A * wt[:, None], px * wt, rcond=None)[0]
            sy, cy = np.linalg.lstsq(A * wt[:, None], py * wt, rcond=None)[0]
            res = float(np.sqrt(np.mean((A @ [sx, cx] - px) ** 2 + (A @ [sy, cy] - py) ** 2)))
            per.append(dict(s=sx, cx=cx, cy=cy - (TY0 + TY1) / 2 * sx + BH / 2 * sx, ang=math.degrees(math.atan2(sy, sx)), n=len(pts),
                            sc=float(np.mean(wt)), res=res, labs="".join(p[4][0] for p in pts)))
    return per, lab_hits


def probe_one(args):
    S, name = args
    T = ref_templates()
    path = os.path.join(r"G:\研究アーカイブ\ビデオ素材", S, name + ".mp4")
    gs = frames(path)
    if not gs: return dict(video=name, note="ERR no frame")
    vote, s0, X0, Y0 = coarse(gs, T)
    per, hits = fine(gs, T, s0, X0, Y0)
    if not per:
        return dict(video=name, xL="", yL="", w="", h="", scale="", angle_deg="", score="", n_frames=0,
                    note=f"NOFIT coarse s={s0} X={X0:.0f} Y={Y0:.0f} vote={vote:.2f} hits={hits}")
    good = [p for p in per if p["res"] < 20] or per  # 遠近のため 6 ラベルの直線の残差は 11〜13px（11/10 実測）
    med = lambda k: float(np.median([p[k] for p in good]))
    s = med("s"); cx = med("cx"); cy = med("cy"); w = BW * s; h = BH * s
    return dict(video=name, xL=round(cx - w / 2), yL=round(cy - h / 2), w=round(w), h=round(h), scale=round(s, 4),
                angle_deg=round(med("ang"), 3), score=round(med("sc"), 3), n_frames=len(good),
                note=f"labels={'/'.join(p['labs'] for p in per)};res={'/'.join(str(round(p['res'],1)) for p in per)};coarse s={s0} X={X0:.0f} Y={Y0:.0f} vote={vote:.2f}")


def main():
    S = sys.argv[1]
    out = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--out=")), os.path.join(ROOT, "dataset", f"band_probe_v2_{S}.csv"))
    nw = int(next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--workers=")), "4"))
    only = [a for a in sys.argv[2:] if not a.startswith("--")]
    vd = os.path.join(r"G:\研究アーカイブ\ビデオ素材", S)
    names = only or sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(vd, f"*_{S}_*.mp4")) if not os.path.basename(f).startswith("VID_"))
    if os.path.exists(out): raise SystemExit(f"既にある＝上書きしない: {out}")
    fields = ["video", "xL", "yL", "w", "h", "scale", "angle_deg", "score", "n_frames", "note"]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=nw) as ex:
        for r in ex.map(probe_one, [(S, n) for n in names]):
            print({k: r.get(k) for k in fields}, flush=True); rows.append(r)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in fields})
    print("WROTE", out, len(rows), flush=True)


if __name__ == "__main__":
    main()
