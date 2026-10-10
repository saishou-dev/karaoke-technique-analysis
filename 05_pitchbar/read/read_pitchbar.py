# -*- coding: utf-8 -*-
"""E4 音程バー読取 v0（2026-10-08）＝録画（DAM 精密採点・WAO 画面）から「お手本の音程バー」を読み、時刻付きの基準メロディ列（10ms）を出す。
   設計メモ：../設計メモ_音程バー読取_v0.md。流用＝帯プローブ（dataset/band_probe_v2_<S>.csv の xL,yL,w,angle）で画面位置を合わせ、
   元動画の pts 表（dataset/pts/<video>.csv）でフレーム番号→時刻、同期表（dataset/sync/sync_offsets_<S>.csv）で録音時刻へ。

   段階1 read : ffmpeg で音程バー領域（帯の上・幅=帯幅・高さ=0.21×帯幅）を切り出し（横 1/2・縦は原寸）→ 毎フレーム
                ①カーソル（白い縦線）の x ②バー（金の塗り／白い中抜き枠）の列ごとの中心 y を取り、ページ（カーソルが左端へ戻る区切り）ごとに
                列 x × 中心 y の投票表に積む。ページ内はカーソル x が時刻に線形 → 列 x ↔ 時刻。 → out/<video>_pages.npz
   段階2 eval : 譜線（横の薄い線・1本=1半音の仮説）で y→半音に換算し、pYIN（dataset/pitch_pyin_<S>/<wav>_f0.csv）との一致率が最大になる
                「半音オフセット・表示遅れ dt」を探索して確定（1曲1組）。 → out/<video>_ref.csv・out/<video>_eval.json・out/<video>_overlay.png
   使い方: python read_pitchbar.py <session> <video_name> [--stage read|eval|all] [--debug-frames 60,120]
"""
import os, sys, csv, json, math, time, argparse, subprocess
import numpy as np, cv2

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
VIDEO_ROOT = r"G:\研究アーカイブ\ビデオ素材"
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(os.path.dirname(HERE), "out")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from kv_config_20260915 import FFMPEG, Cfg  # noqa: E402

# ---- 画面上の寸法（4K 原寸・9/24 lulu. で実測）----
ROI_W_RATIO = 1.50       # 音程バー領域の幅＝1.50×帯幅（lulu. で帯 2334px・画面右端まで ≈3530px・譜面はその手前まで）
FRAME_W = 3840
ROI_TOP_RATIO = 0.21     # 音程バー領域の上端＝帯上端 − 0.21×帯幅（lulu. で 490px・譜線12本が 571〜1004 に収まる）
ROI_BOT_MARGIN = 5       # 下端＝帯上端 − 5px
GOLD_INNER = (40, 56)    # 塗りバーの内側（淡い白）の高さ（原寸 px・実測 47〜49）
HOLLOW_INNER = (41, 50)  # 中抜きバーの内側（黒）の高さ（原寸 px・実測 44〜46・譜線の隙間 35〜37 は除外）
MIN_SEG_W = 16           # バー1本の最小幅（横1/2 解像度の列数・原寸 32px）。丸（歌った音程）は中心一定の幅が ≤10 列
N_LINES = 12             # 譜線の本数（lulu. 実測）
LINE_SPACING_RATIO = 0.01688  # 譜線間隔／帯幅（lulu. 39.4px/2334px）
CURSOR_FRAC = 0.50       # 列の「左右より明るい行」の率がこれ以上ならカーソル
PAGE_JUMP_FRAC = 0.30    # カーソル x がこれ×幅だけ左へ戻ったら新ページ


def imwrite_u(path, img):
    """非ASCIIパス対応の cv2.imwrite（cv2.imwrite は日本語パスで黙って失敗する）"""
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok: open(path, "wb").write(buf.tobytes())
    return ok


def probe_row(S, name):
    p = os.path.join(ROOT, "dataset", f"band_probe_v2_{S}.csv")
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        if r["video"] == name: return r
    raise SystemExit(f"プローブ行が無い: {name} in {p}")


def roi_geom(r):
    xL, yL, w = int(r["xL"]), int(r["yL"]), int(r["w"]); ang = float(r["angle_deg"])
    top = yL - int(round(ROI_TOP_RATIO * w)); bot = yL - ROI_BOT_MARGIN
    w_roi = min(int(round(ROI_W_RATIO * w)), FRAME_W - xL - 20)  # 譜面は帯より右へ長い（lulu. で帯 2334px に対し譜面 ≈3640px）
    return dict(x=xL, y=top, w=w_roi, h=bot - top, angle=ang, band_w=w)


def open_ffmpeg(src, g, every=1):
    mx, my = 20, 48; W2 = g["w"] // 2; H = g["h"]
    vf = (f"crop={g['w']+2*mx}:{g['h']+2*my}:{g['x']-mx}:{g['y']-my},rotate={-math.radians(g['angle']):.6f}:c=black,"
          f"crop={g['w']}:{g['h']}:{mx}:{my},scale={W2}:{H}:flags=area")
    cmd = [FFMPEG, "-v", "error", "-threads", "6", "-i", src, "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 7), W2, H


def _pair_runs(M, B, smin, smax):
    """列ごとに「M の連続ラン（長さ smin..smax）で、上下 4px 以内に B（縁）がある」ものの中心 y を返す（無い列は nan）。
       M=バーの内側（塗り：淡い白／中抜き：黒）、B=バーの縁（塗り：金／中抜き：白線）。譜線の隙間（35〜37px）は長さの下限で落ちる。
       実装＝ランの始点と終点を列優先で並べると k 番目同士が対になる（ランは重ならない）ので、ループ無しで全ランの長さが出る。"""
    H, W = M.shape
    Mt = M.T                                                     # 列優先（x, y）で where すると同じ列のランが y 順に並ぶ
    start = Mt & ~np.roll(Mt, 1, axis=1); start[:, 0] = Mt[:, 0]
    end = Mt & ~np.roll(Mt, -1, axis=1); end[:, -1] = Mt[:, -1]
    xs, ys = np.nonzero(start); xe, ye = np.nonzero(end)
    center = np.full(W, np.nan)
    if len(xs) == 0 or len(xs) != len(xe): return center
    ln = ye - ys + 1
    ok = (ln >= smin) & (ln <= smax) & (ys >= 4) & (ye <= H - 5)
    xs, ys, ye = xs[ok], ys[ok], ye[ok]
    if len(xs) == 0: return center
    Bp = np.zeros((H + 8, W), bool); Bp[4:-4] = B
    above = Bp[ys, xs] | Bp[ys + 1, xs] | Bp[ys + 2, xs] | Bp[ys + 3, xs]            # B のどれか in ys-4..ys-1
    below = Bp[ye + 5, xs] | Bp[ye + 6, xs] | Bp[ye + 7, xs] | Bp[ye + 8, xs]        # B のどれか in ye+1..ye+4
    good = above & below
    xs, cen = xs[good], (ys[good] + ye[good]) / 2.0
    # 列に複数あれば最初（上）のもの
    first = np.ones(len(xs), bool); first[1:] = xs[1:] != xs[:-1]
    center[xs[first]] = cen[first]
    return center


def analyze_frame(img):
    """1フレーム → (xc or None, center[x] (nan=無し), rowsum_lines[3,H])
    バー＝「内側＋縁」で読む（9/24 lulu. で実測）：
      塗りバー＝金の縁（H10〜40・S≥140・V≥80：暗い金 V110〜128 も含む）の内側に淡い白（S≤110・V≥90）が 40〜56px
      中抜きバー＝白い枠線（V≥150・S≤100）の内側が黒（V<70）で 41〜50px（譜線の隙間 35〜37px は除外）
    どちらも背景の明るさに依らない（塗りの内側／中抜きの黒は不透明）。歌った音程の赤丸は除外。マスクは cv2.inRange（numpy 比で 1/10 の時間）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); V = hsv[:, :, 2]
    H, W = V.shape
    red = cv2.inRange(hsv, (0, 120, 80), (8, 255, 255)) | cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
    G = cv2.inRange(hsv, (10, 140, 80), (40, 255, 255)) & ~red                     # 金の縁（光彩込み）
    Wt = cv2.inRange(hsv, (0, 0, 150), (180, 100, 255))                              # 白い枠線・カーソル
    P = cv2.inRange(hsv, (0, 0, 90), (180, 110, 255)) & ~G & ~red                    # 塗りバーの内側（淡い白）
    D = cv2.inRange(hsv, (0, 0, 0), (180, 255, 69))                                   # 中抜きバーの内側（黒）
    # カーソル＝明るい縦線（暗い背景）／演奏済み側が白く明るい段差（明るい背景）：右 +6 列より 15 以上明るく、左 −6 列より 20 以上暗くない行を数える
    Vi = V.astype(np.int16); dr = Vi[:, 6:-6] - Vi[:, 12:]; dl = Vi[:, 6:-6] - Vi[:, :-12]
    cw = np.zeros(W, np.float32); cw[6:-6] = ((dr >= 15) & (dl >= -20)).sum(axis=0); cw[:12] = 0; cw[-8:] = 0
    dark = np.where(V.mean(axis=0) < 10)[0]                                           # 画面外（ベゼル）の列：その左 12 列は画面端の段差なので除外
    for x in dark[:1]: cw[max(0, x - 12): x + 1] = 0
    xc = None
    if cw.max() >= CURSOR_FRAC * H:
        xc = float(np.argmax(cw)); x0, x1 = max(0, int(xc) - 7), min(W, int(xc) + 8)
        for a in (G, Wt, P, D): a[:, x0:x1] = 0
    lm = cv2.inRange(hsv, (0, 0, 115), (180, 90, 184))                                # 譜線の行プロファイル（全フレーム中央値）：全体・左1/4・右1/4（傾き補正用）
    lines = np.stack([lm.sum(axis=1), lm[:, : W // 4].sum(axis=1), lm[:, -(W // 4):].sum(axis=1)]).astype(np.float32) / 255.0
    c_gold = _pair_runs(P > 0, G > 0, GOLD_INNER[0], GOLD_INNER[1])
    c_hollow = _pair_runs(D > 0, Wt > 0, HOLLOW_INNER[0], HOLLOW_INNER[1])
    center = np.where(np.isnan(c_gold), c_hollow, c_gold)
    # 横の連続性：同じ中心（±3）が MIN_SEG_W 列以上続く区間だけ採用（丸・技法マーク・効果を落とす）
    valid = ~np.isnan(center)
    brk = np.ones(W, bool); brk[1:] = (~valid[1:]) | (~valid[:-1]) | (np.abs(np.nan_to_num(center[1:]) - np.nan_to_num(center[:-1])) > 3)
    seg = np.cumsum(brk); lens = np.bincount(seg); keep = valid & (lens[seg] >= MIN_SEG_W)
    center = np.where(keep, center, np.nan)
    return xc, center, lines


def stage_read(S, name, every=1, debug_frames=(), max_frames=None):
    """段階1：全フレームを読み、フレームごとの「バー区間（x0,x1,中心y）」とカーソル x を保存する（ページ分割・時刻付けは段階2＝再実行可能）。"""
    C = Cfg(S); r = probe_row(S, name); g = roi_geom(r)
    src = os.path.join(VIDEO_ROOT, S, name + ".mp4"); pts = C.ensure_src_pts(name)
    proc, W2, H = open_ffmpeg(src, g); nbytes = W2 * H * 3
    print(f"READ {name}: roi={g} W2={W2} H={H} pts={len(pts)}", flush=True)
    seg_rows = []   # (frame, x0, x1, y*10)
    fr_rows = []    # (frame, t, xc(-1=無し))
    line_profs = []; i = -1; t0 = time.time()
    dbg_dir = os.path.join(OUT, "debug"); os.makedirs(dbg_dir, exist_ok=True)
    while True:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes: break
        i += 1
        if i % every: continue
        if i >= len(pts) or (max_frames and i >= max_frames): break
        img = np.frombuffer(buf, np.uint8).reshape(H, W2, 3); t = pts[i]
        xc, center, lines = analyze_frame(img)
        if i % 10 == 0: line_profs.append(lines)
        fr_rows.append((i, t, -1.0 if xc is None else xc))
        valid = ~np.isnan(center)
        if valid.any():
            brk = np.ones(W2, bool); brk[1:] = (~valid[1:]) | (~valid[:-1]) | (np.abs(np.nan_to_num(center[1:]) - np.nan_to_num(center[:-1])) > 3)
            ids = np.cumsum(brk); xs = np.where(valid)[0]
            for sid in np.unique(ids[xs]):
                cols = xs[ids[xs] == sid]
                seg_rows.append((i, int(cols[0]), int(cols[-1]), int(round(float(np.median(center[cols])) * 10))))
        if int(round(t)) in debug_frames and abs(t - round(t)) < 0.01:
            d = img.copy()
            for x in np.where(valid)[0]: cv2.circle(d, (int(x), int(center[x])), 1, (0, 255, 0), -1)
            if xc is not None: cv2.line(d, (int(xc), 0), (int(xc), H - 1), (255, 0, 255), 1)
            imwrite_u(os.path.join(dbg_dir, f"{name}_t{int(round(t))}.png"), d)
        if i % 2000 == 0: print(f"  frame {i}/{len(pts)} t={t:.1f}s segs={len(seg_rows)} {time.time()-t0:.0f}s", flush=True)
    proc.stdout.close(); proc.wait()
    n_frames = i + 1; line_acc = np.median(np.stack(line_profs), axis=0) if line_profs else np.zeros((3, H))
    print(f"  decoded={n_frames} pts_rows={len(pts)} segs={len(seg_rows)} cursor_found={sum(1 for r in fr_rows if r[2] >= 0)}/{len(fr_rows)} {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(os.path.join(OUT, f"{name}_frames.npz"), segs=np.array(seg_rows, np.int32).reshape(-1, 4), frames=np.array(fr_rows, np.float64).reshape(-1, 3),
                        line_acc=line_acc, geom=json.dumps(dict(g, W2=W2, H=H, n_frames=n_frames, every=every)))


def find_lines(prof, H, band_w):
    """譜線の行：行プロファイル（全フレーム中央値）から基線（移動中央値 25 行）を引き、
       事前値 d0=0.01688×帯幅（lulu. 実測 39.4px/2334px）の ±8% で間隔 d と位相 φ の櫛（12 本）を当てて合計が最大の組を返す。"""
    from scipy.ndimage import median_filter
    det = prof - median_filter(prof, size=25, mode="nearest"); det = np.maximum(det, 0)
    d0 = LINE_SPACING_RATIO * band_w; best = None
    for d in np.arange(d0 * 0.92, d0 * 1.08, 0.1):
        for phi in np.arange(0, d, 0.5):
            rows = phi + d * np.arange(N_LINES)
            if rows[-1] >= H - 1: continue
            sc = float(np.sum(np.interp(rows, np.arange(H), det)))
            if best is None or sc > best[0]: best = (sc, d, phi)
    if best is None: return [], None
    _, d, phi = best
    rows = [float(phi + d * k) for k in range(N_LINES)]
    rows = [float(int(r) - min(3, int(r)) + np.argmax(det[max(0, int(r) - 3): int(r) + 4])) for r in rows]  # ±3 行で最大値へ微調整
    return rows, float(np.median(np.diff(rows)))


def _ransac_line(t, x, tol=6.0, iters=300, seed=0):
    """(t,x) に直線を RANSAC で当てる → (v, b, inlier_mask)。傾きは 150〜450 px/s・内点の時間幅 1 秒以上に限定（静止した縦エッジの誤検出＝傾き 0 を除く）。"""
    rng = np.random.default_rng(seed); n = len(t); best = (0, None, None, None)
    if n < 10: return None
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        if i == j or t[j] == t[i]: continue
        v = (x[j] - x[i]) / (t[j] - t[i])
        if not (150 <= v <= 450): continue
        b = x[i] - v * t[i]; inl = np.abs(x - (v * t + b)) <= tol
        if inl.sum() > best[0]: best = (int(inl.sum()), v, b, inl)
    if best[1] is None or best[0] < 10: return None
    inl = best[3]; v, b = np.polyfit(t[inl], x[inl], 1); inl = np.abs(x - (v * t + b)) <= tol
    if inl.sum() < 10 or not (150 <= v <= 450) or np.ptp(t[inl]) < 1.0: return None
    v, b = np.polyfit(t[inl], x[inl], 1)
    return float(v), float(b), inl


def segment_and_time(frames, W2):
    """ページ分割と時刻付けをカーソル x(t) から行う。lulu. 100 秒の試験で判明：**カーソル速度はページごとに違う**（216〜383px/s＝ページの長さ＝楽譜の小節数が違う）。
       1) 静止した候補（約 0.5 秒で x の幅 ≤4px）を捨てる（背景の縦エッジ・前奏画面）。
       2) 「ランプ」（カーソルが左→右へ等速で動く 1 区間）を先頭から貪欲に抽出：次の 400 検出（≈7 秒）の窓で RANSAC 直線（v 150〜450・内点 ±6px・10 点以上・1 秒以上）。
          ランプ 1 本＝1 ページ。t0=−b/v。
       3) ページ境界＝次のランプの t0（カーソルが x=0 を通る時刻）。前のページの予測終了（t0+x_end/v）と次の t0 の隙間が 1 秒以上なら、
          その間はカーソルが読めなかったページ＝隙間を典型ページ長で割った枚数に等分し、各々 v=x_end/長さ で補間（timing=interp／interp_multi）。
          最初のランプの前（前奏＝最初のページのバーが静止表示）は最初のページに含める。最後のランプの後は v の中央値で代用（tail_fallback）。
       返り値：page[frame_row], fits(list), v_med, x_end"""
    n = len(frames); T = frames[:, 1]
    det = np.where(frames[:, 2] >= 0)[0]
    if len(det) < 50: raise SystemExit("カーソルがほとんど読めない")
    t = frames[det, 1]; x = frames[det, 2]
    static = np.zeros(len(x), bool)
    for k in range(0, len(x) - 30):
        if t[k + 29] - t[k] < 1.5 and np.ptp(x[k: k + 30]) <= 4: static[k: k + 30] = True
    det = det[~static]; t = t[~static]; x = x[~static]
    if len(det) < 50: raise SystemExit("カーソルがほとんど読めない（静止候補を除いた後）")
    ramps = []; k = 0; N = len(t)
    while k < N - 10:
        w1 = min(N, k + 400); r = _ransac_line(t[k:w1], x[k:w1], seed=k)
        if r is None: k += 1; continue
        v, b, inl = r; idx = np.where(inl)[0] + k
        if t[idx[0]] - t[k] > 0.3: k += 1; continue           # 直線が今の点から始まらない（今の点はノイズ）
        # 内点の間に大きな隙間（>1 秒）があれば最初の塊だけをランプにする
        gaps = np.where(np.diff(t[idx]) > 1.0)[0]
        if len(gaps): idx = idx[: gaps[0] + 1]
        if len(idx) < 10 or np.ptp(t[idx]) < 1.0: k += 1; continue
        v, b = np.polyfit(t[idx], x[idx], 1); res1 = float(np.abs(x[idx] - (v * t[idx] + b)).max())
        pieces = [idx]
        if res1 > 12 and len(idx) >= 60:
            # 1 ページ内でカーソル速度が変わる（テンポ変化：lulu. 74〜85 秒で 154→300px/s）→ 折れ点で 2 本に分ける（2 本の最大残差 ≤8px になるときだけ）
            best_k = None
            for kk in range(30, len(idx) - 30, 10):
                a1 = np.polyfit(t[idx[:kk]], x[idx[:kk]], 1); a2 = np.polyfit(t[idx[kk:]], x[idx[kk:]], 1)
                r = max(np.abs(x[idx[:kk]] - np.polyval(a1, t[idx[:kk]])).max(), np.abs(x[idx[kk:]] - np.polyval(a2, t[idx[kk:]])).max())
                if best_k is None or r < best_k[0]: best_k = (r, kk)
            if best_k is not None and best_k[0] <= 8: pieces = [idx[: best_k[1]], idx[best_k[1]:]]
        for pc in pieces:
            v, b = np.polyfit(t[pc], x[pc], 1)
            ramps.append(dict(k0=int(pc[0]), k1=int(pc[-1]), t_a=float(t[pc[0]]), t_b=float(t[pc[-1]]), v=float(v), b=float(b), n_inlier=int(len(pc)),
                              resid_max=float(np.abs(x[pc] - (v * t[pc] + b)).max()), split=len(pieces) > 1))
        k = int(idx[-1]) + 1
    if not ramps: raise SystemExit("カーソルのランプが 1 本も見つからない")
    # ページ区間の組み立て（恋風で判明：カーソルが読めない区間＝明るい背景のページが 1/4 ある → 前後のランプから補間する）
    # x_end＝ページ切替の瞬間に前ページのカーソルがある x（隣接ランプ対から v_i·(t0_{i+1}−t0_i) の中央値）。
    for r in ramps: r["t0"] = -r["b"] / r["v"]
    xe = [ramps[i]["v"] * (ramps[i + 1]["t0"] - ramps[i]["t0"]) for i in range(len(ramps) - 1) if ramps[i + 1]["t0"] - ramps[i]["t_b"] < 1.5]
    x_end = float(np.median(xe)) if len(xe) >= 2 and 0.8 * W2 <= np.median(xe) <= 1.1 * W2 else 0.97 * W2
    v_med = float(np.median([r["v"] for r in ramps])); D_med = x_end / v_med
    pages = []  # dict(t_start, t_end, v, t0, timing, ramp)
    for i, r in enumerate(ramps):
        t_end = r["t0"] + x_end / r["v"]
        if i == 0: start = 0.0
        if i + 1 < len(ramps):
            nxt = ramps[i + 1]; gap = nxt["t0"] - t_end
            if gap < 0.5 * D_med: end = nxt["t0"]; fill = []     # 小さな隙間＝このページがやや長い（幅が違う）だけ → 次の t0 まで延ばす
            else:
                end = t_end; n_fill = max(1, int(round(gap / D_med))); dur = gap / n_fill
                fill = [dict(t_start=t_end + k * dur, t_end=t_end + (k + 1) * dur, v=x_end / dur, t0=t_end + k * dur, timing=("interp" if n_fill == 1 else "interp_multi"), ramp=None) for k in range(n_fill)]
        else:
            end = max(t_end, r["t_b"]) ; fill = []
            if T[-1] > end + 0.5: fill = [dict(t_start=end, t_end=T[-1] + 0.01, v=v_med, t0=end, timing="tail_fallback", ramp=None)]
        pages.append(dict(t_start=start, t_end=end, v=r["v"], t0=r["t0"], timing="cursor", ramp=r)); pages.extend(fill)
        start = fill[-1]["t_end"] if fill else end
    page = np.full(n, -1, int); fits = []
    for pi, pgd in enumerate(pages):
        sel = (T >= pgd["t_start"]) & (T < pgd["t_end"]); page[sel] = pi
        f = dict(page=pi, n=int(sel.sum()), n_xc=int((sel & (frames[:, 2] >= 0)).sum()), t_first=float(pgd["t_start"]), t_last=float(pgd["t_end"]), v=float(pgd["v"]), t0=float(pgd["t0"]),
                 n_inlier=(pgd["ramp"]["n_inlier"] if pgd["ramp"] else 0), resid_max=(pgd["ramp"]["resid_max"] if pgd["ramp"] else None), ok=True, timing=pgd["timing"])
        fits.append(f)
    page[page < 0] = 0
    return page, fits, v_med, x_end


def load_pyin(S, name):
    C = Cfg(S); off = None; wav = None
    for r in csv.DictReader(open(C.sync_csv, encoding="utf-8-sig")):
        if r["video"] == name: off = float(r["offset_sec"]); wav = r["wav"]; break
    if off is None: raise SystemExit(f"同期表に無い: {name}")
    p = os.path.join(ROOT, "dataset", f"pitch_pyin_{S}", os.path.splitext(wav)[0] + "_f0.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    t = np.array([float(r["audio_time_sec"]) for r in rows]); f0 = np.array([float(r["f0_hz"]) for r in rows])
    conf = np.array([float(r["conf"]) for r in rows]); voiced = np.array([int(r["voiced"]) for r in rows]) > 0
    midi = np.where(f0 > 40, 69 + 12 * np.log2(np.maximum(f0, 1) / 440.0), np.nan)
    return dict(t=t, midi=midi, conf=conf, voiced=voiced, offset=off, wav=wav)


def stage_eval(S, name, dt_range=(-0.6, 0.6), frames_file=None):
    z = np.load(frames_file or os.path.join(OUT, f"{name}_frames.npz")); segs = z["segs"]; frames = z["frames"]; line_acc = z["line_acc"]; g = json.loads(str(z["geom"]))
    W2, H = g["W2"], g["H"]; rows, sp = find_lines(line_acc[0], H, g["band_w"])
    if sp is None: raise SystemExit(f"譜線が見つからない {name}: rows={rows}")
    rows_l, sp_l = find_lines(line_acc[1], H, g["band_w"]); rows_r, sp_r = find_lines(line_acc[2], H, g["band_w"])
    tilt = float(np.median(np.array(rows_r) - np.array(rows_l))) / (W2 - W2 // 4) if (sp_l and sp_r) else 0.0
    y_top = rows[0]
    page, fits, v_global, x_start = segment_and_time(frames, W2)
    acc = np.zeros((len(fits), W2, H // 2 + 1), np.int16); fidx = {int(f): k for k, f in enumerate(frames[:, 0])}
    for k in range(len(segs)):
        fr, a, b, y = segs[k]; p = page[fidx[int(fr)]]; yb = min(H // 2, max(0, int(round(y / 20.0))))
        acc[p, a: b + 1, yb] += 1
    print(f"  pages={len(fits)} v_med={v_global:.1f}px/s x_end={x_start:.0f} pages=" +
          ",".join(f"{f['page']}:{f['n']}f/{f['n_xc']}xc/{(f['timing'] or '-')[:4]}/v{0 if f['v'] is None else f['v']:.0f}" for f in fits), flush=True)
    # 10ms 格子（動画 pts 時刻）
    t_end = max(f["t_last"] for f in fits if f["t_last"] is not None); T = np.arange(0, t_end + 0.01, 0.01)
    ref_y = np.full(len(T), np.nan); conf = np.zeros(len(T)); nv = np.zeros(len(T), int); pg = np.full(len(T), -1, int)
    for f in fits:
        if not f["ok"]: continue
        a = acc[fits.index(f)]; n = max(1, f["n"])
        xs = np.arange(W2); tx = f["t0"] + xs / f["v"]
        ybin = np.argmax(a, axis=1); votes = a[xs, ybin]
        ycorr = ybin * 2.0 - tilt * (xs - W2 / 2.0)   # 傾き補正（譜線行は全体プロファイル＝中央相当）
        valid = (votes >= max(3, 0.05 * n)) & (tx >= f["t_first"] - 0.2) & (tx <= f["t_last"] + 0.2)
        idx = np.round(tx / 0.01).astype(int); ok = valid & (idx >= 0) & (idx < len(T))
        # 同じ時刻格子に複数列が落ちる場合は投票の多い方
        for x in xs[ok]:
            k = idx[x]
            if votes[x] > nv[k]: nv[k] = votes[x]; ref_y[k] = ycorr[x]; conf[k] = votes[x] / n; pg[k] = f["page"]
    # 格子の穴（隣接列が別格子に落ちた 1 コマ分）を前方補完
    for k in range(1, len(T) - 1):
        if np.isnan(ref_y[k]) and not np.isnan(ref_y[k - 1]) and not np.isnan(ref_y[k + 1]) and ref_y[k - 1] == ref_y[k + 1]:
            ref_y[k] = ref_y[k - 1]; conf[k] = min(conf[k - 1], conf[k + 1]); nv[k] = min(nv[k - 1], nv[k + 1]); pg[k] = pg[k - 1]
    # 譜線単位の位置（上の線=0・下へ正）
    line_pos = (ref_y - y_top) / sp
    # pYIN
    P = load_pyin(S, name); Pt = P["t"] - P["offset"]  # 動画時刻へ
    # (k, dt) の探索＝ページごとに最良の整数オフセット m0_p を許した一致数の合計が最大（DAM はページごとに表示音域を動かすため、1 曲 1 オフセットでは決まらない）
    have = ~np.isnan(line_pos); best = None
    pv_base = (P["voiced"] & (P["conf"] >= 0.5) & ~np.isnan(P["midi"])).astype(float); pm_base = np.nan_to_num(P["midi"], nan=-100.0)
    page_ids = [f["page"] for f in fits if f["ok"]]
    for k_semi in (1, 2):  # 1本の譜線=1半音 or 2半音
        for dt_i in range(int(dt_range[0] * 100), int(dt_range[1] * 100) + 1, 2):
            dt = dt_i / 100.0
            pm_s = np.interp(T + dt, Pt, pm_base); pv_s = np.interp(T + dt, Pt, pv_base) > 0.5
            both = have & pv_s
            if both.sum() < 100: continue
            d_all = pm_s + k_semi * line_pos; tot = 0; m0s = []
            for p_ in page_ids:
                selp = both & (pg == p_)
                if selp.sum() < 30: continue
                d = d_all[selp]; cnts = [int(np.sum(np.abs(d - c) <= 0.5)) for c in range(40, 100)]
                tot += max(cnts); m0s.append(40 + int(np.argmax(cnts)))
            if best is None or tot > best["tot"]: best = dict(tot=tot, k_semi=k_semi, dt=dt, m0=(int(np.median(m0s)) if m0s else 60), n=int(both.sum()), agree=tot / max(1, both.sum()))
    k_semi, dt, m0 = best["k_semi"], best["dt"], best["m0"]
    pm_s = np.interp(T + dt, Pt, np.nan_to_num(P["midi"], nan=-100.0)); pv_s = np.interp(T + dt, Pt, (P["voiced"] & (P["conf"] >= 0.5) & ~np.isnan(P["midi"])).astype(float)) > 0.5
    both = have & pv_s
    # 曲全体で 1 つのオフセット（m0）での一致率＝参考値
    ref_global = np.round(m0 - k_semi * line_pos); agree50_global = float(np.mean(np.abs(pm_s[both] - ref_global[both]) <= 0.5))
    # ページごとのオフセット：DAM はページ（楽節）ごとに表示する音域を上下に動かす（lulu. 100 秒試験：前半は一致・サビで 5〜8 半音ずれた）。
    # 画面に絶対音高の手がかりが無いので、ページごとに整数半音のオフセット m0_p を pYIN との一致率最大で決める（比較が 50 コマ以上あるページだけ・無ければ曲全体の m0）。
    m0_page = {}; ref_midi_raw = m0 - k_semi * line_pos
    for f in fits:
        if not f["ok"]: continue
        selp = both & (pg == f["page"])
        if selp.sum() < 50: m0_page[f["page"]] = dict(m0=m0, n=int(selp.sum()), agree=None); continue
        d = pm_s[selp] + k_semi * line_pos[selp]; cands = range(m0 - 14, m0 + 15)
        sc = [float(np.mean(np.abs(d - c) <= 0.5)) for c in cands]; bi = int(np.argmax(sc)); m0p = m0 - 14 + bi
        m0_page[f["page"]] = dict(m0=int(m0p), n=int(selp.sum()), agree=round(sc[bi], 3))
        ref_midi_raw[pg == f["page"]] = m0p - k_semi * line_pos[pg == f["page"]]
    ref_midi = np.round(ref_midi_raw); diff = pm_s - ref_midi
    agree50 = float(np.mean(np.abs(diff[both]) <= 0.5)); agree100 = float(np.mean(np.abs(diff[both]) <= 1.0))
    agree_oct = float(np.mean(np.abs(((diff[both] + 6) % 12) - 6) <= 0.5))
    cover_voiced = float(np.mean(have[pv_s])) if pv_s.sum() else 0.0
    # 音符単位の一致：ref_midi が一定の連続区間（≥0.15 秒）ごとに、内側（両端 50ms を除く）の pYIN 中央値が ±50 cent 以内か（立ち上がりの滑り・ビブラートの影響を除く）
    notes = []; k = 0; L = len(T)
    while k < L:
        if np.isnan(ref_midi[k]): k += 1; continue
        j = k
        while j + 1 < L and not np.isnan(ref_midi[j + 1]) and ref_midi[j + 1] == ref_midi[k] and pg[j + 1] == pg[k]: j += 1
        if (j - k + 1) * 0.01 >= 0.15:
            inner = slice(k + 5, j - 4); vv = pv_s[inner]
            if vv.sum() >= 5:
                med = float(np.median(pm_s[inner][vv])); notes.append((T[k], T[j], float(ref_midi[k]), med, abs(med - ref_midi[k]) <= 0.5, int(vv.sum())))
        k = j + 1
    note_agree50 = float(np.mean([n_[4] for n_ in notes])) if notes else 0.0
    note_agree100 = float(np.mean([abs(n_[3] - n_[2]) <= 1.0 for n_ in notes])) if notes else 0.0
    # 30 秒ブロックごとの一致率
    blocks = []
    for b0 in np.arange(0, t_end, 30):
        sel = both & (T >= b0) & (T < b0 + 30); selv = pv_s & (T >= b0) & (T < b0 + 30)
        blocks.append(dict(t0=float(b0), n=int(sel.sum()), agree50=(float(np.mean(np.abs(diff[sel]) <= 0.5)) if sel.sum() else None), cover=(float(np.mean(have[selv])) if selv.sum() else None)))
    pages_ok = [f for f in fits if f["ok"]]
    ev = dict(video=name, session=S, wav=P["wav"], sync_offset=P["offset"], lines_rows=rows, line_spacing_px=sp, n_lines=len(rows), tilt_px_per_col=tilt, k_semi_per_line=k_semi,
              midi_of_top_line=m0, display_dt_sec=dt, n_compared=best["n"], agree50=agree50, agree50_single_offset=agree50_global, agree100=agree100, agree50_octave_free=agree_oct,
              m0_by_page={str(k): v for k, v in m0_page.items()},
              coverage_of_voiced=cover_voiced, n_notes_compared=len(notes), note_agree50=note_agree50, note_agree100=note_agree100, ref_seconds=float(have.sum() * 0.01), pages_total=len(fits), pages_fitted=len(pages_ok),
              cursor_speed_px_s_median=round(v_global, 1), page_sec_median=round(x_start / v_global, 2), x_end_px=round(x_start, 1),
              pages=[dict(page=f["page"], n=f["n"], n_xc=f["n_xc"], n_inlier=f.get("n_inlier", 0), timing=f.get("timing"), t_first=round(f["t_first"], 2), t0=(None if f["t0"] is None else round(f["t0"], 2)), v=(None if f["v"] is None else round(f["v"], 1)), resid_max=(None if f.get("resid_max") is None else round(f["resid_max"], 1))) for f in fits],
              n_pages_cursor=sum(1 for f in pages_ok if f["timing"] == "cursor"), n_pages_fallback=sum(1 for f in pages_ok if f["timing"] != "cursor"),
              blocks=blocks, n_frames=g["n_frames"])
    json.dump(ev, open(os.path.join(OUT, f"{name}_eval.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, f"{name}_ref.csv"), "w", newline="", encoding="utf-8-sig") as fo:
        w = csv.writer(fo); w.writerow(["time_sec", "audio_time_sec", "ref_midi", "ref_midi_raw", "conf", "n_votes", "page", "page_m0", "ref_y_px", "pyin_midi"])
        for k in range(len(T)):
            if np.isnan(ref_y[k]): continue
            w.writerow([f"{T[k]:.2f}", f"{T[k] + P['offset']:.2f}", int(ref_midi[k]), f"{ref_midi_raw[k]:.2f}", f"{conf[k]:.2f}", nv[k], pg[k], m0_page.get(pg[k], {}).get("m0", m0), int(ref_y[k]), ("" if not pv_s[k] else f"{pm_s[k]:.2f}")])
    print(f"EVAL {name}: lines={len(rows)} sp={sp:.1f}px k={k_semi} m0={m0} dt={dt:+.2f}s agree50={agree50:.3f} (single_offset={agree50_global:.3f}) agree100={agree100:.3f} oct_free={agree_oct:.3f} "
          f"cover={cover_voiced:.3f} notes={len(notes)} note_agree50={note_agree50:.3f} note_agree100={note_agree100:.3f} ref={have.sum()*0.01:.0f}s pages={len(pages_ok)}/{len(fits)} v={v_global:.1f}", flush=True)
    plot_overlay(name, T, ref_midi, pm_s, pv_s, have, ev)
    return ev


def plot_overlay(name, T, ref_midi, pm, pv, have, ev):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in (r"C:\Windows\Fonts\meiryo.ttc", r"C:\Windows\Fonts\msgothic.ttc"):
        if os.path.exists(fp): font_manager.fontManager.addfont(fp); plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name(); break
    t_end = T[-1]; rows = int(math.ceil(t_end / 30)); fig, axes = plt.subplots(rows, 1, figsize=(16, 2.2 * rows), sharey=False)
    if rows == 1: axes = [axes]
    for i, ax in enumerate(axes):
        s = (T >= i * 30) & (T < (i + 1) * 30)
        pm_plot = np.where(pv & s, pm, np.nan); ax.plot(T[s], pm_plot[s], ".", ms=1.5, color="#d62728", label="pYIN（歌）")
        rm = np.where(have & s, ref_midi, np.nan); ax.plot(T[s], rm[s], "-", lw=2.5, color="#1f77b4", alpha=0.8, label="お手本（読取）")
        ax.set_xlim(i * 30, (i + 1) * 30); ax.grid(alpha=0.3); ax.set_ylabel("MIDI")
        b = ev["blocks"][i] if i < len(ev["blocks"]) else None
        if b and b["agree50"] is not None: ax.text(0.01, 0.85, f"一致率(±50c)={b['agree50']:.0%} n={b['n']} 被覆={b['cover']:.0%}", transform=ax.transAxes, fontsize=9)
        if i == 0: ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{name}  一致率(±50cent)={ev['agree50']:.1%} (n={ev['n_compared']})  音符単位={ev['note_agree50']:.0%} ({ev['n_notes_compared']}音)  被覆={ev['coverage_of_voiced']:.0%}  譜線={ev['n_lines']}本/{ev['line_spacing_px']:.1f}px  "
                      f"{ev['k_semi_per_line']}半音/線  上線=MIDI{ev['midi_of_top_line']}  表示遅れ dt={ev['display_dt_sec']:+.2f}s  ページ={ev['pages_fitted']}/{ev['pages_total']}", fontsize=10)
    axes[-1].set_xlabel("動画時刻 (s)"); fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{name}_overlay.png"), dpi=80); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("session"); ap.add_argument("video"); ap.add_argument("--stage", default="all", choices=["read", "eval", "all"])
    ap.add_argument("--every", type=int, default=1); ap.add_argument("--debug-frames", default=""); ap.add_argument("--max-frames", type=int, default=None); ap.add_argument("--frames-file", default=None)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    dbg = tuple(int(x) for x in a.debug_frames.split(",") if x.strip())
    if a.stage in ("read", "all"): stage_read(a.session, a.video, a.every, dbg, a.max_frames)
    if a.stage in ("eval", "all"): stage_eval(a.session, a.video, frames_file=a.frames_file)
