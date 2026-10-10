# -*- coding: utf-8 -*-
"""方式B：ヒット演出（白い光・キラキラ）の検出で加点時刻を求める。
仮説＝ヒット演出は加点の瞬間に出る。技法の種類は G1（約1秒後にどのカウンターが増えたか）で分かるので演出は技法非依存。
手順:
  1. 各 G1 イベント T について窓 [T−2.5, T−0.2] をノーツ帯 ROI（1/2 縮小）で復号。
  2. 各フレームで再生カーソル x を検出し、カーソル近傍（x−300〜x+100 px @1/2）を「注目領域」とする（無ければ直前の位置、それも無ければ ROI 全体）。
  3. 注目領域の「明るい画素（V≥200）の割合」「平均輝度」「平均彩度」を 2 フレーム前と比べ、
     明るい画素の急増（Δbright ≥ BRIGHT_JUMP）かつ 平均輝度の上昇（ΔV ≥ V_JUMP）を演出の立ち上がりとする。
     立ち上がりが 10 フレーム以内に続くものは 1 つの演出にまとめ、最初のフレームを flash_time とする。
  4. 同じ窓に複数の演出がある場合、既存の色付き印の時刻（mark_time_sec；無ければ mark_time_raw）と 0.35 s 以内で合うものを優先、無ければ最後のもの。
出力: marks_20260924.csv に flash_time_sec / flash_conf / flash_note / flash_n_in_win / flash_cursor 列を追加（既存列は変更しない）。
証拠: evidence/flash/<video>/<tech>_<n>_T<T>.png（flash−4f｜flash｜flash+4f｜flash+12f、注目領域を枠表示）"""
import os, sys, time, glob, argparse, traceback
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_marks as d

WIN_PRE, WIN_POST = 2.5, 0.2          # 窓 [T−2.5, T−0.2]
PAD = 0.15                            # 復号余白
DIV = 2                               # 1/2 縮小
CUR_L, CUR_R = 300, 100               # 注目領域：カーソルの左 300px・右 100px（1/2 縮小座標）
V_BRIGHT = 200                        # 「明るい画素」の輝度閾値
BRIGHT_JUMP = 0.012                   # 明るい画素割合の 2 フレーム差の閾値
V_JUMP = 6.0                          # 平均輝度の 2 フレーム差の閾値
MERGE_FRAMES = 10                     # 立ち上がりの併合幅
MATCH_TOL = 0.35                      # 色付き印との時間一致の許容 [s]
STALL = 600                           # ffmpeg 停滞 kill [s]（番犬）

def region(W, cursor, prev_cursor):
    cx = cursor if cursor is not None else prev_cursor
    if cx is None: return 0, W
    return max(0, cx - CUR_L), min(W, cx + CUR_R)

def features(bgr, bgr_prev, cursor, prev_cursor):
    """現フレームの注目領域（カーソル基準）を 2 フレーム前にも同じ位置で当てて比較する（領域の移動による見かけの変化を除く）"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    W = bgr.shape[1]
    x0, x1 = region(W, cursor, prev_cursor)
    V = hsv[..., 2][:, x0:x1]; S = hsv[..., 1][:, x0:x1]
    bright = float((V >= V_BRIGHT).mean()); mv = float(V.mean()); ms = float(S.mean())
    if bgr_prev is None: return bright, mv, ms, 0.0, 0.0, 0.0, (x0, x1)
    hp = cv2.cvtColor(bgr_prev, cv2.COLOR_BGR2HSV)
    Vp = hp[..., 2][:, x0:x1]; Sp = hp[..., 1][:, x0:x1]
    dbr = bright - float((Vp >= V_BRIGHT).mean()); dv = mv - float(Vp.mean()); ds = float(Sp.mean()) - ms
    return bright, mv, ms, dbr, dv, ds, (x0, x1)

def detect_video(video, save_evidence=True, calib_event=None):
    t_start = time.time()
    df = pd.read_csv(os.path.join(d.OUT_DIR, "marks_20260924.csv"))
    ev = df[df.video == video].sort_values("video_time_sec").reset_index(drop=True)
    band = pd.read_csv(d.BAND_CSV); crop = d.roi_for(video, band)
    path = os.path.join(d.VID_DIR, video + ".mp4")
    # 窓の併合
    ivs = []
    for _, r in ev.iterrows():
        a, b = r.video_time_sec - WIN_PRE - PAD, r.video_time_sec - WIN_POST + PAD
        if ivs and a <= ivs[-1][1] + 0.05: ivs[-1][1] = max(ivs[-1][1], b)
        else: ivs.append([a, b])
    d.DIV = DIV
    flashes = []          # dict(t, score, dv, ds, x0, x1, tiles)
    n_frames = 0
    for a, b in ivs:
        a = max(0.0, a); n = int((b - a) * d.FPS_NOMINAL) + 4
        buf = []          # (idx, bgr, cursor, feat)
        prev_cursor = None
        feats = []; frames = []; cursors = []
        for idx, fr in d.decode_segment(path, a, n, crop, hwaccel="", threads=3, stall_timeout=STALL):
            m = d.classify(fr); cur = d.find_cursor(m)
            f = features(fr, frames[-2] if len(frames) >= 2 else None, cur, prev_cursor)
            if cur is not None: prev_cursor = cur
            feats.append(f); frames.append(fr); cursors.append(cur)
        pts = d.read_pts(d.decode_segment.last_errpath)
        nn = min(len(frames), len(pts)); frames, pts, feats = frames[:nn], pts[:nn], feats[:nn]
        n_frames += nn
        if calib_event is not None:
            for j in range(nn):
                br, mv, ms, dbr, dv, ds, (x0, x1) = feats[j]
                print(f"{pts[j]:.3f} cur={cursors[j]} bright={br:.4f} dbr={dbr:+.4f} V={mv:.1f} dV={dv:+.1f} S={ms:.1f} dS={ds:+.1f}")
        # 立ち上がり検出
        last_j = -100
        for j in range(2, nn):
            br, mv, ms, dbr, dv, ds, (x0, x1) = feats[j]
            if dbr >= BRIGHT_JUMP and dv >= V_JUMP:
                if j - last_j <= MERGE_FRAMES:
                    last_j = j; flashes[-1]["peak"] = max(flashes[-1]["peak"], dbr); continue
                last_j = j
                tiles = None
                if save_evidence:
                    tiles = []
                    for jj in (j - 4, j, j + 4, j + 12):
                        jj = min(max(0, jj), nn - 1)
                        t = frames[jj].copy()
                        cv2.rectangle(t, (x0, 0), (x1 - 1, t.shape[0] - 1), (0, 255, 255) if jj == j else (255, 255, 255), 2)
                        t = cv2.resize(t[:, max(0, x0 - 60):min(t.shape[1], x1 + 60)], None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
                        cv2.putText(t, f"{pts[jj]:.3f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        tiles.append(t)
                    W = max(t.shape[1] for t in tiles); tiles = np.hstack([np.pad(t, ((0, 0), (0, W - t.shape[1]), (0, 0))) for t in tiles])
                flashes.append(dict(t=pts[j], dbr=dbr, dv=dv, ds=ds, peak=dbr, cursor=cursors[j], x0=x0, x1=x1, tiles=tiles))
    # イベントへの対応付け
    rows = []
    ev_dir = os.path.join(d.OUT_DIR, "evidence", "flash", video)
    if save_evidence: os.makedirs(ev_dir, exist_ok=True)
    for _, r in ev.iterrows():
        T = float(r.video_time_sec)
        cands = [f for f in flashes if T - WIN_PRE <= f["t"] <= T - WIN_POST]
        ref = r.mark_time_sec if pd.notna(r.mark_time_sec) else (r.mark_time_raw if "mark_time_raw" in r and pd.notna(r.mark_time_raw) else None)
        pick, note = None, ""
        if cands:
            if ref is not None:
                near = [f for f in cands if abs(f["t"] - ref) <= MATCH_TOL]
                if near: pick = min(near, key=lambda f: abs(f["t"] - ref)); note = "色付き印と一致"
            if pick is None:
                pick = cands[-1]; note = "窓内の最後の演出" + ("（印と不一致）" if ref is not None else "（印なし）")
        else:
            note = "演出なし"
        row = dict(video=video, counter=r.counter, count_value=int(r.count_value),
                   flash_time_sec=np.nan, flash_conf=np.nan, flash_note=note, flash_n_in_win=len(cands), flash_cursor=np.nan)
        if pick is not None:
            conf = min(1.0, pick["peak"] / (3 * BRIGHT_JUMP)) * (1.0 if pick["cursor"] is not None else 0.7)
            row.update(flash_time_sec=round(pick["t"], 3), flash_conf=round(conf, 3), flash_cursor=pick["cursor"] if pick["cursor"] is not None else -1)
            if save_evidence and pick["tiles"] is not None:
                d.imwrite_u(os.path.join(ev_dir, f"{r.counter}_{int(r.count_value):03d}_T{T:.3f}.png"), pick["tiles"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(d.OUT_DIR, "partial", video + "_flash.csv"), index=False, encoding="utf-8-sig")
    n_ok = sum(1 for r in rows if pd.notna(r["flash_time_sec"]))
    d.log(f"[{video}] 方式B 完了 抽出{n_ok}/{len(rows)} 演出{len(flashes)} 復号{n_frames}枚 {time.time()-t_start:.0f}s")

def worker(video):
    try: detect_video(video)
    except Exception as e: d.log(f"[{video}] 方式B 失敗 {e}\n{traceback.format_exc()}")

def merge():
    df = pd.read_csv(os.path.join(d.OUT_DIR, "marks_20260924.csv"))
    parts = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(d.OUT_DIR, "partial", "*_flash.csv")))]
    fl = pd.concat(parts, ignore_index=True)
    df = df.drop(columns=[c for c in ["flash_time_sec", "flash_conf", "flash_note", "flash_n_in_win", "flash_cursor"] if c in df.columns])
    df = df.merge(fl, on=["video", "counter", "count_value"], how="left")
    df["lead_flash_sec"] = (df.video_time_sec - df.flash_time_sec).round(3)
    df.to_csv(os.path.join(d.OUT_DIR, "marks_20260924.csv"), index=False, encoding="utf-8-sig")
    d.log(f"方式B 結合 {int(df.flash_time_sec.notna().sum())}/{len(df)} 抽出率 {df.flash_time_sec.notna().mean():.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--videos", nargs="*")
    ap.add_argument("--merge", action="store_true"); ap.add_argument("--calib", nargs=2, metavar=("VIDEO", "T"), help="1 イベント窓の特徴量を印字")
    a = ap.parse_args()
    if a.calib:
        v, T = a.calib[0], float(a.calib[1])
        df = pd.read_csv(os.path.join(d.OUT_DIR, "marks_20260924.csv"))
        sub = df[(df.video == v) & ((df.video_time_sec - T).abs() < 0.01)]
        # 1 イベントだけの一時フレームでウィンドウを流す
        band = pd.read_csv(d.BAND_CSV); crop = d.roi_for(v, band); d.DIV = DIV
        prev = None; feats = []; pts_all = []; frs = []
        n = int((WIN_PRE + 0.4) * d.FPS_NOMINAL)
        for idx, fr in d.decode_segment(os.path.join(d.VID_DIR, v + ".mp4"), max(0, T - WIN_PRE - PAD), n, crop, hwaccel="", threads=4, stall_timeout=STALL):
            m = d.classify(fr); cur = d.find_cursor(m); f = features(fr, frs[-2] if len(frs) >= 2 else None, cur, prev)
            if cur is not None: prev = cur
            feats.append((cur, f)); frs.append(fr)
        pts = d.read_pts(d.decode_segment.last_errpath)
        for j in range(min(len(pts), len(feats))):
            cur, (br, mv, ms, dbr, dv, ds, _) = feats[j]
            flag = "<<" if (dbr >= BRIGHT_JUMP and dv >= V_JUMP) else ""
            print(f"{pts[j]:.3f} (T{pts[j]-T:+.3f}) cur={cur} bright={br:.4f} dbr={dbr:+.4f} V={mv:.1f} dV={dv:+.1f} S={ms:.1f} dS={ds:+.1f} {flag}")
        sys.exit(0)
    if not a.merge:
        vids = a.videos or sorted(pd.read_csv(os.path.join(d.OUT_DIR, "marks_20260924.csv")).video.unique())
        d.log(f"方式B（演出検出）開始 曲数{len(vids)} 並列{a.workers} 番犬{STALL}s")
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            for _ in pool.imap_unordered(worker, vids): pass
    merge()
