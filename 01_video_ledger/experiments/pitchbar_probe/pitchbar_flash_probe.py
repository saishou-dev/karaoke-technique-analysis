# -*- coding: utf-8 -*-
"""音程バーの発光から加点時刻が取れるかの可能性調査（2026-10-22・ゼミ④の先生の指摘を受けて）

先生の指摘＝「技法カウンターだけでなく、音程バーが光った所を時刻にしては。形は見えなくてよい。
ビブラートはカウンターに出るまで時間がかかる外れ値がある」。

先行作業との違い＝04_marks/flash_detect.py（方式B）は G1 イベントの時刻 T を与えて窓
[T-2.5, T-0.2] の中だけを見ていた。本スクリプトは**正解を使わずに曲の全長**を走査して
発光の候補時刻を出し、そのあとで G1 のカウンター増分時刻と突き合わせる。

段階（復号は 1 回・特徴量を CSV に保存して検出と突合は何度でもやり直せる）:
  scan     ＝ 音程バー領域を全長復号し、フレームごとの特徴量を feat_<video>.csv に保存
  detect   ＝ 特徴量から 3 つの簡易法で「光った瞬間」の候補時刻を出す
  match    ＝ 候補時刻と G1 のカウンター増分時刻を突き合わせ、技法別の遅れ分布を出す
  evidence ＝ 発光の瞬間の根拠画像（前後フレームのタイル）を書き出す

領域の定義＝設計メモ_音程バー読取_v0 の実測に従う。
  x ＝ xL 〜 xL + 1.5w ／ y ＝ yL − 0.21w 〜 yL − 5 （xL, yL, w は帯プローブ band_probe_v2_<S>.csv）
  回転（angle_deg ≈ −0.3°）は明るさの統計には効かないので当てない。

検出の 3 法（いずれも 2 フレーム差・カーソル近傍に限定・閾値は曲ごとの中央値＋k×MAD）:
  M1 輝度の時間微分  ΔV_mean      のピーク
  M2 白い画素の面積  Δwhite_frac  のピーク（S<60 かつ V>=230＝閃光）
  M3 金色画素の面積  Δgold_frac   のピーク（H10-40,S>=140,V>=80＝演奏済みバーの金色）
"""
import os, re, sys, csv, time, json, argparse, subprocess, threading, traceback
import numpy as np, pandas as pd, cv2

SESSION = 20261014
VID_DIR = rf"G:\研究アーカイブ\ビデオ素材\{SESSION}"
BAND_CSV = rf"G:\研究アーカイブ\成果物\動画解析\dataset\band_probe_v2_{SESSION}.csv"
G1_CSV = rf"G:\研究アーカイブ\成果物\動画解析\dataset\G1_events_{SESSION}_wao.csv"
OUT_DIR = r"G:\研究アーカイブ\成果物\動画解析\dataset\pitchbar_probe_2026-10-22"
FPS_NOMINAL = 59.33

VIDEOS = [
    "怪獣の花唄_20261014_104344",
    "RPG(クレヨンしんちゃんアニメバージョン)_20261014_114902",
    "共犯_20261014_111506",
]

DIV = 4                      # 領域の縮小率（明るさの統計だけなので粗くてよい）
CUR_L, CUR_R = 300, 100      # 注目領域＝カーソルの左 300px・右 100px（1/2 縮小基準）
V_BRIGHT = 200               # 「明るい画素」の輝度下限
MERGE_FRAMES = 10            # 立ち上がりの併合幅（約 0.17 秒）
STALL = 900                  # ffmpeg 停滞 kill [s]

def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "progress.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")

def imwrite_u(path, img):
    """日本語パス対策＝imencode でバイト列にしてから書く"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").write(cv2.imencode(".png", img)[1].tobytes())

# ---------- 領域 ----------
def pitchbar_roi(video, band=None):
    """音程バー領域（設計メモ v0 の実測）を (x, y, w, h) で返す。偶数に丸める。"""
    band = band if band is not None else pd.read_csv(BAND_CSV)
    r = band[band.video == video].iloc[0]
    xL, yL, w = int(r.xL), int(r.yL), int(r.w)
    x = max(0, xL)
    cw = min(3840 - x, int(1.5 * w))
    y = max(0, int(yL - 0.21 * w))
    ch = (yL - 5) - y
    return (x - x % 2, y - y % 2, cw - cw % 2, ch - ch % 2)

def band_roi(video, band=None):
    """技法カウンター帯（比較表示用）"""
    band = band if band is not None else pd.read_csv(BAND_CSV)
    r = band[band.video == video].iloc[0]
    return (int(r.xL), int(r.yL), int(r.w), int(r.h))

# ---------- 復号 ----------
def decode(path, t0, n_frames, crop, div=DIV, threads=3, hwaccel="", tag=""):
    """ffmpeg で t0 から n_frames 枚を ROI 切り出し・1/div 縮小で復号し (i, BGR) を yield する。
    stderr（showinfo の pts）はファイルへ逃がし、終了後に read_pts() で index→pts を得る。"""
    cx, cy, cw, ch = crop
    ow, oh = cw // div, ch // div
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats", "-threads", str(threads)]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-ss", f"{t0:.3f}", "-copyts", "-i", path, "-an"]
    if n_frames:
        cmd += ["-frames:v", str(n_frames)]
    cmd += ["-vf", f"crop={cw}:{ch}:{cx}:{cy},scale={ow}:{oh},showinfo",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    errpath = os.path.join(OUT_DIR, "tmp", f"ffmpeg_{os.getpid()}_{tag}.log")
    os.makedirs(os.path.dirname(errpath), exist_ok=True)
    errf = open(errpath, "wb")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, stdin=subprocess.DEVNULL, bufsize=10 ** 7)
    last = [time.time()]
    def watchdog():
        while p.poll() is None:
            if time.time() - last[0] > STALL:
                try: p.kill()
                except Exception: pass
                log(f"ffmpeg 停滞 {STALL}s 超過のため kill: {os.path.basename(path)} t0={t0:.3f}")
                return
            time.sleep(1.0)
    threading.Thread(target=watchdog, daemon=True).start()
    nbytes = ow * oh * 3
    i = 0
    try:
        while True:
            buf = p.stdout.read(nbytes)
            if len(buf) < nbytes: break
            last[0] = time.time()
            yield i, np.frombuffer(buf, np.uint8).reshape(oh, ow, 3).copy()
            i += 1
    finally:
        try: p.stdout.close()
        except Exception: pass
        try: p.wait(timeout=30)
        except Exception: p.kill()
        errf.close()
    decode.last_errpath = errpath

def read_pts(errpath):
    pts = []
    with open(errpath, "rb") as f:
        for line in f:
            m = re.search(rb"pts_time:\s*([0-9.]+)", line)
            if m: pts.append(float(m.group(1)))
    return pts

def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
    try: return float(out.stdout.strip())
    except Exception: return 0.0

# ---------- 特徴量 ----------
def find_cursor(bgr):
    """再生カーソル（領域の高さの大半を占める明るい縦線）の x（縮小座標）。無ければ None。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 100) & (hsv[..., 2] >= 180))
    col = white.sum(axis=0).astype(np.float32)
    col = np.convolve(col, np.ones(3) / 3, mode="same")
    thr = 0.35 * bgr.shape[0]
    hits = np.where(col >= thr)[0]
    if len(hits) == 0 or len(hits) > 40 // (DIV // 2 or 1): return None
    if hits.max() - hits.min() > 40 // (DIV // 2 or 1): return None
    return int(hits[np.argmax(col[hits])])

def stats_in(hsv, x0, x1):
    V = hsv[..., 2][:, x0:x1]; S = hsv[..., 1][:, x0:x1]; H = hsv[..., 0][:, x0:x1]
    mv = float(V.mean()); ms = float(S.mean())
    br = float((V >= V_BRIGHT).mean())
    wh = float(((S < 60) & (V >= 230)).mean())
    gd = float(((H >= 10) & (H <= 40) & (S >= 140) & (V >= 80)).mean())
    return mv, ms, br, wh, gd

def scan(video, hwaccel=""):
    """曲の全長を復号してフレームごとの特徴量を feat_<video>.csv に保存する"""
    t_start = time.time()
    path = os.path.join(VID_DIR, video + ".mp4")
    crop = pitchbar_roi(video)
    dur = duration(path)
    log(f"[{video}] scan 開始 ROI={crop} 長さ{dur:.1f}s")
    scale_px = 2 // (DIV // 2 or 1)      # 1/2 縮小基準の寸法を現在の縮小率に直す係数
    cl, cr = CUR_L * 2 // DIV, CUR_R * 2 // DIV
    rows = []
    frames = []          # 直近 3 フレームの HSV
    prev_cursor = None
    n = 0
    for i, bgr in decode(path, 0.0, 0, crop, hwaccel=hwaccel, tag="scan"):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        cur = find_cursor(bgr)
        if cur is not None: prev_cursor = cur
        W = bgr.shape[1]
        cx = cur if cur is not None else prev_cursor
        if cx is None: x0, x1 = 0, W
        else: x0, x1 = max(0, cx - cl), min(W, cx + cr)
        mv, ms, br, wh, gd = stats_in(hsv, x0, x1)
        amv, ams, abr, awh, agd = stats_in(hsv, 0, W)
        if len(frames) >= 2:
            p = frames[-2]
            pmv, pms, pbr, pwh, pgd = stats_in(p, x0, x1)
        else:
            pmv = pms = pbr = pwh = pgd = np.nan
        rows.append((i, cur if cur is not None else -1, x0, x1,
                     mv, ms, br, wh, gd, mv - pmv, ms - pms, br - pbr, wh - pwh, gd - pgd,
                     amv, ams, abr, awh, agd))
        frames.append(hsv)
        if len(frames) > 3: frames.pop(0)
        n += 1
    pts = read_pts(decode.last_errpath)
    nn = min(n, len(pts))
    cols = ["i", "cursor", "x0", "x1", "V", "S", "bright", "white", "gold",
            "dV", "dS", "dbright", "dwhite", "dgold", "V_all", "S_all", "bright_all", "white_all", "gold_all"]
    df = pd.DataFrame(rows[:nn], columns=cols)
    df.insert(1, "t", pts[:nn])
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, f"feat_{video}.csv"), index=False, encoding="utf-8-sig")
    log(f"[{video}] scan 完了 {nn}枚 カーソル検出率{(df.cursor >= 0).mean():.3f} {time.time()-t_start:.0f}s")

# ---------- 2 巡目＝列ごとの断面（局所化） ----------
NCOL = 96        # 領域を横 96 本の短冊に割る（原寸で約 33px ＝ 発光の粒の大きさに近い）

def profiles(hsv, ncol=NCOL):
    """金色・白・明るい の 3 種のマスクについて、横 ncol 本の短冊ごとの画素数を返す"""
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gold = ((H >= 10) & (H <= 40) & (S >= 140) & (V >= 80))
    white = ((S < 60) & (V >= 230))
    bright = (V >= V_BRIGHT)
    W = hsv.shape[1]
    edges = np.linspace(0, W, ncol + 1).astype(int)[:-1]
    out = []
    for m in (gold, white, bright):
        cs = m.sum(axis=0).astype(np.float32)
        out.append(np.add.reduceat(cs, edges).astype(np.float32))
    return np.stack(out)      # (3, ncol)

def scan2(video, hwaccel=""):
    """曲の全長を復号して、フレームごとの列断面（3 種×96 列）を npz に保存する。
    領域全体の平均では背景映像の明暗に埋もれるので、発光を横方向に局所化して見るための 2 巡目。"""
    t_start = time.time()
    path = os.path.join(VID_DIR, video + ".mp4")
    crop = pitchbar_roi(video)
    log(f"[{video}] scan2 開始 ROI={crop} 列数{NCOL}")
    prof = []
    for i, bgr in decode(path, 0.0, 0, crop, hwaccel=hwaccel, tag="scan2"):
        prof.append(profiles(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)))
    pts = read_pts(decode.last_errpath)
    nn = min(len(prof), len(pts))
    arr = np.stack(prof[:nn])        # (N, 3, ncol)
    np.savez_compressed(os.path.join(OUT_DIR, f"prof_{video}.npz"),
                        prof=arr, t=np.array(pts[:nn], np.float64), crop=np.array(crop))
    log(f"[{video}] scan2 完了 {nn}枚 {time.time()-t_start:.0f}s")

# ---------- 検出 ----------
METHODS = {
    "M1_dV":       ("dV",       "輝度の時間微分（カーソル近傍）"),
    "M2_dwhite":   ("dwhite",   "白い画素の面積の増加（カーソル近傍）"),
    "M3_dgold":    ("dgold",    "金色画素の面積の増加（カーソル近傍）"),
    "M4_dV_all":   ("dV_all",   "輝度の時間微分（領域全体）"),
    "M5_dwhite_all": ("dwhite_all", "白い画素の面積の増加（領域全体）"),
    "M6_dgold_all":  ("dgold_all",  "金色画素の面積の増加（領域全体）"),
}

def add_all_diffs(df):
    """領域全体の特徴量の 2 フレーム差を後から足す（カーソル検出率が低い曲への備え）"""
    for c in ("V_all", "S_all", "bright_all", "white_all", "gold_all"):
        df["d" + c] = df[c].diff(2)
    return df

def peaks(t, x, q=0.99, merge_frames=MERGE_FRAMES):
    """上位 (1−q) の立ち上がりを拾い、merge_frames 以内の連続は 1 つにまとめて先頭を返す。
    閾値は分位点で決める（面積系の特徴量は 0 が多数を占め、中央値＋k×MAD では MAD が 0 になって働かない）。"""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    thr = float(np.quantile(x[ok], q))
    if thr <= 0: thr = float(np.min(x[ok][x[ok] > 0])) if (x[ok] > 0).any() else 0.0
    hit = np.where(ok & (x >= thr) & (x > 0))[0]
    out = []
    last = -10 ** 9
    for j in hit:
        if j - last <= merge_frames:
            last = j
            if x[j] > out[-1][1]: out[-1] = (out[-1][0], x[j])
            continue
        last = j
        out.append((float(t[j]), float(x[j])))
    return out, float(thr)

def detect(video, q=0.99, quiet=False):
    df = add_all_diffs(pd.read_csv(os.path.join(OUT_DIR, f"feat_{video}.csv")))
    rows = []
    info = {}
    for name, (col, jp) in METHODS.items():
        pk, thr = peaks(df.t.values, df[col].values, q=q)
        info[name] = dict(threshold=round(thr, 8), n=len(pk))
        for t, v in pk:
            rows.append(dict(video=video, method=name, t=round(t, 3), peak=round(v, 8)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, f"cand_{video}.csv"), index=False, encoding="utf-8-sig")
    if not quiet:
        log(f"[{video}] detect q={q} " + " ".join(f"{m}:{info[m]['n']}件(閾値{info[m]['threshold']})" for m in METHODS))
    return out, info

# ---------- 突合 ----------
def match(videos, win_lo=-3.0, win_hi=0.5):
    g1 = pd.read_csv(G1_CSV)
    rows = []; summ = []
    for video in videos:
        cand = pd.read_csv(os.path.join(OUT_DIR, f"cand_{video}.csv"))
        ev = g1[(g1.video == video)].sort_values("video_time_sec").reset_index(drop=True)
        for name in METHODS:
            ct = np.sort(cand[cand.method == name].t.values)
            used = set()
            for _, r in ev.iterrows():
                T = float(r.video_time_sec)
                # 窓 [T+win_lo, T+win_hi]（発光は加点表示より先に出るはずなので前側を広く取る）
                sel = np.where((ct >= T + win_lo) & (ct <= T + win_hi))[0]
                best = None
                if len(sel):
                    free = [j for j in sel if j not in used]
                    pool = free if free else list(sel)
                    j = min(pool, key=lambda j: abs(T - ct[j]))
                    best = j; used.add(j)
                rows.append(dict(video=video, method=name, counter=r.counter,
                                 count_value=int(r.count_value), multi=int(r.multi_increment),
                                 T=round(T, 3),
                                 flash_t=round(float(ct[best]), 3) if best is not None else np.nan,
                                 lag=round(T - float(ct[best]), 3) if best is not None else np.nan,
                                 n_in_win=len(sel)))
            summ.append(dict(video=video, method=name, n_events=len(ev), n_cand=len(ct),
                             n_cand_unused=int(len(ct) - len(used))))
    m = pd.DataFrame(rows); s = pd.DataFrame(summ)
    m.to_csv(os.path.join(OUT_DIR, "match.csv"), index=False, encoding="utf-8-sig")
    s.to_csv(os.path.join(OUT_DIR, "match_summary.csv"), index=False, encoding="utf-8-sig")
    return m, s

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["roi", "scan", "scan2", "detect", "match"])
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--hwaccel", default="")
    ap.add_argument("--q", type=float, default=0.99)
    a = ap.parse_args()
    vids = a.videos or VIDEOS
    if a.stage == "roi":
        band = pd.read_csv(BAND_CSV)
        for v in vids:
            print(v, "pitchbar", pitchbar_roi(v, band), "band", band_roi(v, band))
    elif a.stage == "scan":
        for v in vids:
            try: scan(v, a.hwaccel)
            except Exception as e: log(f"[{v}] scan 失敗 {e}\n{traceback.format_exc()}")
    elif a.stage == "scan2":
        for v in vids:
            try: scan2(v, a.hwaccel)
            except Exception as e: log(f"[{v}] scan2 失敗 {e}\n{traceback.format_exc()}")
    elif a.stage == "detect":
        for v in vids: detect(v, a.q)
    elif a.stage == "match":
        m, s = match(vids)
        print(s.to_string())
