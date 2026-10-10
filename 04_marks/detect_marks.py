# -*- coding: utf-8 -*-
"""
加点マーク出現時刻の自動抽出（9/24 パイロット）

方針（8月 v3 の再実装＋時間的持続性の条件を追加）
  1. 録画（4K・約59fps）を ffmpeg（d3d11va ハード復号）で G1 イベント周辺だけ復号し、
     ノーツ帯 ROI を切り出して 1/2 縮小、BGR で受け取る。
  2. 各フレームを HSV に変換し、技法ごとの色相ファミリ（赤・青・緑・シアン・マゼンタ）と
     補助ビット（黄コア・白コア）を 1 バイトのビットマスクにする。
  3. 「出現」= その画素が直前（2,4,6 フレーム前）は OFF で、当該フレームから 12 フレーム（約0.2秒）
     以上 ON が続く（= 静止した表示物）。演出のキラキラ・背景MVは動くので落ちる。
  4. 出現画素を連結成分にし、面積・外接矩形の上限下限で「小さな印」だけ残す。
     出現画素が ROI の 4% を超えるフレームは全面変化（ページ切替・フラッシュ演出）として捨てる。
     再生カーソル（明るい縦線）の近傍は音程バーの塗り進みが出るので除外する。
  5. 各曲・各技法でイベント（G1 の T）と候補を時刻順に対応付ける。
出力: marks_20260924.csv（video,counter,count_value,video_time_sec,mark_time_sec,lead_sec,conf,method,note ほか）
"""
import os, sys, re, csv, json, time, threading, subprocess, argparse, traceback
from collections import deque
import numpy as np, pandas as pd, cv2

VID_DIR = r"G:\研究アーカイブ\ビデオ素材\20260924"
OUT_DIR = r"G:\研究アーカイブ\成果物\動画解析\marks_20260928"
G1_CSV = r"G:\研究アーカイブ\成果物\動画解析\dataset\G1_events_all.csv"
BAND_CSV = r"G:\研究アーカイブ\成果物\動画解析\dataset\band_probe_v2_20260924.csv"
SESSION = 20260924
MARKS_CSV = os.path.join(OUT_DIR, "marks_20260924.csv")
PARTIAL_DIR = os.path.join(OUT_DIR, "partial")
FPS_NOMINAL = 59.33

def configure(session):
    """セッション切替：録画フォルダ・帯位置・G1 の session・出力名を差し替える（9/24 は従来どおり）"""
    global VID_DIR, BAND_CSV, SESSION, MARKS_CSV, PARTIAL_DIR
    session = int(session); SESSION = session
    VID_DIR = rf"G:\研究アーカイブ\ビデオ素材\{session}"
    if session == 20260924:
        BAND_CSV = r"G:\研究アーカイブ\成果物\動画解析\dataset\band_probe_v2_20260924.csv"
        MARKS_CSV = os.path.join(OUT_DIR, "marks_20260924.csv"); PARTIAL_DIR = os.path.join(OUT_DIR, "partial")
    else:
        BAND_CSV = os.path.join(OUT_DIR, f"band_probe_marks_{session}.csv")
        MARKS_CSV = os.path.join(OUT_DIR, f"marks_{session}.csv"); PARTIAL_DIR = os.path.join(OUT_DIR, f"partial_{session}")
    os.makedirs(PARTIAL_DIR, exist_ok=True)
    # テンプレート倍率：このセッションの帯スケール中央値 / 9-24 の 0.934（9-24 のテンプレートを画面サイズに合わせる）
    global TEMPLATE_SCALE, TEMPLATES
    try:
        _b = pd.read_csv(BAND_CSV); TEMPLATE_SCALE = float(_b.scale.median()) / 0.934
    except Exception:
        TEMPLATE_SCALE = 1.0
    TEMPLATES = {}
    global NCC_MIN_RED, GATE_FACTOR
    ov = SESSION_OVERRIDES.get(session, {})
    NCC_MIN_RED = ov.get("NCC_MIN_RED", 0.6); GATE_FACTOR = ov.get("GATE_FACTOR", 1.0)
    global SAT_MIN
    SAT_MIN = ov.get("SAT_MIN", 110)   # 8/18 で S>=70 にすると暖色の背景MVが赤ファミリに溢れ、場面転換ゲートで印ごと落ちた（9/29）

# ---- ビットマスク定義 ----
B_RED, B_BLUE, B_GREEN, B_CYAN, B_MAGENTA, B_YEL, B_WHITE = 1, 2, 4, 8, 16, 32, 64
SAT_MIN = 110  # 色ファミリの彩度下限（既定。9/24 で検証済み。8/12 は録画が低彩度のため SESSION_OVERRIDES で 70）
FAMILY = {  # 技法 → (色ファミリ, コア条件, 先行時間の事前値[s], 最大先行[s])
    "shakuri":   (B_RED,     "white",  0.30, 2.2),
    "accent":    (B_RED,     "yellow", 0.50, 2.2),
    "kobushi":   (B_BLUE,    None,     0.47, 2.2),
    "vibrato":   (B_GREEN,   None,     0.90, 3.0),
    "hammering": (B_BLUE,    None,     0.50, 2.2),   # 仮：こぶしと同じ青ファミリ
    "fall":      (B_MAGENTA, None,     0.51, 2.2),
}
DIV = 1                               # ROI の縮小率（1=等倍。1/2 縮小では赤の細い輪郭が断片化した）
_s = 2.0 / DIV; _k = _s * _s          # 1/2 縮小基準の寸法・面積をスケールする係数
PRE_PAD, POST_PAD = 0.15, 0.55      # 窓の前後余白（持続性判定に +30 フレーム必要）
LOOK_BACK, LOOK_AHEAD = 6, 30        # 出現判定に使う前後フレーム数
FLASH_FRAC = 0.01                    # 出現画素がこの割合を超えたら全面変化として捨てる
MAX_COMP_PER_FRAME = 4               # 1フレームに同色の候補がこれより多ければ場面転換として捨てる
DIFF_FLASH = 0.25                    # 前フレームとの大差画素の割合がこれを超えたらフラッシュ演出
AREA_MIN, AREA_MAX = 20, int(1500 * _k)               # 出現成分の最小面積は固定 20px（断片で良い）、印全体の最大面積は 1/2 縮小基準 1500
BBOX_MAX = int(110 * _s)                              # 外接矩形の最大辺
CURSOR_EXCL = int(45 * _s)                            # カーソル通過位置からの除外距離 [px]
MIN_W, MIN_H, AREA_MARK_MIN = int(12 * _s), int(9 * _s), int(60 * _k)   # 印の最小寸法（音程軌跡の細い断片を落とす）
ONSET_RATIO_MIN = 0.0                # 瞬時出現の下限（診断の結果、真の印でも低いので未使用）
CORE_MIN_PX, CORE_MIN_FRAC, RING_MAX_FRAC = 0, 0.0, 1.0   # コア判定の閾値（診断の結果、単独では分離せず未使用）
NCC_MIN_RED = 0.6                    # 赤ファミリ（しゃくり／アクセント）のテンプレート相関の下限
GATE_FACTOR = 1.0                    # 精緻化の種類相関ゲートに掛ける係数（セッション上書き用）
SESSION_OVERRIDES = {                # 録画品質に合わせたセッション別の上書き（9/29 目視・診断に基づく）
    20260812: dict(NCC_MIN_RED=0.3, GATE_FACTOR=0.6, SAT_MIN=70),   # ぼやけ＋低彩度：真の ʃ でも相関 0.36・赤が S<120
}

TPL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
TEMPLATES = {}   # fam → [(kind, gray_template, mask)]
TEMPLATE_SCALE = 1.0   # configure() で設定（セッションの画面スケール / 0.934）
def load_templates():
    global TEMPLATES
    if TEMPLATES: return TEMPLATES
    spec = {"shakuri": B_RED, "accent": B_RED, "vibrato": B_GREEN, "kobushi": B_BLUE, "hammering": B_BLUE}
    for kind, fam in spec.items():
        fp = os.path.join(TPL_DIR, f"{kind}_color.png")
        if os.path.exists(fp):
            bgr = cv2.imdecode(np.fromfile(fp, np.uint8), cv2.IMREAD_COLOR)
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            m = classify(bgr)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            icon = ((m & fam) > 0) | ((hsv[..., 2] >= 180) & (hsv[..., 1] < 100)) | ((m & B_YEL) > 0)
            icon = cv2.dilate(icon.astype(np.uint8), np.ones((5, 5), np.uint8))
            # 最大連結成分（アイコン本体）だけをマスクに使う
            n, lab, st, _ = cv2.connectedComponentsWithStats(icon, 8)
            if n > 1:
                k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); icon = (lab == k).astype(np.uint8)
            f = TEMPLATE_SCALE / DIV
            if abs(f - 1.0) > 1e-3:
                g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
                icon = cv2.resize(icon, g.shape[::-1], interpolation=cv2.INTER_NEAREST)
            TEMPLATES.setdefault(fam, []).append((kind, g, icon * 255))
    return TEMPLATES

def masked_ncc(I, T, M):
    """マスク M（0/1）内だけで計算する正規化相関係数（TM_CCOEFF_NORMED のマスク版）を FFT 相関で一括計算する。"""
    I = I.astype(np.float32); T = T.astype(np.float32); M = (M > 0).astype(np.float32)
    n = float(M.sum())
    if n < 10: return np.full((1, 1), -1.0, np.float32)
    TM = T * M
    sum_it = cv2.matchTemplate(I, TM, cv2.TM_CCORR)          # Σ M·I·T
    sum_i = cv2.matchTemplate(I, M, cv2.TM_CCORR)            # Σ M·I
    sum_i2 = cv2.matchTemplate(I * I, M, cv2.TM_CCORR)       # Σ M·I²
    sum_t = float(TM.sum()); sum_t2 = float((TM * T).sum())
    num = sum_it - sum_i * sum_t / n
    var_i = np.maximum(sum_i2 - sum_i * sum_i / n, 1e-6)
    var_t = max(sum_t2 - sum_t * sum_t / n, 1e-6)
    return num / np.sqrt(var_i * var_t)

def ncc_verify(gray_patch, fam):
    """候補周辺のグレー画像と、そのファミリの各テンプレート（アイコン画素のみ）との正規化相関の最大値と種類を返す"""
    best, kind = -1.0, ""
    for k, tpl, msk in load_templates().get(fam, []):
        if gray_patch.shape[0] < tpl.shape[0] or gray_patch.shape[1] < tpl.shape[1]: continue
        r = masked_ncc(gray_patch, tpl, msk)
        r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
        v = float(r.max())
        if v > best: best, kind = v, k
    return best, kind

def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT_DIR, "progress.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")

def heartbeat(msg):
    """窓ごとの生存記録（10/08）。progress.log は曲完了ごと（30〜110分）にしか増えず、番犬（job_watchdog.py）が
    「10分間進捗なし」と誤判定して 9/29 22:44 に正常稼働中のジョブを停止した。番犬には --extra-files でこのファイルを渡す。"""
    try:
        with open(os.path.join(OUT_DIR, "tmp", "heartbeat.log"), "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}" + chr(10))
    except Exception: pass

def imwrite_u(path, img):
    open(path, "wb").write(cv2.imencode(".png", img)[1].tobytes())

# ---------- 復号 ----------
FFMPEG_THREADS = int(os.environ.get("MARKS_FFMPEG_THREADS", "4"))
def decode_segment(path, t0, n_frames, crop, hwaccel=None, threads=None, stall_timeout=240):   # 10/08: 600→240s（真の停滞は窓を打ち切って先へ進む）
    threads = threads or FFMPEG_THREADS
    """ffmpeg で t0 から n_frames 枚を ROI 切り出し・1/2 縮小で復号し、(frame_index, BGR画像) を逐次 yield する。
    stderr（showinfo の pts）はファイルに逃がし、stdout だけをパイプで読む（デッドロック回避）。
    終了後 read_pts() で index→pts_time の表を得る。stall_timeout 秒フレームが来なければ ffmpeg を kill する。"""
    cx, cy, cw, ch = crop
    ow, oh = cw // DIV, ch // DIV
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats", "-threads", str(threads)]
    if hwaccel: cmd += ["-hwaccel", hwaccel]
    cmd += ["-ss", f"{t0:.3f}", "-copyts", "-i", path, "-frames:v", str(n_frames), "-an",
            "-vf", f"crop={cw}:{ch}:{cx}:{cy},scale={ow}:{oh},showinfo",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    errpath = os.path.join(OUT_DIR, "tmp", f"ffmpeg_{os.getpid()}.log")
    os.makedirs(os.path.dirname(errpath), exist_ok=True)
    errf = open(errpath, "wb")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, stdin=subprocess.DEVNULL, bufsize=10**7)
    last = [time.time()]; killed = [False]
    def watchdog():
        while p.poll() is None:
            if time.time() - last[0] > stall_timeout:
                killed[0] = True
                try: p.kill()
                except Exception: pass
                log(f"ffmpeg 停滞 {stall_timeout}s 超過のため kill: t0={t0:.3f} {os.path.basename(path)}")
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
        except Exception:
            p.kill()
        errf.close()
    decode_segment.last_errpath = errpath
    decode_segment.last_killed = killed[0]

def read_pts(errpath):
    pts = []
    with open(errpath, "rb") as f:
        for line in f:
            m = re.search(rb"pts_time:\s*([0-9.]+)", line)
            if m: pts.append(float(m.group(1)))
    return pts

# ---------- 画素分類 ----------
def classify(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = np.zeros(H.shape, np.uint8)
    sat = (S >= SAT_MIN) & (V >= 100)
    m[((H <= 14) | (H >= 170)) & (S >= SAT_MIN) & (V >= 100)] |= B_RED    # しゃくり・アクセントの赤輪郭
    m[(H >= 78) & (H <= 125) & sat] |= B_BLUE                              # こぶしの青渦（シアン〜青のグラデ）
    m[(H >= 45) & (H <= 77) & sat] |= B_GREEN                              # ビブラートの緑 M
    m[(H >= 82) & (H <= 99) & sat & (V >= 120)] |= B_CYAN                  # （ハンマリング仮・現状は青ファミリを使用）
    m[(H >= 140) & (H <= 169) & sat] |= B_MAGENTA                          # フォール（仮：マゼンタ系）
    m[(H >= 18) & (H <= 35) & (S >= 140) & (V >= 180)] |= B_YEL            # アクセントの黄コア
    m[(S < 100) & (V >= 180)] |= B_WHITE                                   # しゃくりの白〜クリームのコア／カーソル
    return m

def find_cursor(m):
    """再生カーソル（帯の高さの大半を占める明るい縦線）の x を返す。無ければ None。
    白ビット（S<100,V>=180）の列プロファイルを 5px で平滑化し、高さの 35% 以上の狭いピークを採る。"""
    col = (m & B_WHITE).astype(bool).sum(axis=0).astype(np.float32)
    col = np.convolve(col, np.ones(5) / 5, mode="same")
    thr = 0.35 * m.shape[0]
    hits = np.where(col >= thr)[0]
    if len(hits) == 0 or len(hits) > 40: return None       # 無い／フラッシュで全面明るい
    if hits.max() - hits.min() > 40: return None            # 離れた複数のピーク＝信頼できない
    return int(hits[np.argmax(col[hits])])

# ---------- 出現候補検出（ストリーミング） ----------
class OnsetDetector:
    def __init__(self, families):
        self.families = families          # 検出したい色ファミリのビット集合
        self.buf = deque()                # (pts, mask, bgr)
        self.cands = []                   # dict(list)
        self.flash_frames = 0
        self.scene_frames = 0
        self.flash_recent = 0
        self.prev_gray = None
        self.kernel = np.ones((3, 3), np.uint8)

    def push(self, idx, bgr):
        pts = idx
        m = classify(bgr)
        fm = {fam: cv2.dilate(((m & fam) > 0).astype(np.uint8), self.kernel) > 0 for fam in self.families}   # 3x3 膨張で 1px のゆらぎを吸収
        g = cv2.cvtColor(cv2.resize(bgr, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if self.prev_gray is not None:
            d = (cv2.absdiff(g, self.prev_gray) > 30).mean()
            if d > DIFF_FLASH:
                self.flash_frames += 1
                self.flash_recent = LOOK_AHEAD      # 以後 30 フレームは信頼度を下げる
                self.cands.append(dict(fam=0, t=pts, flash=1, kind="diff"))
            elif self.flash_recent > 0:
                self.flash_recent -= 1
        self.prev_gray = g
        self.buf.append((pts, m, bgr, fm, find_cursor(m)))
        need = LOOK_BACK + LOOK_AHEAD + 1
        if len(self.buf) >= need:
            self._eval(LOOK_BACK)
            self.buf.popleft()

    def flush(self):
        # 末尾は先読みが足りないので、先読み 12 フレーム以上あれば持続性を短縮して評価する
        while len(self.buf) > LOOK_BACK + 13:
            self._eval(LOOK_BACK)
            self.buf.popleft()

    def _eval(self, j):
        b = self.buf
        n_ahead = len(b) - 1 - j
        pts_j = b[j][0]
        m_j = b[j][1]
        Hh, Ww = m_j.shape
        # 直近 15 フレーム〜3 フレーム先のカーソル位置（音程軌跡はカーソル直下で生まれるので除外に使う）
        cursor_xs = [b[k][4] for k in range(max(0, j - 15), min(len(b), j + 4)) if b[k][4] is not None]
        for fam in self.families:
            M = lambda k: b[k][3][fam]
            stable = M(j) & M(j + 4) & M(j + 8) & M(j + 12)
            onset = stable & ~M(j - 2) & ~M(j - 4) & ~M(j - 6)
            n_on = int(onset.sum())
            if n_on == 0: continue
            if n_on > FLASH_FRAC * Hh * Ww:
                self.scene_frames += 1
                self.cands.append(dict(fam=fam, t=pts_j, flash=1, kind="onset_frac"))
                continue
            on8 = cv2.dilate(onset.astype(np.uint8), self.kernel)
            n, lab, stats, cent = cv2.connectedComponentsWithStats(on8, 8)
            full = M(j + 8).astype(np.uint8)
            frame_cands = []
            for i in range(1, n):
                x, y, w, h, a = stats[i]
                if a < AREA_MIN or w > BBOX_MAX or h > BBOX_MAX or w < 5 or h < 5: continue
                # 出現後 8 フレームのマスクで「印全体」の成分を取り、大きすぎる塊（背景・バー）を除く
                PAD = int(60 * _s)
                x0, y0 = max(0, x - PAD), max(0, y - PAD)
                x1, y1 = min(Ww, x + w + PAD), min(Hh, y + h + PAD)
                sub = full[y0:y1, x0:x1]
                n2, lab2, st2, _ = cv2.connectedComponentsWithStats(sub, 8)
                ov = lab2[(lab == i)[y0:y1, x0:x1]]
                ov = ov[ov > 0]
                if len(ov) == 0: continue
                k = np.bincount(ov).argmax()
                fx, fy, fw, fh, fa = st2[k]
                touches = (fx == 0 and x0 > 0) or (fy == 0 and y0 > 0) or (fx + fw == x1 - x0 and x1 < Ww) or (fy + fh == y1 - y0 and y1 < Hh)
                if fa > AREA_MAX or fw > BBOX_MAX or fh > BBOX_MAX or touches: continue
                if fa < AREA_MARK_MIN or fw < MIN_W or fh < MIN_H: continue
                comp_sub = (lab2 == k)
                comp = np.zeros(m_j.shape, bool)
                comp[y0:y1, x0:x1] = comp_sub
                # 持続性：以後 30 フレームで成分の 50% 以上が残る割合（矩形内だけで計算）
                cov = []
                for kk in range(j + 1, j + 1 + min(LOOK_AHEAD, n_ahead)):
                    cov.append((M(kk)[y0:y1, x0:x1] & comp_sub).sum() / fa)
                pers = float(np.mean(np.array(cov) >= 0.5))
                if pers < 0.6: continue
                cx, cy = x0 + fx + fw / 2, y0 + fy + fh / 2
                onset_ratio = float(((lab == i)[y0:y1, x0:x1] & comp_sub).sum()) / fa
                g_now = cv2.cvtColor(b[j][2][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                g_prev = cv2.cvtColor(b[j - 2][2][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                pop = float(((cv2.absdiff(g_now, g_prev) >= 40) & comp_sub).sum()) / fa
                hsv_c = cv2.cvtColor(b[j + 8][2][y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                sat = float(hsv_c[..., 1][comp_sub].mean()); val = float(hsv_c[..., 2][comp_sub].mean())
                if onset_ratio < ONSET_RATIO_MIN: continue
                near_cursor = any(abs(cx - cxr) < CURSOR_EXCL for cxr in cursor_xs)
                # 凸包の内側（成分自身を除く）にある明るいコア画素と、外側リングの明るさ
                comp_u8 = comp_sub.astype(np.uint8)
                pts_c = cv2.findNonZero(comp_u8)
                hull = cv2.convexHull(pts_c)
                hull_mask = np.zeros_like(comp_u8); cv2.fillConvexPoly(hull_mask, hull, 1)
                interior = (hull_mask > 0) & ~comp_sub
                ring = (cv2.dilate(hull_mask, np.ones((7, 7), np.uint8)) > 0) & ~(hull_mask > 0)
                bb = b[j + 8][1][y0:y1, x0:x1]
                bright = hsv_c[..., 2] >= 190
                yel = int(((bb & B_YEL) > 0)[interior].sum()); wht = int(((bb & B_WHITE) > 0)[interior].sum())
                core_in = int((bright & interior).sum()); n_int = int(interior.sum())
                core_frac = core_in / max(1, n_int)
                ring_frac = float((bright & ring).sum()) / max(1, int(ring.sum()))
                core = "yellow" if (yel >= 8 * _k and yel >= wht) else ("white" if wht >= 5 * _k else "none")
                # テンプレート照合（候補の外接矩形 ±25px のグレー画像）
                mx, my = int(60 * _s), int(50 * _s)     # テンプレート（最大 ~100x92px）が必ず収まる余白
                gx0, gy0 = max(0, x0 + fx - mx), max(0, y0 + fy - my)
                gx1, gy1 = min(Ww, x0 + fx + fw + mx), min(Hh, y0 + fy + fh + my)
                gpatch = cv2.cvtColor(b[j + 8][2][gy0:gy1, gx0:gx1], cv2.COLOR_BGR2GRAY)
                ncc, kind = ncc_verify(gpatch, fam)
                if fam == B_RED and ncc < NCC_MIN_RED: continue      # 赤ファミリはテンプレート照合で門前払い
                if core_in < CORE_MIN_PX or core_frac < CORE_MIN_FRAC or ring_frac > RING_MAX_FRAC: continue
                conf = (ncc if fam == B_RED else min(1.0, fa / (80.0 * _k))) * pers
                if self.flash_recent > 0: conf *= 0.7   # 直近にフラッシュ演出があれば信頼度を下げる
                frame_cands.append(dict(fam=fam, t=pts_j, x=float(cx), y=float(cy), area=int(fa),
                                       w=int(fw), h=int(fh), pers=pers, conf=conf, core=core,
                                       onset_ratio=round(onset_ratio, 3), pop=round(pop, 3), sat=round(sat, 1), val=round(val, 1),
                                       core_in=core_in, core_frac=round(core_frac, 3), ring_frac=round(ring_frac, 3), n_int=n_int,
                                       ncc=round(ncc, 3), kind=kind,
                                       yel=yel, wht=wht, near_cursor=int(near_cursor), flash=0,
                                       after_flash=int(self.flash_recent > 0), j=j))
            if sum(1 for c in frame_cands if not c["near_cursor"]) > MAX_COMP_PER_FRAME:
                self.scene_frames += 1
                self.cands.append(dict(fam=fam, t=pts_j, flash=1, kind="scene"))
                continue
            for c in frame_cands:
                c["strip"] = self._strip(c.pop("j"), c["x"], c["y"])
                self.cands.append(c)

    def _strip(self, j, cx, cy):
        """証拠画像：出現 6 フレーム前・出現フレーム・8 フレーム後・20 フレーム後 の 200x140 切り出し（2倍拡大）"""
        tiles = []
        for k in (j - 6, j, j + 8, min(j + 20, len(self.buf) - 1)):
            pts, m, bgr, _, _ = self.buf[k]
            Hh, Ww = bgr.shape[:2]
            cw_, ch_ = int(100 * _s), int(70 * _s)
            x0 = int(min(max(0, cx - cw_), Ww - 2 * cw_)); y0 = int(min(max(0, cy - ch_), Hh - 2 * ch_))
            t = cv2.resize(bgr[y0:y0 + 2 * ch_, x0:x0 + 2 * cw_], (400, 280), interpolation=cv2.INTER_AREA if _s > 1 else cv2.INTER_NEAREST)
            cv2.putText(t, f"#{int(pts)} ({(k - j):+d}f)", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if k == j:
                cv2.rectangle(t, (2, 2), (t.shape[1] - 3, t.shape[0] - 3), (0, 255, 255), 2)
            tiles.append(t)
        return np.hstack(tiles)

# ---------- 1 曲の処理 ----------
def roi_for(video, band):
    r = band[band.video == video].iloc[0]
    xL, yL, scale = int(r.xL), int(r.yL), float(r.scale)
    cx = max(0, xL - 40)
    cw = min(3840 - cx, int(3620 * scale / 0.934))
    cy = max(0, yL - 570); ch = (yL - 40) - cy    # 下端はカウンター帯の混入を避け yL−40
    cw -= cw % 2; ch -= ch % 2
    return (cx, cy, cw, ch)

def process_video(video, hwaccel="", save_strips=True):
    t_start = time.time()
    g1 = pd.read_csv(G1_CSV)
    ev = g1[(g1.session == SESSION) & (g1.multi_increment == 0) & (g1.video == video)].copy()
    ev = ev.sort_values("video_time_sec").reset_index(drop=True)
    band = pd.read_csv(BAND_CSV)
    crop = roi_for(video, band)
    path = os.path.join(VID_DIR, video + ".mp4")
    fams = {FAMILY[c][0] for c in ev.counter.unique() if c in FAMILY}
    # 窓の併合
    ivs = []
    for _, r in ev.iterrows():
        lmax = FAMILY.get(r.counter, (0, None, 0.5, 2.2))[3]
        a, b_ = r.video_time_sec - lmax - PRE_PAD, r.video_time_sec + 0.5 + POST_PAD
        if ivs and a <= ivs[-1][1] + 0.05: ivs[-1][1] = max(ivs[-1][1], b_)
        else: ivs.append([a, b_])
    log(f"[{video}] イベント{len(ev)}件 窓{len(ivs)}本 合計{sum(b_-a for a,b_ in ivs):.1f}s ROI={crop}")
    cands = []; n_frames_total = 0; flash_total = 0
    for a, b_ in ivs:
        a = max(0.0, a)
        n = int((b_ - a) * FPS_NOMINAL) + 4
        det = OnsetDetector(fams)
        for idx, fr in decode_segment(path, a, n, crop, hwaccel):
            det.push(idx, fr); n_frames_total += 1
        det.flush()
        heartbeat(f"[{video}] 窓 {a:.1f}-{b_:.1f} 復号{n_frames_total}枚")
        pts = read_pts(decode_segment.last_errpath)
        if decode_segment.last_killed:
            log(f"[{video}] 窓 {a:.2f}-{b_:.2f} は ffmpeg 停滞で打ち切り（{len(pts)}枚まで）")
        for c in det.cands:   # index → pts_time
            k = int(c["t"]); c["t"] = pts[k] if k < len(pts) else (a + k / FPS_NOMINAL)
            if "strip" in c and c["strip"] is not None: pass
        flash_total += det.flash_frames
        cands.extend(det.cands)
    # 候補の重複除去（同ファミリ・同位置 25px・0.35s 以内 → 早い方）
    real = [c for c in cands if not c["flash"]]
    real.sort(key=lambda c: c["t"])
    uniq = []
    for c in real:
        dup = False
        for u in uniq:
            if u["fam"] == c["fam"] and abs(u["t"] - c["t"]) < 0.35 and abs(u["x"] - c["x"]) < 25 * _s and abs(u["y"] - c["y"]) < 25 * _s:
                dup = True; break
        if not dup: uniq.append(c)
    flash_times = sorted({round(c["t"], 2) for c in cands if c["flash"]})
    scene_total = sum(1 for c in cands if c["flash"] and c.get("kind") != "diff")
    # 対応付け
    rows = []
    taken = set()
    strip_dir = os.path.join(OUT_DIR, "evidence", "samples", video)
    if save_strips: os.makedirs(strip_dir, exist_ok=True)
    for _, r in ev.iterrows():
        T = float(r.video_time_sec); tech = r.counter
        fam, core_req, prior, lmax = FAMILY[tech]
        pool = []
        for i, c in enumerate(uniq):
            if c["fam"] != fam or i in taken: continue
            if not (T - lmax <= c["t"] <= T + 0.5): continue
            if fam == B_RED and c.get("kind", "") != tech: continue     # しゃくり／アクセントはテンプレート照合の種類で振り分け
            pool.append((i, c))
        note = ""
        n_flash_in_win = sum(1 for ft in flash_times if T - lmax <= ft <= T + 0.5)
        if pool:
            usable = [(i, c) for i, c in pool if not c["near_cursor"]]
            if not usable:
                note = "候補がカーソル近傍のみ（バー塗り進みの可能性）"; best = None
            else:
                usable.sort(key=lambda ic: -(ic[1]["conf"] - 0.10 * abs((T - ic[1]["t"]) - prior)))
                best = usable[0]
        else:
            best = None
            note = "候補なし" + (f"（全面変化フレーム{n_flash_in_win}）" if n_flash_in_win else "")
            # 同ファミリの候補が窓内にあるがコア条件で外れた場合
            other = [c for i, c in enumerate(uniq) if c["fam"] == fam and T - lmax <= c["t"] <= T + 0.5 and i not in taken]
            if other: note += f"；同色候補{len(other)}件は種類不一致"
        row = dict(video=video, counter=tech, count_value=int(r.count_value), video_time_sec=T,
                   mark_time_sec="", lead_sec="", conf="", method="v4_hsv_persist", note=note,
                   mark_x="", mark_y="", area="", core="", after_flash="", n_cand_in_win=len(pool), n_flash_in_win=n_flash_in_win)
        if best is not None:
            i, c = best; taken.add(i)
            row.update(mark_time_sec=round(c["t"], 3), lead_sec=round(T - c["t"], 3), conf=round(c["conf"], 3),
                       mark_x=int(c["x"] * DIV + crop[0]), mark_y=int(c["y"] * DIV + crop[1]), area=c["area"], core=c["core"],
                       after_flash=c["after_flash"],
                       note=("代替候補あり" if len(pool) > 1 else "") + ("；直前にフラッシュ演出" if c["after_flash"] else ""))
            if save_strips:
                fn = f"{tech}_{int(r.count_value):03d}_T{T:.3f}_M{c['t']:.3f}.png"
                imwrite_u(os.path.join(strip_dir, fn), c["strip"])
        rows.append(row)
    os.makedirs(PARTIAL_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(PARTIAL_DIR, video + ".csv"), index=False, encoding="utf-8-sig")
    # 候補一覧も保存（診断用）
    pd.DataFrame([{k: v for k, v in c.items() if k != "strip"} for c in uniq]).to_csv(
        os.path.join(PARTIAL_DIR, video + "_cands.csv"), index=False, encoding="utf-8-sig")
    n_ok = sum(1 for r in rows if r["mark_time_sec"] != "")
    log(f"[{video}] 完了 抽出{n_ok}/{len(rows)} 候補{len(uniq)} フラッシュ{flash_total} 場面転換{scene_total} 復号{n_frames_total}枚 {time.time()-t_start:.0f}s")
    return rows

def worker(args):
    video, hw, session = args
    try:
        configure(session)
        return process_video(video, hw)
    except Exception as e:
        log(f"[{video}] 失敗: {e}\n{traceback.format_exc()}")
        return []

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hwaccel", default="")
    ap.add_argument("--merge", action="store_true", help="partial を結合して marks_<session>.csv を作る")
    ap.add_argument("--session", default="20260924")
    ap.add_argument("--threads", type=int, default=4, help="ffmpeg スレッド数")
    ap.add_argument("--resume", action="store_true", help="partial_<session>/<video>.csv が既にある曲は飛ばす（10/08・中断再開用）")
    a = ap.parse_args()
    configure(a.session); FFMPEG_THREADS = a.threads
    if a.videos is None:
        g1 = pd.read_csv(G1_CSV)
        vids = sorted(g1[(g1.session == SESSION)].video.unique())
    else:
        vids = a.videos
    if not a.merge:
        from multiprocessing import Pool
        todo = vids
        if a.resume:
            todo = [v for v in vids if not os.path.exists(os.path.join(PARTIAL_DIR, v + ".csv"))]
            log(f"[{SESSION}] 再開: 済 {len(vids)-len(todo)} 曲を飛ばし残り {len(todo)} 曲")
        log(f"開始 曲数{len(todo)} 並列{a.workers} hwaccel={a.hwaccel}")
        if todo:
            with Pool(a.workers) as pool:
                for _ in pool.imap_unordered(worker, [(v, a.hwaccel, a.session) for v in todo]): pass
    parts = []
    for v in vids:
        p = os.path.join(PARTIAL_DIR, v + ".csv")
        if os.path.exists(p): parts.append(pd.read_csv(p))
    if parts:
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(MARKS_CSV, index=False, encoding="utf-8-sig")
        log(f"[{SESSION}] 結合 {len(df)}行 → {os.path.basename(MARKS_CSV)} 抽出率 {df.mark_time_sec.notna().mean():.3f}")

