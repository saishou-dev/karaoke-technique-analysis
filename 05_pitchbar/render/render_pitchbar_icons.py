# -*- coding: utf-8 -*-
r"""【マーク図形版・2026-10-08】render_pitchbar.py の予測／正解マーカー（▲◆●▼■✚）を、DAM画面の技法マークの「形と色を模したベクター図形」
   （matplotlib の Path／ベジェ・手本＝marks_20260928/scripts/templates/*_color.png を目視）に置き換えたもの。
   予測＝上側レーンに塗り（帯＝技法色・芯＝白、濃さ＝確率）／正解＝下側レーンに同じ形の縁取りのみ（中抜き・dpi220 で 3.5px）。
   一致の実線・誤検知の赤帯・見逃しの紫帯・音程バーは原本と同じ。マークの高さ＝1段の約 --icon_frac（既定 0.2）。
   技法色は DAM の色に合わせて変更（しゃくり赤・こぶし青・ビブラート緑・フォール マゼンタ・アクセント橙・ハンマリング シアン）。
   追加引数: --icon_frac 0.2  --icon_out <dir>（凡例用の透過PNG・分割スクリプト用）  --shapes_out <png>（6技法の図形見本）
   以下は原本（render_pitchbar.py）の説明：
   録音WAV → カラオケ風「音程バー」画像（PNG 1枚・20秒×N段）。学習済みCRNN（既定＝v2d,v2d2,v2e の確率平均）で
   加点イベントを予測し、index.json に正解（G1）があれば重ねて 一致／誤検知／見逃し を可視化する（2026-10-03）。
   見た目＝横軸 時間・縦軸 半音（音名）。歌った音程＝太い横バー（持続音ごと・半音に吸着）。
          予測＝段の上側レーンに技法別の色の塗りマーク（濃さ＝確率）／正解＝段の下側レーンに同色の中抜きマーク。
          一致（±tol s）＝実線でつなぐ／予測だけ＝誤検知（赤の薄い帯）／正解だけ＝見逃し（紫の薄い帯）。
   使い方: python render_pitchbar.py --wav "G:\研究アーカイブ\録音\songs_20260924_raw\T21_20260924_121616.wav" --out G:\研究アーカイブ\成果物\動画解析\pitchbar\
          省略可: --video <index.json の video 名>  --tags v2d,v2d2,v2e  --lag 0.3  --tol 1.0  --row 20  --dev 20260911
   前提: feats 生成（extract_feats.py と同じ手順）・pick（rethreshold_raw.py と同じ：dev で技法別に しきい値・最短間隔 を決める）。
         しきい値は out/pitchbar_thr_<tags>.json にキャッシュ（初回のみ dev 30曲の推論で数分）。新しい学習はしない。"""
import os, sys, json, argparse, warnings, time
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"
SR = 22050; HOP = 220; NMEL = 80; FR = 100
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
JA = {"shakuri": "しゃくり", "kobushi": "こぶし", "vibrato": "ビブラート", "fall": "フォール", "accent": "アクセント", "hammering": "ハンマリング"}
# 技法色＝DAM画面の技法マークの色に合わせる（手本テンプレを目視）。形は MK の代わりに SHAPE（ベクター図形）で表す
COL = {"shakuri": "#d9262c", "kobushi": "#2a5bd7", "vibrato": "#1fb83a", "fall": "#c13ab6", "accent": "#ef8a1c", "hammering": "#15b1c9"}
MK = {"shakuri": "^", "kobushi": "D", "vibrato": "o", "fall": "v", "accent": "s", "hammering": "P"}
C_FP = "#e34948"; C_FN = "#4a3aa7"; C_BAR = "#44506b"; C_OUT = "#9a9a9a"
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

ap = argparse.ArgumentParser()
ap.add_argument("--wav", required=True); ap.add_argument("--video", default="")
ap.add_argument("--out", default=f"{os.path.dirname(ROOT)}/pitchbar")
ap.add_argument("--tags", default="v2d,v2d2,v2e"); ap.add_argument("--lag", type=float, default=0.3)
ap.add_argument("--tol", type=float, default=1.0); ap.add_argument("--row", type=float, default=20.0)
ap.add_argument("--dev", default="20260911"); ap.add_argument("--hid", type=int, default=128)
ap.add_argument("--dpi", type=int, default=110); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--icon_frac", type=float, default=0.2); ap.add_argument("--icon_out", default=""); ap.add_argument("--shapes_out", default="")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
import torch, torch.nn as nn, torch.nn.functional as Fn
torch.set_num_threads(a.threads)
index = json.load(open(f"{ROOT}/index.json", encoding="utf-8"))
TAGS = a.tags.split(",")

# ---------- 1. index との対応（同名WAVがあれば正解を重ねる） ----------
def norm_path(p): return os.path.normcase(os.path.normpath(p)).replace("\\", "/")
entry = None
if a.video:
    entry = next((d for d in index if d["video"] == a.video), None)
    if entry is None: sys.exit(f"index.json に video={a.video} がありません")
else:
    base = os.path.basename(a.wav)
    hits = [d for d in index if os.path.basename(d["wav"]) == base and norm_path(d["wav"]) == norm_path(a.wav)] or [d for d in index if os.path.basename(d["wav"]) == base]
    entry = hits[0] if hits else None
stem = os.path.splitext(os.path.basename(a.wav))[0]
name = entry["video"] if entry else stem
title = name.rsplit("_", 2)[0] if entry else stem
print(f"WAV={a.wav}\nindex={'あり: ' + entry['video'] if entry else 'なし（予測のみ）'}", flush=True)

# ---------- 2. 特徴量（extract_feats.py と同一手順・無ければWAVから作る） ----------
def build_feats(wav, out):
    import librosa, soundfile as sf
    y, sr = sf.read(wav, dtype="float32", always_2d=True); y = y.mean(1)
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=HOP, n_mels=NMEL, fmin=50, fmax=8000, power=2.0)
    logmel = np.log(S + 1e-6).astype(np.float32)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]; rms = np.log(rms + 1e-6).astype(np.float32)
    T = logmel.shape[1]; t = np.arange(T) * HOP / SR
    f0, vf, vp = librosa.pyin(y, fmin=70, fmax=800, sr=SR, frame_length=2048, hop_length=HOP)
    f0 = np.nan_to_num(f0, nan=0.0); conf = np.nan_to_num(vp, nan=0.0); pt = np.arange(len(f0)) * HOP / SR
    st = np.where(f0 > 40, 69 + 12 * np.log2(np.maximum(f0, 1) / 440.0), np.nan)
    idx = np.clip(np.searchsorted(pt, t), 0, len(pt) - 1)
    st_g = st[idx]; voiced = (~np.isnan(st_g)).astype(np.float32); conf_g = conf[idx].astype(np.float32)
    np.savez_compressed(out, logmel=logmel.astype(np.float16), rms=rms, st=np.nan_to_num(st_g, nan=0.0).astype(np.float32), voiced=voiced, conf=conf_g, t0=0.0, hop=HOP / SR, src="pyin_now")
fpath = f"{ROOT}/feats/{name}.npz"
if not os.path.exists(fpath):
    t0 = time.time(); print("特徴量を生成中（pYIN・数分）…", flush=True); build_feats(a.wav, fpath); print(f"  {time.time() - t0:.0f}s", flush=True)
Z = np.load(fpath); DUR = Z["st"].shape[0] / FR

def feats_of(z):
    lm = z["logmel"].astype(np.float32); T = lm.shape[1]; st = z["st"]; v = z["voiced"]
    stn = np.where(v > 0, (st - 60.0) / 12.0, 0.0).astype(np.float32)
    dst = np.zeros(T, np.float32); dst[1:] = np.diff(st); dst[1:] *= (v[:-1] > 0); dst = np.where(v > 0, np.clip(dst, -3, 3), 0.0).astype(np.float32)
    return np.concatenate([lm, z["rms"][None, :], stn[None, :], v[None, :], z["conf"][None, :], dst[None, :]], 0).astype(np.float32)

# ---------- 3. 学習済みモデル（rethreshold_raw.py と同じ構造・確率平均） ----------
norm0 = np.load(f"{ROOT}/out/{TAGS[0]}/norm.npy"); CIN = norm0.shape[1]
class Net(nn.Module):
    def __init__(s, cin=CIN, h=a.hid, nout=len(TYPES)):
        super().__init__()
        s.c1 = nn.Sequential(nn.Conv1d(cin, h, 5, padding=2), nn.BatchNorm1d(h), nn.ReLU(), nn.Conv1d(h, h, 5, padding=2, stride=2), nn.BatchNorm1d(h), nn.ReLU(), nn.Conv1d(h, h, 5, padding=4, dilation=2), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.15))
        s.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15); s.head = nn.Linear(2 * h, nout)
    def forward(s, x):
        T = x.shape[-1]; h = s.c1(x); h, _ = s.gru(h.transpose(1, 2)); o = s.head(h).transpose(1, 2)
        return Fn.interpolate(o, size=T, mode="linear", align_corners=False)
nets = []
for tg in TAGS:
    n_ = Net(); n_.load_state_dict(torch.load(f"{ROOT}/out/{tg}/model.pt")); n_.eval(); nm = np.load(f"{ROOT}/out/{tg}/norm.npy")
    nets.append((n_, torch.tensor(nm[0][:, None]), torch.tensor(nm[1][:, None])))
def probs_of(z):
    x = torch.tensor(feats_of(z)[None])
    with torch.no_grad(): return np.mean([torch.sigmoid(n_((x - m_) / s_))[0].numpy() for n_, m_, s_ in nets], 0)
def pick(prob, thr, md):
    w = int(md * FR); n = len(prob)
    return [(i / FR + a.lag, float(prob[i])) for i in range(1, n - 1) if prob[i] >= thr and prob[i] >= prob[max(0, i - w):min(n, i + w + 1)].max()]
def match(pred, truth, tol):
    """貪欲照合（rethreshold_raw.py と同じ規則）→ 対 (p_idx, t_idx) の一覧"""
    used = set(); pairs = []
    for pi in sorted(range(len(pred)), key=lambda i: pred[i]):
        best = None
        for k, t in enumerate(truth):
            if k in used or abs(t - pred[pi]) > tol: continue
            if best is None or abs(t - pred[pi]) < abs(truth[best] - pred[pi]): best = k
        if best is not None: used.add(best); pairs.append((pi, best))
    return pairs

# ---------- 4. しきい値（dev で ±1s F1 最大・技法別・キャッシュ） ----------
thr_file = f"{ROOT}/out/pitchbar_thr_{'+'.join(TAGS)}_lag{a.lag}.json"
if os.path.exists(thr_file):
    best = {c: tuple(v) for c, v in json.load(open(thr_file, encoding="utf-8")).items()}
else:
    print(f"しきい値をdev（{a.dev}）で決定中（初回のみ）…", flush=True); t0 = time.time()
    dev = [d for d in index if d["session"] == a.dev and os.path.exists(f"{ROOT}/feats/{d['video']}.npz")]
    P = {d["video"]: probs_of(np.load(f"{ROOT}/feats/{d['video']}.npz")) for d in dev}
    def f1_dev(c, th, md):
        k = TYPES.index(c); H = NP = NT = 0
        for d in dev:
            lo, hi = d["span"]; tr = [t for cc, t in d["events"] if cc == c and lo <= t <= hi]; pr = [t for t, _ in pick(P[d["video"]][k], th, md) if lo <= t <= hi]
            H += len(match(pr, tr, 1.0)); NP += len(pr); NT += len(tr)
        p = H / NP if NP else 0; r = H / NT if NT else 0; return 2 * p * r / (p + r) if p + r else 0
    best = {}
    for c in TYPES:
        cand = max(((f1_dev(c, th, md), th, md) for th in np.arange(0.2, 0.96, 0.05) for md in (0.3, 0.5, 0.8)), key=lambda x: x[0]); best[c] = (round(float(cand[1]), 2), cand[2])
    json.dump(best, open(thr_file, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(f"  {time.time() - t0:.0f}s → {thr_file}", flush=True)
print("しきい値(thr, 最短間隔s)=", best, flush=True)

# ---------- 5. 予測・正解・照合 ----------
PR = probs_of(Z)
span = entry["span"] if entry else [0.0, DUR]
lo, hi = span
pred = {c: [(t, p) for t, p in pick(PR[TYPES.index(c)], *best[c]) if lo <= t <= hi] for c in TYPES}
truth = {c: ([t for cc, t in entry["events"] if cc == c and lo <= t <= hi] if entry else []) for c in TYPES}
res = {}
for c in TYPES:
    pt = [t for t, _ in pred[c]]; pairs = match(pt, truth[c], a.tol) if entry else []
    mp = {i for i, _ in pairs}; mt = {k for _, k in pairs}
    res[c] = dict(pairs=pairs, fp=[i for i in range(len(pt)) if i not in mp], fn=[k for k in range(len(truth[c])) if k not in mt])
def f1(c):
    H = len(res[c]["pairs"]); NP = len(pred[c]); NT = len(truth[c]); p = H / NP if NP else 0; r = H / NT if NT else 0
    return (2 * p * r / (p + r) if p + r else 0), H, NP, NT
tot = dict(hit=sum(len(res[c]["pairs"]) for c in TYPES), fp=sum(len(res[c]["fp"]) for c in TYPES), fn=sum(len(res[c]["fn"]) for c in TYPES))

# ---------- 6. 持続音（song_metrics.py と同じ規則：有声ラン≥0.15s・跳び>0.8半音で分割）→ 半音に吸着 ----------
from scipy.ndimage import median_filter
st = Z["st"].astype(float); v = Z["voiced"] > 0; st[~v] = np.nan; n = len(st)
idx = np.where(v)[0]
if len(idx) > 10: st[idx] = median_filter(st[idx], size=5, mode="nearest")
segs = []; i = 0
while i < n:
    if not v[i]: i += 1; continue
    j = i
    while j + 1 < n and v[j + 1] and abs(np.nanmedian(st[max(i, j - 4):j + 1]) - np.nanmedian(st[j + 1:j + 6])) <= 0.8: j += 1
    if (j - i + 1) / FR >= 0.15: segs.append((i / FR, (j + 1) / FR, int(round(float(np.nanmedian(st[i:j + 1]))))))
    i = j + 1
notes = np.array([s[2] for s in segs]) if segs else np.array([60])
ymin, ymax = int(np.percentile(notes, 1)) - 1, int(np.percentile(notes, 99)) + 1
if ymax - ymin < 12: ymin, ymax = (ymin + ymax) // 2 - 6, (ymin + ymax) // 2 + 6
# レーンの余白＝マーク（高さ＝段の icon_frac・帯のはみ出し 13%）が グリッド線(±1.2)＋余白0.5 の外側に収まる幅：M = (2.2 + 1.13 f range)/(1 - 2.26 f)
RNG = ymax - ymin; M = (2.2 + 1.13 * a.icon_frac * RNG) / (1 - 2.26 * a.icon_frac); M = float(np.ceil(M * 10) / 10)
ICON_SEMI = a.icon_frac * (RNG + 2 * M)                       # マークの高さ（半音単位）
LANE_P = ymax + 1.7 + 0.565 * ICON_SEMI; LANE_T = ymin - 1.7 - 0.565 * ICON_SEMI; YLO = ymin - M; YHI = ymax + M

# ---------- 7. 描画 ----------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, PathPatch
from matplotlib.path import Path
from PIL import Image
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
ROW = a.row; nrow = int(np.ceil(DUR / ROW))
W = 15.0; HEAD = 3.0; RH = 1.0 + 0.1 * (YHI - YLO)
FIG_H = HEAD + RH * nrow; hfrac = HEAD / FIG_H; HSP = 0.32
AX_H_IN = (RH * nrow - 0.015 * FIG_H) / (nrow + HSP * (nrow - 1))    # 1段の axes 高さ [inch]
AX_H_PX = AX_H_IN * a.dpi; ICON_PX = int(round(a.icon_frac * AX_H_PX))  # マークの高さ [px]＝段の icon_frac
AX_W_IN = W * (0.905 - 0.055); PX_PER_SEC = AX_W_IN * a.dpi / ROW; PX_PER_SEMI = AX_H_PX / (YHI - YLO)
ASPECT = 1.0                                                         # マークの幅／高さ
OUTLINE_PX = 3.5                                                     # 正解（中抜き）の縁取り線幅 [px]（dpi220 で 3〜4px）
BAND_R = 0.13                                                        # 帯の半幅（単位座標・高さ2に対して）＝帯幅 13%

# --- 技法マークの形（単位座標 x,y∈[-1,1]）：手本＝marks_20260928/scripts/templates/*_color.png（DAM画面の技法マーク）を目視して決めた
#     しゃくり＝赤のS字スライド（ʃ）／こぶし＝青の渦巻き（C）／ビブラート＝緑の波（M）／フォール＝マゼンタの下降カーブ／
#     アクセント＝橙のV＋右へ水平（V‾・芯は黄）／ハンマリング＝シアンの段差スライド（_/‾）。帯＝技法色・芯＝白（アクセントは黄）
def bezier(P, n=60):
    P = np.asarray(P, float); t = np.linspace(0, 1, n)[:, None]
    if len(P) == 3: return (1 - t) ** 2 * P[0] + 2 * (1 - t) * t * P[1] + t ** 2 * P[2]
    return (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]
def rounded_polyline(V, rc=0.22, n=24):
    """折れ線 V の角を半径相当 rc の2次ベジェで丸めて密にサンプルする"""
    V = np.asarray(V, float); pts = [V[0]]
    for i in range(1, len(V) - 1):
        a_, b_, c_ = V[i - 1], V[i], V[i + 1]
        da = (a_ - b_) / np.linalg.norm(a_ - b_); dc = (c_ - b_) / np.linalg.norm(c_ - b_)
        p = b_ + da * min(rc, np.linalg.norm(a_ - b_) / 2); q = b_ + dc * min(rc, np.linalg.norm(c_ - b_) / 2)
        seg = np.linspace(pts[-1], p, n); pts.extend(seg[1:]); pts.extend(bezier([p, b_, q], n)[1:])
    seg = np.linspace(pts[-1], V[-1], n); pts.extend(seg[1:]); return np.array(pts)
def spiral(th0=75, th1=-230, r0=0.88, r1=0.55, n=120):
    th = np.radians(np.linspace(th0, th1, n)); r = np.linspace(r0, r1, n); return np.c_[r * np.cos(th), r * np.sin(th)]
SHAPE = {
    "shakuri":   lambda: bezier([(-0.72, -0.95), (0.95, -0.95), (-0.95, 0.95), (0.72, 0.95)], 80),
    "kobushi":   lambda: spiral(),
    "vibrato":   lambda: rounded_polyline([(-1.0, -0.85), (-0.5, 0.85), (0.0, -0.35), (0.5, 0.85), (1.0, -0.85)], 0.4),
    "fall":      lambda: bezier([(-0.9, 0.72), (0.6, 0.72), (0.5, -0.95)], 80),
    "accent":    lambda: rounded_polyline([(-0.95, 0.75), (-0.35, -0.92), (0.05, 0.55), (0.98, 0.55)], 0.36),
    "hammering": lambda: rounded_polyline([(-1.0, -0.6), (-0.35, -0.6), (0.35, 0.6), (1.0, 0.6)], 0.28),
}
CORE = {c: "white" for c in TYPES}; CORE["accent"] = "#ffe14d"
def band_polygon(pts, r):
    """中心線 pts（N×2・密）を半幅 r でオフセットした帯のポリゴン（丸いキャップ）→ 中抜き描画のための閉路。
    角の内側でオフセット曲線が逆走する点（自己交差のループ）は落とす"""
    d = np.gradient(pts, axis=0); d /= np.linalg.norm(d, axis=1)[:, None] + 1e-9; nrm = np.c_[-d[:, 1], d[:, 0]]
    def remove_loops(q, w=40):
        """オフセット折れ線の局所的な自己交差（角の内側のループ）を、交点で切り詰めて除く"""
        q = list(map(tuple, q)); i = 0
        while i < len(q) - 3:
            p1, p2 = np.array(q[i]), np.array(q[i + 1]); hit = None
            for j in range(i + 2, min(len(q) - 1, i + w)):
                p3, p4 = np.array(q[j]), np.array(q[j + 1]); r1 = p2 - p1; r2 = p4 - p3; den = r1[0] * r2[1] - r1[1] * r2[0]
                if abs(den) < 1e-12: continue
                t = ((p3 - p1)[0] * r2[1] - (p3 - p1)[1] * r2[0]) / den; u = ((p3 - p1)[0] * r1[1] - (p3 - p1)[1] * r1[0]) / den
                if 0 < t < 1 and 0 < u < 1: hit = (j, tuple(p1 + t * r1)); break
            if hit: q[i + 1:hit[0] + 1] = [hit[1]]
            i += 1
        return np.array(q)
    def side(sign):
        q = pts + sign * r * nrm; dq = np.gradient(q, axis=0); ok = (dq * d).sum(1) > 0; ok[0] = ok[-1] = True
        return remove_loops(q[ok])
    left = side(+1); right = side(-1)
    def cap(center, n_from, k=14):   # n_from から時計回りに半周（進行方向側を回る）
        a0 = np.arctan2(n_from[1], n_from[0]); ang = np.linspace(a0, a0 - np.pi, k); return center + r * np.c_[np.cos(ang), np.sin(ang)]
    end_cap = cap(pts[-1], nrm[-1]); start_cap = cap(pts[0], -nrm[0])
    return np.vstack([left, end_cap, right[::-1], start_cap])
MARK = {c: (SHAPE[c](), band_polygon(SHAPE[c](), BAND_R)) for c in TYPES}
def draw_mark(ax, c, cx, cy, sx, sy, h_px, alpha=1.0, hollow=False, zorder=5, transform=None):
    """技法 c のマークを中心 (cx,cy)・単位→座標の倍率 (sx,sy)・高さ h_px で描く。hollow=正解（縁取りのみ）／それ以外＝予測（塗り＋芯・alpha）"""
    tr = transform or ax.transData; line, poly = MARK[c]
    P = np.c_[cx + poly[:, 0] * sx, cy + poly[:, 1] * sy]; path = Path(P, closed=True)
    if hollow:
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=COL[c], lw=OUTLINE_PX * 72 / a.dpi, joinstyle="round", transform=tr, zorder=zorder))
    else:
        ax.add_patch(PathPatch(path, facecolor=COL[c], edgecolor="none", alpha=alpha, transform=tr, zorder=zorder))
        core_pt = 0.42 * BAND_R * h_px * 72 / a.dpi                 # 芯の太さ＝帯幅の 42%
        ax.add_line(Line2D(cx + line[:, 0] * sx, cy + line[:, 1] * sy, color=CORE[c], lw=core_pt, alpha=alpha, solid_capstyle="round", solid_joinstyle="round", transform=tr, zorder=zorder + 0.1))
SY = ICON_SEMI / 2; SX = SY * PX_PER_SEMI / PX_PER_SEC * ASPECT        # データ座標での倍率（半音／秒）
# 分割スクリプトの凡例用：各マーク（予測様式）を透過PNGに書き出す
if a.icon_out:
    os.makedirs(a.icon_out, exist_ok=True)
    for c in TYPES:
        hpx = 120; f_ = plt.figure(figsize=(ASPECT * 1.2 * hpx / 100, 1.2 * hpx / 100), dpi=100); ax_ = f_.add_axes([0, 0, 1, 1]); ax_.axis("off")
        ax_.set_xlim(-1.2 * ASPECT, 1.2 * ASPECT); ax_.set_ylim(-1.2, 1.2); draw_mark(ax_, c, 0, 0, ASPECT, 1.0, hpx)
        f_.savefig(os.path.join(a.icon_out, f"{c}.png"), dpi=100, transparent=True); plt.close(f_)
print(f"マーク: 高さ {ICON_PX}px（段の {a.icon_frac:.2f}・{ICON_SEMI:.1f}半音）・幅 {ICON_PX * ASPECT / PX_PER_SEC:.2f}s・帯幅 {BAND_R * ICON_PX:.0f}px・縁取り {OUTLINE_PX}px", flush=True)

fig = plt.figure(figsize=(W, FIG_H), dpi=a.dpi); fig.patch.set_facecolor("white")
gs = fig.add_gridspec(nrow, 1, left=0.055, right=0.905, top=1 - hfrac, bottom=0.015, hspace=HSP)
rowstat = [dict(hit=0, fp=0, fn=0) for _ in range(nrow)]
def rowof(t): return min(nrow - 1, max(0, int(t // ROW)))
for c in TYPES:
    pt = [t for t, _ in pred[c]]
    for i, k in res[c]["pairs"]: rowstat[rowof(pt[i])]["hit"] += 1
    for i in res[c]["fp"]: rowstat[rowof(pt[i])]["fp"] += 1
    for k in res[c]["fn"]: rowstat[rowof(truth[c][k])]["fn"] += 1
for r in range(nrow):
    ax = fig.add_subplot(gs[r, 0]); x0, x1 = r * ROW, (r + 1) * ROW
    ax.set_xlim(x0, x1); ax.set_ylim(YLO, YHI); ax.set_facecolor("#fafafa")
    for sp in ax.spines.values(): sp.set_color("#cccccc")
    # 採点対象外（span の外）
    if x0 < lo: ax.axvspan(x0, min(lo, x1), color=C_OUT, alpha=0.13, lw=0)
    if x1 > hi: ax.axvspan(max(hi, x0), x1, color=C_OUT, alpha=0.13, lw=0)
    # 半音グリッド・音名
    for s in range(ymin, ymax + 1): ax.axhline(s, color="#e4e4e4" if s % 12 else "#bdbdbd", lw=0.5, zorder=0)
    ticks = list(range(ymin, ymax + 1)); dense = (ymax - ymin) > 18
    ax.set_yticks(ticks); ax.set_yticklabels([(f"{NOTE[s % 12]}{s // 12 - 1}" if (s % 12 in (0, 4, 7) if dense else "#" not in NOTE[s % 12]) else "") for s in ticks], fontsize=6.5, color="#555")
    ax.tick_params(axis="y", length=0); ax.set_xticks(np.arange(x0, x1 + 0.1, 2.0)); ax.tick_params(axis="x", labelsize=7, colors="#555")
    ax.axhline(ymax + 1.2, color="#dddddd", lw=0.6); ax.axhline(ymin - 1.2, color="#dddddd", lw=0.6)
    ax.text(x0 + 0.08, LANE_P, "予測", fontsize=7, color="#777", va="center", ha="left", zorder=1)
    if entry: ax.text(x0 + 0.08, LANE_T, "正解", fontsize=7, color="#777", va="center", ha="left", zorder=1)
    # 音程バー
    for s0, s1, nt in segs:
        if s1 < x0 or s0 > x1: continue
        ax.add_patch(Rectangle((s0, nt - 0.42), s1 - s0, 0.84, facecolor=C_BAR, edgecolor="white", lw=0.4, zorder=3))
    # 帯（誤検知・見逃し）→ 線（一致）→ マーク
    for c in TYPES:
        pt = [t for t, _ in pred[c]]
        for i in res[c]["fp"]:
            if x0 - 1 <= pt[i] <= x1 + 1: ax.axvspan(pt[i] - 0.3, pt[i] + 0.3, color=C_FP, alpha=0.14, lw=0, zorder=1)
        for k in res[c]["fn"]:
            if x0 - 1 <= truth[c][k] <= x1 + 1: ax.axvspan(truth[c][k] - 0.3, truth[c][k] + 0.3, color=C_FN, alpha=0.14, lw=0, zorder=1)
        for i, k in res[c]["pairs"]:
            if min(pt[i], truth[c][k]) <= x1 and max(pt[i], truth[c][k]) >= x0:
                ax.plot([pt[i], truth[c][k]], [LANE_P - ICON_SEMI / 2, LANE_T + ICON_SEMI / 2], color=COL[c], lw=1.1, alpha=0.75, zorder=2)
        for t, p in pred[c]:
            if x0 - 0.5 <= t <= x1 + 0.5: draw_mark(ax, c, t, LANE_P, SX, SY, ICON_PX, alpha=0.3 + 0.7 * p)
        for t in truth[c]:
            if x0 - 0.5 <= t <= x1 + 0.5: draw_mark(ax, c, t, LANE_T, SX, SY, ICON_PX, hollow=True)
    rs = rowstat[r]
    ax.text(1.008, 0.5, f"一致 {rs['hit']}\n誤検知 {rs['fp']}\n見逃し {rs['fn']}" if entry else f"予測 {rs['fp'] + rs['hit']}", transform=ax.transAxes, fontsize=8.5, va="center", ha="left", color="#333", linespacing=1.5)
    ax.text(0.0, 1.01, f"{int(x0 // 60)}:{int(x0 % 60):02d} 〜 {int(x1 // 60)}:{int(x1 % 60):02d}", transform=ax.transAxes, fontsize=7.5, color="#777", va="bottom")
# ヘッダ（曲名・モデル・凡例（図形＋技法名）・F1表）
def Y(inch): return 1 - inch / FIG_H                                 # 上端から inch 下の figure 座標
fig.text(0.055, Y(0.15), title, fontsize=17, fontweight="bold", va="top", ha="left")
fig.text(0.055, Y(0.45), f"WAV: {os.path.basename(a.wav)}　モデル: {'+'.join(TAGS)}（確率平均・lag +{a.lag}s）　照合 ±{a.tol}s　正解: {'G1 ' + entry['video'] + '（' + str(sum(len(truth[c]) for c in TYPES)) + '件・採点区間内）' if entry else 'なし'}", fontsize=8.5, color="#555", va="top", ha="left")
LEG_H_IN = 0.42; LEG_W_IN = 0.65 * W; LEG_MARK_IN = 0.26                # 凡例帯の高さ／幅・凡例マークの高さ [inch]
axl = fig.add_axes([0.055, Y(1.15), 0.65, LEG_H_IN / FIG_H]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
lsy = LEG_MARK_IN / 2 / LEG_H_IN; lsx = LEG_MARK_IN / 2 * ASPECT / LEG_W_IN; xx = lsx
for c in TYPES:
    draw_mark(axl, c, xx, 0.5, lsx, lsy, LEG_MARK_IN * a.dpi); draw_mark(axl, c, xx + 2.3 * lsx, 0.5, lsx, lsy, LEG_MARK_IN * a.dpi, hollow=True)
    axl.text(xx + 3.6 * lsx, 0.5, JA[c], fontsize=9, color=COL[c], va="center", ha="left")
    xx += 3.6 * lsx + 0.012 * len(JA[c]) + 0.03
hdl = [Patch(facecolor="none", edgecolor="#333", lw=1.2, label="正解（同じ形の縁取りのみ・下側）"), Patch(facecolor="#999", label="予測（塗り・上側・濃さ＝確率）"),
       Line2D([], [], color="#333", lw=1.2, label=f"一致（±{a.tol}s）"), Patch(facecolor=C_FP, alpha=0.3, label="誤検知（予測のみ）"), Patch(facecolor=C_FN, alpha=0.3, label="見逃し（正解のみ）"), Patch(facecolor=C_OUT, alpha=0.3, label="採点対象外"), Patch(facecolor=C_BAR, label="歌った音程（半音に吸着）")]
fig.legend(handles=hdl, loc="upper left", bbox_to_anchor=(0.055, Y(1.25)), ncol=4, fontsize=8, frameon=False, handletextpad=0.5, columnspacing=1.4)
tb_h = 1.9 / FIG_H; axt = fig.add_axes([0.72, Y(0.15) - tb_h, 0.265, tb_h]); axt.axis("off")
if entry:
    cells = [[JA[c], f1(c)[3], f1(c)[2], f1(c)[1], f"{f1(c)[0]:.2f}"] for c in TYPES]
    H = tot["hit"]; NP = H + tot["fp"]; NT = H + tot["fn"]; p = H / NP if NP else 0; rr = H / NT if NT else 0
    cells.append(["全体", NT, NP, H, f"{(2 * p * rr / (p + rr) if p + rr else 0):.2f}"])
    tbl = axt.table(cellText=cells, colLabels=["技法", "正解", "予測", "一致", f"F1(±{a.tol:g}s)"], loc="upper center", cellLoc="center", colWidths=[0.34, 0.15, 0.15, 0.15, 0.21])
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.05)
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd"); cell.set_linewidth(0.6)
        if ri == 0 or ri == len(cells): cell.set_facecolor("#f0f0f0"); cell.set_text_props(fontweight="bold")
        if ri > 0 and ci == 0 and ri <= len(TYPES): cell.set_text_props(color=COL[TYPES[ri - 1]])
    axt.text(0.5, -0.02, f"一致 {H}／誤検知 {tot['fp']}／見逃し {tot['fn']}", transform=axt.transAxes, fontsize=8.5, ha="center", va="top", color="#333")
else:
    axt.text(1.0, 1.0, "正解なし（index.json に未登録）" + chr(10) + "予測件数: " + "／".join(f"{JA[c]} {len(pred[c])}" for c in TYPES), transform=axt.transAxes, fontsize=8.5, va="top", ha="right", bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#dddddd"))
outpng = os.path.join(a.out, f"{name}_pitchbar_icons.png"); fig.savefig(outpng, dpi=a.dpi, facecolor="white"); plt.close(fig)

# --- 形の見本（確認用）：技法ごとに 手本テンプレ（DAM画面の切り出し）／予測様式（確率1.0・0.5）／正解様式（縁取り） ---
if a.shapes_out:
    TPL_DIRS = [f"{os.path.dirname(ROOT)}/pitchbar/lulu_20260924_large_icons/templates", f"{os.path.dirname(ROOT)}/marks_20260928/scripts/templates"]
    fs = plt.figure(figsize=(13, 6.2), dpi=150); fs.patch.set_facecolor("white")
    fs.text(0.5, 0.965, "技法マークの図形見本（上＝手本テンプレ・DAM画面の切り出し／中＝予測（左 確率1.0・右 0.5）／下＝正解（縁取りのみ））", ha="center", va="top", fontsize=11)
    for j, c in enumerate(TYPES):
        ax_ = fs.add_axes([0.03 + j * 0.16, 0.66, 0.14, 0.24]); ax_.axis("off")
        for d_ in TPL_DIRS:
            pth = os.path.join(d_, f"{c}_color.png")
            if os.path.exists(pth): ax_.imshow(np.asarray(Image.open(pth).convert("RGB"))); break
        ax_.set_title(JA[c], fontsize=11, color=COL[c], fontweight="bold")
        ax_ = fs.add_axes([0.03 + j * 0.16, 0.36, 0.14, 0.26]); ax_.axis("off"); ax_.set_xlim(-2.6, 2.6); ax_.set_ylim(-1.3, 1.3); ax_.set_aspect("equal")
        draw_mark(ax_, c, -1.3, 0, ASPECT, 1.0, 105, alpha=1.0); draw_mark(ax_, c, 1.3, 0, ASPECT, 1.0, 105, alpha=0.65)
        ax_ = fs.add_axes([0.03 + j * 0.16, 0.06, 0.14, 0.26]); ax_.axis("off"); ax_.set_xlim(-2.6, 2.6); ax_.set_ylim(-1.3, 1.3); ax_.set_aspect("equal")
        draw_mark(ax_, c, 0, 0, ASPECT, 1.0, 105, hollow=True)
    fs.savefig(a.shapes_out, dpi=150, facecolor="white"); plt.close(fs); print(f"形の見本: {a.shapes_out}")

# ---------- 8. 報告 ----------
print(f"PNG: {outpng}  ({nrow}段×{ROW:.0f}s・{DUR:.0f}s・アイコン高さ {ICON_PX}px＝段の {a.icon_frac:.2f}・レーン余白 {M} 半音)")
if entry:
    print(f"一致 {tot['hit']}／誤検知 {tot['fp']}／見逃し {tot['fn']}（±{a.tol}s・採点区間 {lo:.1f}〜{hi:.1f}s）")
    for c in TYPES: f, H, NP, NT = f1(c); print(f"  {JA[c]:<6} 正解{NT:>3} 予測{NP:>3} 一致{H:>3} F1={f:.3f}")
else:
    print("予測件数:", {JA[c]: len(pred[c]) for c in TYPES})
