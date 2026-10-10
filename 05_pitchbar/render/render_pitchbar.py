# -*- coding: utf-8 -*-
r"""録音WAV → カラオケ風「音程バー」画像（PNG 1枚・20秒×N段）。学習済みCRNN（既定＝v2d,v2d2,v2e の確率平均）で
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
# 技法色＝dataviz 基準パレット（カテゴリ 1〜6・固定順）＋形で二重符号化
COL = {"shakuri": "#2a78d6", "kobushi": "#eb6834", "vibrato": "#1baf7a", "fall": "#eda100", "accent": "#e87ba4", "hammering": "#008300"}
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
LANE_P = ymax + 2.5; LANE_T = ymin - 2.5; YLO = ymin - 4; YHI = ymax + 4

# ---------- 7. 描画 ----------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
ROW = a.row; nrow = int(np.ceil(DUR / ROW))
W = 15.0; HEAD = 2.6; RH = 1.0 + 0.1 * (YHI - YLO)
fig = plt.figure(figsize=(W, HEAD + RH * nrow), dpi=a.dpi); fig.patch.set_facecolor("white")
hfrac = HEAD / (HEAD + RH * nrow)
gs = fig.add_gridspec(nrow, 1, left=0.055, right=0.905, top=1 - hfrac, bottom=0.015, hspace=0.32)
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
                ax.plot([pt[i], truth[c][k]], [LANE_P - 0.6, LANE_T + 0.6], color=COL[c], lw=1.1, alpha=0.75, zorder=2)
        for t, p in pred[c]:
            if x0 - 0.5 <= t <= x1 + 0.5: ax.plot(t, LANE_P, MK[c], color=COL[c], ms=8, alpha=0.3 + 0.7 * p, mec="white", mew=0.5, zorder=5)
        for t in truth[c]:
            if x0 - 0.5 <= t <= x1 + 0.5: ax.plot(t, LANE_T, MK[c], mfc="none", mec=COL[c], mew=1.6, ms=8, zorder=5)
    rs = rowstat[r]
    ax.text(1.008, 0.5, f"一致 {rs['hit']}\n誤検知 {rs['fp']}\n見逃し {rs['fn']}" if entry else f"予測 {rs['fp'] + rs['hit']}", transform=ax.transAxes, fontsize=8.5, va="center", ha="left", color="#333", linespacing=1.5)
    ax.text(0.0, 1.01, f"{int(x0 // 60)}:{int(x0 % 60):02d} 〜 {int(x1 // 60)}:{int(x1 % 60):02d}", transform=ax.transAxes, fontsize=7.5, color="#777", va="bottom")
# ヘッダ（曲名・モデル・凡例・F1表）
fig.text(0.055, 0.993, title, fontsize=17, fontweight="bold", va="top", ha="left")
fig.text(0.055, 0.993 - 0.95 * 0.3 / (HEAD + RH * nrow), f"WAV: {os.path.basename(a.wav)}　モデル: {'+'.join(TAGS)}（確率平均・lag +{a.lag}s）　照合 ±{a.tol}s　正解: {'G1 ' + entry['video'] + '（' + str(sum(len(truth[c]) for c in TYPES)) + '件・採点区間内）' if entry else 'なし'}", fontsize=8.5, color="#555", va="top", ha="left")
hdl = [Line2D([], [], marker=MK[c], color=COL[c], ls="", ms=8, label=JA[c]) for c in TYPES]
hdl += [Line2D([], [], marker="o", mfc="none", mec="#333", mew=1.5, ls="", ms=8, label="正解（中抜き・下側）"), Line2D([], [], marker="o", color="#333", ls="", ms=8, label="予測（塗り・上側・濃さ＝確率）"),
        Line2D([], [], color="#333", lw=1.2, label=f"一致（±{a.tol}s）"), Patch(facecolor=C_FP, alpha=0.3, label="誤検知（予測のみ）"), Patch(facecolor=C_FN, alpha=0.3, label="見逃し（正解のみ）"), Patch(facecolor=C_OUT, alpha=0.3, label="採点対象外"), Patch(facecolor=C_BAR, label="歌った音程（半音に吸着）")]
fig.legend(handles=hdl, loc="upper left", bbox_to_anchor=(0.055, 0.993 - 0.95 * 0.9 / (HEAD + RH * nrow)), ncol=5, fontsize=8, frameon=False, handletextpad=0.5, columnspacing=1.4)
tb_h = 0.95 * 1.9 / (HEAD + RH * nrow); axt = fig.add_axes([0.72, 0.993 - tb_h, 0.265, tb_h]); axt.axis("off")
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
outpng = os.path.join(a.out, f"{name}_pitchbar.png"); fig.savefig(outpng, dpi=a.dpi, facecolor="white"); plt.close(fig)

# ---------- 8. 報告 ----------
print(f"PNG: {outpng}  ({nrow}段×{ROW:.0f}s・{DUR:.0f}s)")
if entry:
    print(f"一致 {tot['hit']}／誤検知 {tot['fp']}／見逃し {tot['fn']}（±{a.tol}s・採点区間 {lo:.1f}〜{hi:.1f}s）")
    for c in TYPES: f, H, NP, NT = f1(c); print(f"  {JA[c]:<6} 正解{NT:>3} 予測{NP:>3} 一致{H:>3} F1={f:.3f}")
else:
    print("予測件数:", {JA[c]: len(pred[c]) for c in TYPES})
