# -*- coding: utf-8 -*-
r"""ビブラートの深さ・速さ・持続の仮検証（2026-10-08・D2 の小実験・新しい学習はしない）。

目的: DAM の加点イベント（正解・counter=vibrato）の周辺と、学習済み検出器がビブラートと誤検知した箇所（FP）の周辺で、
      歌声のピッチ曲線（pYIN f0・100fps）から 深さ[cent]・速さ[Hz]・持続[s] を同じ手順で測り、分布が分かれるかを見る。

定義（卒論用）:
  窓      = イベント時刻 t0 の [t0−1.5s, t0+0.3s]（DAM の加点は歌唱より 1〜2s 遅れるため後ろ寄りに取る・G3v2 報告参照）。
  揺れ成分 = 窓内の有声区間ごとに f0 をセント（MIDI 半音×100）に変換し、0.3s の移動中央値（＝旋律の土台）を引いた残差。
  極値列  = 残差の山・谷を交互に取り、隣り合う山谷の差（p-p, cent）と時間差（半周期, s）を得る。
  周期的な揺れ = 連続する山谷対が「差 ≥ 15 cent かつ 半周期 ∈ [1/18, 1/6] s（3〜9 Hz）」を満たす最長の連鎖で、対が 3 つ以上（1.5 周期以上）。
  深さ    = 連鎖内の山谷差（p-p cent）の中央値。連鎖が無ければ窓内の全山谷差の中央値（無ければ 0）。
  速さ    = 連鎖区間の残差の FFT ピーク周波数（3〜9 Hz に限定・Hann 窓・4096 点ゼロ埋め）。連鎖が無ければ最長有声区間で同様。
  持続    = 連鎖の最初の極値から最後の極値までの時間（s）。連鎖が無ければ 0。
  補助    = 周期性 = 3〜9Hz 帯の残差パワー ／ 0.5〜20Hz のパワー。有声率 = 窓内の有声フレーム率。

FP の作り方: render_pitchbar.py / build_ledgers_20261008.py と同じ学習済み CRNN アンサンブル（v2d+v2d2+v2e・確率平均・lag 0.3）
      ＋同じしきい値キャッシュ（out/pitchbar_thr_*.json）で test（9/24・28曲）を予測し、同じ貪欲 1 対 1 照合（±1.0s）で正解に付かなかった予測＝FP。
      台帳 dataset/ledgers_20261008/検出照合台帳_test20260924.csv があれば件数を照合して記録する。

使い方: python vibrato_depth.py [--test 20260924] [--pre 1.5 --post 0.3] [--tags v2d,v2d2,v2e --lag 0.3 --tol 1.0]
出力: vibrato_depth_events.csv（イベント単位）・vibrato_depth_hist.png・vibrato_depth_報告.md・vibrato_depth_stats.json"""
import os, sys, json, csv, time, argparse, warnings, glob
import numpy as np
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score

T_START = time.time()
BASE = "G:/研究アーカイブ/成果物/動画解析"; ROOT = f"{BASE}/g3v2_20260928"; FR = 100
OUT = f"{ROOT}/vibrato_depth_20261008"; os.makedirs(OUT, exist_ok=True)
LEDGER = f"{BASE}/dataset/ledgers_20261008/検出照合台帳_test20260924.csv"
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
ap = argparse.ArgumentParser()
ap.add_argument("--test", default="20260924"); ap.add_argument("--dev", default="20260911")
ap.add_argument("--pre", type=float, default=1.5); ap.add_argument("--post", type=float, default=0.3)
ap.add_argument("--med", type=float, default=0.3, help="移動中央値の窓[s]")
ap.add_argument("--dmin", type=float, default=15.0, help="山谷差の下限[cent]（pYIN の分解能 10cent より上）")
ap.add_argument("--fmin", type=float, default=3.0); ap.add_argument("--fmax", type=float, default=9.0)
ap.add_argument("--tags", default="v2d,v2d2,v2e"); ap.add_argument("--lag", type=float, default=0.3); ap.add_argument("--tol", type=float, default=1.0)
ap.add_argument("--hid", type=int, default=128); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--suffix", default="", help="出力ファイル名の末尾（感度分析用）"); ap.add_argument("--n_bg", type=int, default=300, help="対照（BG）＝正解・予測から2s以上離れた有声点の数")
a = ap.parse_args(); TAGS = a.tags.split(",")
index = json.load(open(f"{ROOT}/index.json", encoding="utf-8"))
test = [d for d in index if d["session"] == a.test and os.path.exists(f"{ROOT}/feats/{d['video']}.npz")]
print(f"test={a.test} {len(test)}曲  窓=[-{a.pre}, +{a.post}]s  移動中央値={a.med}s  dmin={a.dmin}cent  帯={a.fmin}-{a.fmax}Hz", flush=True)

# ---------------------------------------------------------------- 1. ピッチ曲線（feats npz: st=MIDI半音, voiced, conf, hop）
FEATS = {}
def load_feats(d):
    if d["video"] not in FEATS: FEATS[d["video"]] = np.load(f"{ROOT}/feats/{d['video']}.npz")
    return FEATS[d["video"]]

# ---------------------------------------------------------------- 2. 予測（build_ledgers_20261008.py と同一）→ TP/FP/FN
def predict_all():
    import torch, torch.nn as nn, torch.nn.functional as Fn
    torch.set_num_threads(a.threads)
    def feats_of(z):
        lm = z["logmel"].astype(np.float32); T = lm.shape[1]; st = z["st"]; v = z["voiced"]
        stn = np.where(v > 0, (st - 60.0) / 12.0, 0.0).astype(np.float32)
        dst = np.zeros(T, np.float32); dst[1:] = np.diff(st); dst[1:] *= (v[:-1] > 0); dst = np.where(v > 0, np.clip(dst, -3, 3), 0.0).astype(np.float32)
        return np.concatenate([lm, z["rms"][None, :], stn[None, :], v[None, :], z["conf"][None, :], dst[None, :]], 0).astype(np.float32)
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
    thr_file = f"{ROOT}/out/pitchbar_thr_{'+'.join(TAGS)}_lag{a.lag}.json"
    best = {c: tuple(v) for c, v in json.load(open(thr_file, encoding="utf-8")).items()}
    thr, md = best["vibrato"]; print(f"しきい値(vibrato)=({thr}, 最短間隔{md}s)  cache={os.path.basename(thr_file)}", flush=True)
    k = TYPES.index("vibrato"); w = int(md * FR)
    rows = []
    for d in test:
        z = load_feats(d); prob = probs_of(z)[k]; n = len(prob); lo, hi = d["span"]
        pred = [(i / FR + a.lag, float(prob[i])) for i in range(1, n - 1) if prob[i] >= thr and prob[i] >= prob[max(0, i - w):min(n, i + w + 1)].max()]
        pred = [(t, p) for t, p in pred if lo <= t <= hi]
        truth = sorted(t for c, t in d["events"] if c == "vibrato" and lo <= t <= hi)
        used = set(); pairs = {}
        for pi in sorted(range(len(pred)), key=lambda i: pred[i][0]):
            bst = None
            for kk, t in enumerate(truth):
                if kk in used or abs(t - pred[pi][0]) > a.tol: continue
                if bst is None or abs(t - pred[pi][0]) < abs(truth[bst] - pred[pi][0]): bst = kk
            if bst is not None: used.add(bst); pairs[pi] = bst
        for kk, t in enumerate(truth):
            m = [pi for pi, b in pairs.items() if b == kk]
            rows.append(dict(video=d["video"], kind="truth", sub="TP" if m else "FN", t_event=round(t, 3), prob=round(pred[m[0]][1], 3) if m else "", dt_pred=round(pred[m[0]][0] - t, 3) if m else ""))
        for pi, (t, p) in enumerate(pred):
            if pi not in pairs: rows.append(dict(video=d["video"], kind="FP", sub="FP", t_event=round(t, 3), prob=round(p, 3), dt_pred=""))
    return rows

# ---------------------------------------------------------------- 3. 測り方
def measure(z, t_ev, pre=None, post=None):
    pre = a.pre if pre is None else pre; post = a.post if post is None else post
    hop = float(z["hop"]); t0 = float(z["t0"]); st = z["st"]; v = z["voiced"] > 0.5; T = len(st)
    i0 = max(0, int(round((t_ev - pre - t0) / hop))); i1 = min(T, int(round((t_ev + post - t0) / hop)))
    cents = st[i0:i1].astype(np.float64) * 100.0; vo = v[i0:i1]; n = len(vo)
    out = dict(voiced_frac=round(float(vo.mean()), 3) if n else 0.0, depth_cents=0.0, rate_hz=np.nan, rate_chain_hz=np.nan, duration_s=0.0, n_halfcycles=0, periodicity=np.nan, depth_all=0.0, n_runs=0)
    if n == 0: return out
    # 有声区間ごとに残差
    runs = []; i = 0
    while i < n:
        if not vo[i]: i += 1; continue
        j = i
        while j < n and vo[j]: j += 1
        if j - i >= int(a.med * FR): runs.append((i, j))
        i = j
    out["n_runs"] = len(runs)
    if not runs: return out
    kmed = int(a.med * FR) | 1
    resid = np.full(n, np.nan)
    for (i, j) in runs:
        c = cents[i:j]; resid[i:j] = c - median_filter(c, size=kmed, mode="nearest")
    # 極値列（山谷交互）
    ext = []  # (idx, value, +1/-1)
    for (i, j) in runs:
        r = resid[i:j]
        pk, _ = find_peaks(r, distance=3); tr, _ = find_peaks(-r, distance=3)
        e = sorted([(p + i, r[p], 1) for p in pk] + [(q + i, r[q], -1) for q in tr])
        alt = []
        for x in e:
            if alt and alt[-1][2] == x[2]:
                if (x[2] == 1 and x[1] > alt[-1][1]) or (x[2] == -1 and x[1] < alt[-1][1]): alt[-1] = x
            else: alt.append(x)
        ext.append(alt)
    diffs_all = [abs(al[k + 1][1] - al[k][1]) for al in ext for k in range(len(al) - 1)]
    if diffs_all: out["depth_all"] = round(float(np.median(diffs_all)), 1)
    hmin, hmax = 1.0 / (2 * a.fmax), 1.0 / (2 * a.fmin)
    # 最長の連鎖
    best = None
    for al in ext:
        k = 0
        while k < len(al) - 1:
            m = k
            while m < len(al) - 1 and abs(al[m + 1][1] - al[m][1]) >= a.dmin and hmin <= (al[m + 1][0] - al[m][0]) * hop <= hmax: m += 1
            if m - k >= 3:
                span = (al[m][0] - al[k][0]) * hop
                if best is None or span > best[0]: best = (span, al[k:m + 1])
            k = max(m, k + 1)
    def fft_rate(seg):
        seg = seg[np.isfinite(seg)]
        if len(seg) < 10: return np.nan, np.nan
        x = (seg - seg.mean()) * np.hanning(len(seg)); N = 4096
        P = np.abs(np.fft.rfft(x, N)) ** 2; f = np.fft.rfftfreq(N, hop)
        band = (f >= a.fmin) & (f <= a.fmax); wide = (f >= 0.5) & (f <= 20.0)
        if not band.any() or P[wide].sum() <= 0: return np.nan, np.nan
        return float(f[band][np.argmax(P[band])]), float(P[band].sum() / P[wide].sum())
    if best is not None:
        chain = best[1]
        pp = [abs(chain[k + 1][1] - chain[k][1]) for k in range(len(chain) - 1)]
        half = [(chain[k + 1][0] - chain[k][0]) * hop for k in range(len(chain) - 1)]
        out["depth_cents"] = round(float(np.median(pp)), 1); out["duration_s"] = round(float(best[0]), 2); out["n_halfcycles"] = len(pp)
        out["rate_chain_hz"] = round(1.0 / (2 * float(np.median(half))), 2)
        seg = resid[chain[0][0]:chain[-1][0] + 1]
        if len(seg) < 30: seg = resid[max(0, chain[0][0] - 10):min(n, chain[-1][0] + 11)]
        r, per = fft_rate(seg)
    else:
        out["depth_cents"] = out["depth_all"]
        i, j = max(runs, key=lambda r: r[1] - r[0]); r, per = fft_rate(resid[i:j])
    out["rate_hz"] = round(r, 2) if np.isfinite(r) else np.nan; out["periodicity"] = round(per, 3) if np.isfinite(per) else np.nan
    return out

# ---------------------------------------------------------------- 4. 実行
rows = predict_all()
# 対照（BG）: 採点区間内の有声点で、ビブラートの正解・予測のどちらからも 2s 以上離れた時刻を乱数で選ぶ（seed 固定）
rng = np.random.default_rng(20261008); per = max(1, a.n_bg // len(test)); bg = []
for d in test:
    z = load_feats(d); hop = float(z["hop"]); lo, hi = d["span"]; avoid = np.array([r["t_event"] for r in rows if r["video"] == d["video"]])
    cand = np.arange(max(0, int(lo / hop)), min(len(z["voiced"]), int(hi / hop))); cand = cand[z["voiced"][cand] > 0.5] * hop
    if len(avoid): cand = cand[np.min(np.abs(cand[:, None] - avoid[None, :]), 1) >= 2.0]
    for t in rng.choice(cand, size=min(per, len(cand)), replace=False): bg.append(dict(video=d["video"], kind="BG", sub="BG", t_event=round(float(t), 3), prob="", dt_pred=""))
rows += bg
for r in rows:
    d = next(x for x in test if x["video"] == r["video"]); r.update(measure(load_feats(d), r["t_event"]))
cols = ["video", "kind", "sub", "t_event", "prob", "dt_pred", "voiced_frac", "depth_cents", "rate_hz", "rate_chain_hz", "duration_s", "n_halfcycles", "periodicity", "depth_all", "n_runs"]
with open(f"{OUT}/vibrato_depth_events{a.suffix}.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
truth = [r for r in rows if r["kind"] == "truth"]; fp = [r for r in rows if r["kind"] == "FP"]; bgr = [r for r in rows if r["kind"] == "BG"]
tp = [r for r in truth if r["sub"] == "TP"]; fn = [r for r in truth if r["sub"] == "FN"]
print(f"正解 {len(truth)}件（TP {len(tp)}・FN {len(fn)}）／FP {len(fp)}件／対照BG {len(bgr)}件  {time.time()-T_START:.0f}s", flush=True)
ledger_note = "台帳なし（自前で予測・照合）"
if os.path.exists(LEDGER):
    try:
        import pandas as pd
        L = pd.read_csv(LEDGER, encoding="utf-8-sig"); L.columns = [c.strip() for c in L.columns]
        kc = next((c for c in L.columns if c.lower() in ("kind", "判定", "result")), None); tc = next((c for c in L.columns if c.lower() in ("technique", "type", "技法", "counter")), None)
        if kc and tc:
            sub = L[L[tc].astype(str).str.contains("vibrato|ビブラート")]
            cnt = sub[kc].astype(str).value_counts().to_dict(); ledger_note = f"台帳 {os.path.basename(LEDGER)} の vibrato 件数={cnt}（本スクリプト: TP {len(tp)}・FN {len(fn)}・FP {len(fp)}）"
        else: ledger_note = f"台帳あり（列名不一致のため照合せず: {list(L.columns)[:8]}）"
    except Exception as e: ledger_note = f"台帳あり（読み込み失敗: {e}）"
print(ledger_note)

# ---------------------------------------------------------------- 5. 統計
F = {"depth_cents": "深さ [cent, p-p]", "rate_hz": "速さ [Hz]", "duration_s": "持続 [s]", "periodicity": "周期性（3-9Hz帯の割合）", "voiced_frac": "有声率"}
def arr(rs, k, fill=0.0): return np.array([(r[k] if (r[k] == r[k] and r[k] != "") else fill) for r in rs], float)
def q(x): return [round(float(np.percentile(x, p)), 2) for p in (25, 50, 75)]
stats = {"n": dict(truth=len(truth), TP=len(tp), FN=len(fn), FP=len(fp), BG=len(bgr)), "window": [-a.pre, a.post], "ledger": ledger_note, "features": {}}
y = np.array([1] * len(truth) + [0] * len(fp))
for k in F:
    xt, xf = arr(truth, k), arr(fp, k); x = np.concatenate([xt, xf])
    auc = float(roc_auc_score(y, x)) if len(fp) and len(truth) else float("nan")
    # 1次元しきい値（x ≥ thr → 正解）: バランス精度最大
    cands = np.unique(x); bestt = (0, None, None)
    for th in cands:
        pr = x >= th; tpr = pr[y == 1].mean(); tnr = (~pr[y == 0]).mean(); ba = (tpr + tnr) / 2
        if ba > bestt[0]: bestt = (ba, float(th), (float(tpr), float(tnr)))
    xb = arr(bgr, k); auc_bg = float(roc_auc_score(np.r_[np.ones(len(xt)), np.zeros(len(xb))], np.r_[xt, xb])) if len(bgr) else float("nan")
    stats["features"][k] = dict(label=F[k], truth_q=q(xt), fp_q=q(xf), bg_q=q(xb) if len(bgr) else None, auc_truth_vs_bg=round(auc_bg, 3), fp_vs_bg_auc=round(float(roc_auc_score(np.r_[np.ones(len(xf)), np.zeros(len(xb))], np.r_[xf, xb])), 3) if len(bgr) else None, tp_q=q(arr(tp, k)) if tp else None, fn_q=q(arr(fn, k)) if fn else None,
                                truth_zero=round(float((xt == 0).mean()), 3), fp_zero=round(float((xf == 0).mean()), 3),
                                auc=round(auc, 3), best_thr=bestt[1], best_bal_acc=round(bestt[0], 3), best_tpr_tnr=bestt[2])
# 速さは帯の中にあるか（4〜8Hz）
rt, rf = arr(truth, "rate_hz", np.nan), arr(fp, "rate_hz", np.nan)
stats["rate_in_4_8"] = dict(truth=round(float(np.nanmean((rt >= 4) & (rt <= 8))), 3), fp=round(float(np.nanmean((rf >= 4) & (rf <= 8))), 3), truth_nan=int(np.isnan(rt).sum()), fp_nan=int(np.isnan(rf).sum()))
# 2次元ルール: 深さ > X かつ 速さ ∈ [lo, hi] かつ 持続 ≥ D
dt, df_ = arr(truth, "depth_cents"), arr(fp, "depth_cents"); ut, uf = arr(truth, "duration_s"), arr(fp, "duration_s")
rt0, rf0 = np.nan_to_num(rt, nan=0.0), np.nan_to_num(rf, nan=0.0)
best2 = (0, None)
for X in range(0, 160, 5):
    for (lo, hi) in ((3, 9), (4, 8), (4.5, 7.5), (5, 7)):
        for D in (0.0, 0.2, 0.3, 0.4, 0.5):
            pt = (dt > X) & (rt0 >= lo) & (rt0 <= hi) & (ut >= D); pf = (df_ > X) & (rf0 >= lo) & (rf0 <= hi) & (uf >= D)
            tpr = pt.mean(); tnr = 1 - pf.mean(); ba = (tpr + tnr) / 2
            if ba > best2[0]: best2 = (ba, dict(depth_gt=X, rate_band=[lo, hi], dur_ge=D, tpr=round(float(tpr), 3), tnr=round(float(tnr), 3), keep_truth=int(pt.sum()), keep_fp=int(pf.sum())))
stats["rule2d"] = dict(best_bal_acc=round(float(best2[0]), 3), **best2[1])
# 「周期的な揺れ有り」の単純判定（連鎖あり＝持続>0）
stats["has_chain"] = dict(truth=round(float((ut > 0).mean()), 3), fp=round(float((uf > 0).mean()), 3), bg=round(float((arr(bgr, "duration_s") > 0).mean()), 3) if bgr else None, tp=round(float(arr(tp, "duration_s").__gt__(0).mean()), 3) if tp else None, fn=round(float(arr(fn, "duration_s").__gt__(0).mean()), 3) if fn else None)
json.dump(stats, open(f"{OUT}/vibrato_depth_stats{a.suffix}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(stats, ensure_ascii=False, indent=1))

# ---------------------------------------------------------------- 6. 図
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic"]; plt.rcParams["axes.unicode_minus"] = False
C_T, C_F, C_B = "#2F6FDB", "#E07B2A", "#8A8A8A"
fig, axs = plt.subplots(2, 2, figsize=(11, 8), dpi=130)
bins = {"depth_cents": np.arange(0, 310, 10), "rate_hz": np.arange(2.5, 9.6, 0.25), "duration_s": np.arange(0, 1.85, 0.1)}
for ax, k in zip(axs.ravel()[:3], ["depth_cents", "rate_hz", "duration_s"]):
    xt, xf = arr(truth, k, np.nan), arr(fp, k, np.nan); xt, xf = xt[np.isfinite(xt)], xf[np.isfinite(xf)]
    b = bins[k]; xt = np.clip(xt, b[0], b[-1] - 1e-6); xf = np.clip(xf, b[0], b[-1] - 1e-6)
    ax.hist(xt, bins=b, density=True, alpha=0.55, color=C_T, label=f"正解（DAM加点・n={len(xt)}）", edgecolor="white", linewidth=0.5)
    ax.hist(xf, bins=b, density=True, alpha=0.55, color=C_F, label=f"誤検知 FP（n={len(xf)}）", edgecolor="white", linewidth=0.5)
    xb = arr(bgr, k, np.nan); xb = np.clip(xb[np.isfinite(xb)], b[0], b[-1] - 1e-6)
    if len(xb): ax.hist(xb, bins=b, density=True, histtype="step", color=C_B, lw=1.4, label=f"対照 BG（無作為の有声点・n={len(xb)}）")
    s = stats["features"][k]; ax.axvline(s["truth_q"][1], color=C_T, lw=1.5, ls="--"); ax.axvline(s["fp_q"][1], color=C_F, lw=1.5, ls="--")
    ax.set_title(f"{F[k]}　中央値 正解 {s['truth_q'][1]} / FP {s['fp_q'][1]} / BG {s['bg_q'][1] if s['bg_q'] else '-'}　AUC 正解vsFP={s['auc']:.2f}・正解vsBG={s['auc_truth_vs_bg']:.2f}", fontsize=9)
    ax.set_ylabel("密度", fontsize=9); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=8, frameon=False)
    if k == "depth_cents": ax.set_xlabel("山谷差の中央値 [cent]（右端は 300 以上を含む）", fontsize=9)
    if k == "rate_hz": ax.set_xlabel("FFT ピーク周波数 [Hz]（3〜9Hz に限定）", fontsize=9)
    if k == "duration_s": ax.set_xlabel("周期的な揺れの連鎖の長さ [s]（0＝連鎖なし）", fontsize=9)
ax = axs[1, 1]
ax.scatter(arr(fp, "depth_cents"), np.nan_to_num(arr(fp, "rate_hz", np.nan), nan=2.6), s=14, color=C_F, alpha=0.6, label="誤検知 FP", edgecolor="white", linewidth=0.3)
ax.scatter(arr(truth, "depth_cents"), np.nan_to_num(arr(truth, "rate_hz", np.nan), nan=2.6), s=14, color=C_T, alpha=0.6, label="正解", edgecolor="white", linewidth=0.3)
r2 = stats["rule2d"]; ax.axvline(r2["depth_gt"], color="#555", lw=1, ls=":"); ax.axhspan(r2["rate_band"][0], r2["rate_band"][1], color="#999", alpha=0.12)
ax.set_xlim(0, 300); ax.set_ylim(2.4, 9.3); ax.set_xlabel("深さ [cent]", fontsize=9); ax.set_ylabel("速さ [Hz]（2.6＝算出不能）", fontsize=9)
ax.set_title(f"深さ×速さ　最良ルール: 深さ>{r2['depth_gt']} & 速さ∈{r2['rate_band']} & 持続≥{r2['dur_ge']}  バランス精度 {r2['best_bal_acc']:.2f}", fontsize=9.5)
ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=8, frameon=False)
fig.suptitle(f"ビブラート 深さ・速さ・持続の分布　正解 vs 誤検知（test 9/24・28曲・窓 [{-a.pre:+.1f}, {a.post:+.1f}]s・pYIN 100fps）", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(f"{OUT}/vibrato_depth_hist{a.suffix}.png"); plt.close(fig)
print(f"図 → {OUT}/vibrato_depth_hist{a.suffix}.png  {time.time()-T_START:.0f}s")

# ---------------------------------------------------------------- 7. 遅れ走査: 幅 1.0s の窓をイベント時刻の前後にずらし、周期的な揺れがどこにあるかを見る（DAM の遅れの推定）
if not a.suffix:
    centers = np.arange(-4.0, 0.76, 0.25); scan = []
    V = {d["video"]: load_feats(d) for d in test}
    for c in centers:
        res = {}
        for name, grp in (("truth", truth), ("FP", fp), ("BG", bgr)):
            m = [measure(V[r["video"]], r["t_event"], pre=-(c - 0.5), post=c + 0.5) for r in grp]
            per = np.array([x["periodicity"] if x["periodicity"] == x["periodicity"] else 0.0 for x in m]); ch = np.array([x["duration_s"] > 0 for x in m], float); dp = np.array([x["depth_cents"] for x in m])
            res[name] = dict(per=per, chain=float(ch.mean()), per_med=float(np.median(per)), depth_med=float(np.median(dp)))
        auc_bg = float(roc_auc_score(np.r_[np.ones(len(truth)), np.zeros(len(bgr))], np.r_[res["truth"]["per"], res["BG"]["per"]]))
        auc_fp = float(roc_auc_score(np.r_[np.ones(len(truth)), np.zeros(len(fp))], np.r_[res["truth"]["per"], res["FP"]["per"]]))
        scan.append(dict(center_s=round(float(c), 2), win=f"[{c-0.5:+.2f},{c+0.5:+.2f}]", chain_truth=round(res["truth"]["chain"], 3), chain_FP=round(res["FP"]["chain"], 3), chain_BG=round(res["BG"]["chain"], 3),
                         per_truth=round(res["truth"]["per_med"], 3), per_FP=round(res["FP"]["per_med"], 3), per_BG=round(res["BG"]["per_med"], 3),
                         depth_truth=round(res["truth"]["depth_med"], 1), depth_FP=round(res["FP"]["depth_med"], 1), depth_BG=round(res["BG"]["depth_med"], 1), auc_per_truth_vs_BG=round(auc_bg, 3), auc_per_truth_vs_FP=round(auc_fp, 3)))
    with open(f"{OUT}/vibrato_depth_lagscan.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(scan[0].keys())); w.writeheader(); w.writerows(scan)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), dpi=130)
    xs = [r["center_s"] for r in scan]
    axs[0].plot(xs, [r["chain_truth"] for r in scan], "-o", ms=4, color=C_T, label="正解（DAM加点）"); axs[0].plot(xs, [r["chain_FP"] for r in scan], "-o", ms=4, color=C_F, label="誤検知 FP"); axs[0].plot(xs, [r["chain_BG"] for r in scan], "-o", ms=4, color=C_B, label="対照 BG")
    axs[0].set_xlabel("窓の中心（イベント時刻からのずれ）[s]・窓幅 1.0s", fontsize=9); axs[0].set_ylabel("周期的な揺れ（連鎖）が見つかった割合", fontsize=9); axs[0].set_title("周期的な揺れの位置", fontsize=10)
    axs[1].plot(xs, [r["auc_per_truth_vs_BG"] for r in scan], "-o", ms=4, color=C_T, label="周期性 AUC: 正解 vs 対照BG"); axs[1].plot(xs, [r["auc_per_truth_vs_FP"] for r in scan], "-o", ms=4, color=C_F, label="周期性 AUC: 正解 vs FP"); axs[1].axhline(0.5, color="#999", lw=1, ls=":")
    axs[1].set_xlabel("窓の中心（イベント時刻からのずれ）[s]・窓幅 1.0s", fontsize=9); axs[1].set_ylabel("AUC", fontsize=9); axs[1].set_title("窓の位置と分離の強さ", fontsize=10)
    for ax in axs: ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=8, frameon=False); ax.axvline(0, color="#555", lw=1)
    fig.suptitle("遅れ走査：加点イベントの何秒前に周期的な揺れがあるか（test 9/24・28曲）", fontsize=10.5); fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{OUT}/vibrato_depth_lagscan.png"); plt.close(fig)
    for r in scan: print(r["win"], "連鎖率 正解/FP/BG", r["chain_truth"], r["chain_FP"], r["chain_BG"], " 周期性中央値", r["per_truth"], r["per_FP"], r["per_BG"], " AUC vsBG", r["auc_per_truth_vs_BG"], "vsFP", r["auc_per_truth_vs_FP"])
    print(f"遅れ走査 → {OUT}/vibrato_depth_lagscan.csv/.png  {time.time()-T_START:.0f}s")
