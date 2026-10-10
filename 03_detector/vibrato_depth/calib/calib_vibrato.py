# -*- coding: utf-8 -*-
r"""ビブラート測定（vibrato_depth.py の measure）の較正＝「答えが分かっている入力」で測定の信頼性を数値化する（2026-10-08）。

本人の問い（10/08）: 「基準値を理論から持ってきても、その解析（測定）が信頼できるかの評価はできない」
→ 測定の信頼性は、深さ・速さ・持続が既知の合成音を 録音→pYIN→測定 の全経路に通し、真値との誤差で示す（較正）。
→ 実録音では、同じ WAV を 0.1s ずらして測り直し、値が一致するか（テスト・リテスト信頼性）を示す。

測定関数は ../vibrato_depth.py の measure() をそのまま写した（元ファイルは変更しない。argparse 由来の a.* だけ P[*] に置換）。
pYIN は既存の f0 抽出（scripts/pitch_pyin_20260924.py・extract_feats.py）と同じ: SR 22050・hop 220・fmin 70・fmax 800・frame 2048（librosa 1.0）。

合成音: 基本波＋倍音 9 本（振幅 1/h）。基本周波数 220／330／440 Hz。ビブラート＝周波数変調（セント単位の正弦波・山谷差 depth_pp）。
       深さ(山谷差) 20〜150 cent・速さ 4〜8 Hz・持続 0.3〜2.0 s（前後に一定音 0.5 s）。劣化＝白色雑音 SNR 30/20/10 dB・簡易残響（指数減衰 IR・RT60 0.3 s・直接音と残響のエネルギー比 0 dB）・残響＋SNR20。
       乱数シード 3 本（ビブラート位相・倍音位相・雑音・IR が変わる）。

使い方: python calib_vibrato.py [--workers 12] [--analyze_only] [--skip_retest]
出力（同フォルダ）: calib_results.csv（合成 1 本=1 行）・calib_summary.json・calib_heatmap.png・calib_duration.png・retest.csv・retest_summary.json"""
import os, sys, json, csv, time, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy.signal import find_peaks, fftconvolve
from scipy.ndimage import median_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"
SR = 22050; HOP = 220; HOP_S = HOP / SR; FR = 100
P = dict(med=0.3, dmin=15.0, fmin=3.0, fmax=9.0)   # vibrato_depth.py の既定値と同じ
PYIN = dict(fmin=70, fmax=800, sr=SR, frame_length=2048, hop_length=HOP)

# ---------------------------------------------------------------- 測定（vibrato_depth.py measure() の写し）
def measure(st, voiced, hop, t_ev, pre, post, t0=0.0):
    v = voiced > 0.5; T = len(st)
    i0 = max(0, int(round((t_ev - pre - t0) / hop))); i1 = min(T, int(round((t_ev + post - t0) / hop)))
    cents = st[i0:i1].astype(np.float64) * 100.0; vo = v[i0:i1]; n = len(vo)
    out = dict(voiced_frac=round(float(vo.mean()), 3) if n else 0.0, depth_cents=0.0, rate_hz=np.nan, rate_chain_hz=np.nan, duration_s=0.0, n_halfcycles=0, periodicity=np.nan, depth_all=0.0, n_runs=0)
    if n == 0: return out
    runs = []; i = 0
    while i < n:
        if not vo[i]: i += 1; continue
        j = i
        while j < n and vo[j]: j += 1
        if j - i >= int(P["med"] * FR): runs.append((i, j))
        i = j
    out["n_runs"] = len(runs)
    if not runs: return out
    kmed = int(P["med"] * FR) | 1
    resid = np.full(n, np.nan)
    for (i, j) in runs:
        c = cents[i:j]; resid[i:j] = c - median_filter(c, size=kmed, mode="nearest")
    ext = []
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
    hmin, hmax = 1.0 / (2 * P["fmax"]), 1.0 / (2 * P["fmin"])
    best = None
    for al in ext:
        k = 0
        while k < len(al) - 1:
            m = k
            while m < len(al) - 1 and abs(al[m + 1][1] - al[m][1]) >= P["dmin"] and hmin <= (al[m + 1][0] - al[m][0]) * hop <= hmax: m += 1
            if m - k >= 3:
                span = (al[m][0] - al[k][0]) * hop
                if best is None or span > best[0]: best = (span, al[k:m + 1])
            k = max(m, k + 1)
    def fft_rate(seg):
        seg = seg[np.isfinite(seg)]
        if len(seg) < 10: return np.nan, np.nan
        x = (seg - seg.mean()) * np.hanning(len(seg)); N = 4096
        Pw = np.abs(np.fft.rfft(x, N)) ** 2; f = np.fft.rfftfreq(N, hop)
        band = (f >= P["fmin"]) & (f <= P["fmax"]); wide = (f >= 0.5) & (f <= 20.0)
        if not band.any() or Pw[wide].sum() <= 0: return np.nan, np.nan
        return float(f[band][np.argmax(Pw[band])]), float(Pw[band].sum() / Pw[wide].sum())
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

# ---------------------------------------------------------------- pYIN（既存の f0 抽出と同一設定）→ MIDI 半音・有声
def pyin_st(y, frame_length=None):
    import librosa
    kw = dict(PYIN)
    if frame_length: kw["frame_length"] = int(frame_length)
    f0, vf, vp = librosa.pyin(y.astype(np.float32), **kw)
    st = np.where(np.isfinite(f0) & (f0 > 40), 69 + 12 * np.log2(np.maximum(np.nan_to_num(f0, nan=1.0), 1) / 440.0), np.nan)
    voiced = (~np.isnan(st)).astype(np.float32)
    return np.nan_to_num(st, nan=0.0).astype(np.float32), voiced

# ---------------------------------------------------------------- 合成音
F0S = [220.0, 330.0, 440.0]; DEPTHS = [20, 40, 60, 80, 100, 150]; RATES = [4, 5, 6, 7, 8]; DURS = [0.3, 0.6, 1.0, 2.0]
DEGS = [("clean", None, False), ("snr30", 30, False), ("snr20", 20, False), ("snr10", 10, False), ("rev", None, True), ("rev+snr20", 20, True)]
SEEDS = [1, 2, 3]; PRE = 0.5; POST = 0.5; NHARM = 10

def synth(f0, depth_pp, rate, dur, rng):
    n = int(round((PRE + dur + POST) * SR)); t = np.arange(n) / SR
    env = np.zeros(n); i0 = int(PRE * SR); i1 = int((PRE + dur) * SR); env[i0:i1] = 1.0
    ramp = int(0.02 * SR); w = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
    env[i0:i0 + ramp] *= w; env[i1 - ramp:i1] *= w[::-1]
    phi = rng.uniform(0, 2 * np.pi)
    cents = (depth_pp / 2.0) * env * np.sin(2 * np.pi * rate * (t - PRE) + phi)
    f_inst = f0 * 2.0 ** (cents / 1200.0)
    ph = 2 * np.pi * np.cumsum(f_inst) / SR
    y = np.zeros(n)
    for h in range(1, NHARM + 1):
        if h * f0 < SR / 2: y += (1.0 / h) * np.sin(h * ph + rng.uniform(0, 2 * np.pi))
    fade = int(0.05 * SR); fw = 0.5 * (1 - np.cos(np.pi * np.arange(fade) / fade)); y[:fade] *= fw; y[-fade:] *= fw[::-1]
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-12) * 0.1

def degrade(y, snr_db, reverb, rng):
    if reverb:
        L = int(0.3 * SR); tt = np.arange(L) / SR
        tail = rng.normal(size=L) * np.exp(-6.908 * tt / 0.3); tail *= 1.0 / np.sqrt(np.sum(tail ** 2))   # 残響エネルギー = 直接音（DRR 0 dB）
        h = np.concatenate([[1.0], tail]); y = fftconvolve(y, h)[:len(y)]
    if snr_db is not None:
        ps = np.mean(y ** 2); nz = rng.normal(size=len(y)); nz *= np.sqrt(ps / 10 ** (snr_db / 10) / np.mean(nz ** 2)); y = y + nz
    return y

def job(args):
    f0, depth, rate, dur, (deg, snr, rev), seed = args[:6]; fl = args[6] if len(args) > 6 else PYIN["frame_length"]
    rng = np.random.default_rng(seed * 1000003 + int(f0) * 1009 + depth * 101 + rate * 11 + int(dur * 10))
    y = degrade(synth(f0, depth, rate, dur, rng), snr, rev, rng)
    st, v = pyin_st(y, fl); total = len(y) / SR
    m = measure(st, v, HOP_S, t_ev=total, pre=total, post=0.0)
    r = dict(f0=f0, depth_true=depth, rate_true=rate, dur_true=dur, deg=deg, snr_db="" if snr is None else snr, reverb=int(rev), seed=seed, frame_length=fl)
    r.update(m)
    r["chain_found"] = int(m["duration_s"] > 0)
    r["depth_err"] = round(m["depth_cents"] - depth, 1); r["depth_ratio"] = round(m["depth_cents"] / depth, 3)
    r["rate_err"] = round(m["rate_hz"] - rate, 2) if np.isfinite(m["rate_hz"]) else np.nan
    r["rate_chain_err"] = round(m["rate_chain_hz"] - rate, 2) if np.isfinite(m["rate_chain_hz"]) else np.nan
    r["dur_err"] = round(m["duration_s"] - dur, 2)                      # 定義どおり（極値間）
    r["dur_err_corr"] = round(m["duration_s"] + 0.5 / rate - dur, 2) if m["duration_s"] > 0 else np.nan   # 極値間は真の区間より半周期短い分を補正
    return r

# ---------------------------------------------------------------- 実録音のテスト・リテスト
def retest_song(args):
    import librosa
    d, shift = args
    y, _ = librosa.load(d["wav"], sr=SR, mono=True)
    if shift > 0: y = y[int(round(shift * SR)):]
    st, v = pyin_st(y)
    lo, hi = d["span"]; rows = []
    for c, t in d["events"]:
        if c != "vibrato" or not (lo <= t <= hi): continue
        for wname, pre, post in (("w_-1.5_+0.3", 1.5, 0.3), ("w_-2.5_-1.0", 2.5, -1.0)):
            m = measure(st, v, HOP_S, t, pre, post, t0=shift)
            rows.append(dict(video=d["video"], t_event=round(t, 3), shift_s=shift, window=wname, **m))
    return rows

# ---------------------------------------------------------------- 集計・図
def analyze(out_dir):
    import pandas as pd
    df = pd.read_csv(f"{out_dir}/calib_results.csv", encoding="utf-8-sig")
    df["abs_depth_err"] = df["depth_err"].abs(); df["abs_rate_err"] = df["rate_err"].abs(); df["abs_rate_chain_err"] = df["rate_chain_err"].abs(); df["abs_dur_err"] = df["dur_err_corr"].abs()
    df["within10"] = (df["abs_depth_err"] <= 10).astype(float); df["rate_within05"] = (df["abs_rate_err"] <= 0.5).astype(float)
    S = {}
    def agg(g):
        return dict(n=int(len(g)), depth_mean_err=round(float(g["depth_err"].mean()), 1), depth_medabs=round(float(g["abs_depth_err"].median()), 1), depth_maxabs=round(float(g["abs_depth_err"].max()), 1),
                    depth_within10=round(float(g["within10"].mean()), 3), depth_ratio_med=round(float(g["depth_ratio"].median()), 3), chain_found=round(float(g["chain_found"].mean()), 3),
                    rate_mean_err=round(float(g["rate_err"].mean()), 2) if g["rate_err"].notna().any() else None, rate_medabs=round(float(g["abs_rate_err"].median()), 2) if g["rate_err"].notna().any() else None,
                    rate_maxabs=round(float(g["abs_rate_err"].max()), 2) if g["rate_err"].notna().any() else None, rate_within05=round(float(g["rate_within05"].mean()), 3),
                    rate_chain_medabs=round(float(g["abs_rate_chain_err"].median()), 2) if g["rate_chain_err"].notna().any() else None,
                    dur_mean_err_raw=round(float(g["dur_err"].mean()), 2), dur_medabs_corr=round(float(g["abs_dur_err"].median()), 2) if g["abs_dur_err"].notna().any() else None, dur_maxabs_corr=round(float(g["abs_dur_err"].max()), 2) if g["abs_dur_err"].notna().any() else None)
    long = df[df["dur_true"] >= 0.6]
    S["overall_clean_dur>=0.6"] = agg(long[long["deg"] == "clean"])
    S["by_depth_clean_dur>=0.6"] = {int(k): agg(g) for k, g in long[long["deg"] == "clean"].groupby("depth_true")}
    S["by_deg_dur>=0.6"] = {k: agg(g) for k, g in long.groupby("deg")}
    S["by_deg_x_depth_dur>=0.6"] = {f"{k[0]}|{int(k[1])}": agg(g) for k, g in long.groupby(["deg", "depth_true"])}
    S["by_f0_clean_dur>=0.6"] = {int(k): agg(g) for k, g in long[long["deg"] == "clean"].groupby("f0")}
    S["by_rate_clean_dur>=0.6"] = {int(k): agg(g) for k, g in long[long["deg"] == "clean"].groupby("rate_true")}
    S["by_dur_x_rate_clean"] = {f"{k[0]}|{int(k[1])}": agg(g) for k, g in df[df["deg"] == "clean"].groupby(["dur_true", "rate_true"])}
    S["by_dur_clean"] = {float(k): agg(g) for k, g in df[df["deg"] == "clean"].groupby("dur_true")}
    S["by_dur_all_deg"] = {f"{k[0]}|{k[1]}": agg(g) for k, g in df.groupby(["deg", "dur_true"])}
    json.dump(S, open(f"{out_dir}/calib_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic"]; plt.rcParams["axes.unicode_minus"] = False
    labels = {"clean": "劣化なし", "snr30": "雑音 SNR 30 dB", "snr20": "雑音 SNR 20 dB", "snr10": "雑音 SNR 10 dB", "rev": "残響 RT 0.3 s", "rev+snr20": "残響＋SNR 20 dB"}
    rowspec = [("abs_depth_err", "深さ |誤差| 中央値 [cent]", 0, 40, "{:.0f}"), ("abs_rate_err", "速さ |誤差| 中央値 [Hz]", 0, 1.0, "{:.2f}"), ("abs_dur_err", "持続 |誤差| 中央値 [s]（半周期補正後）", 0, 0.3, "{:.2f}")]
    fig, axs = plt.subplots(len(rowspec), len(DEGS), figsize=(3.0 * len(DEGS), 2.9 * len(rowspec)), dpi=130)
    for ci, (deg, _, _) in enumerate(DEGS):
        sub = long[long["deg"] == deg]
        for ri, (col, title, vmin, vmax, fmt) in enumerate(rowspec):
            ax = axs[ri, ci]
            M = sub.groupby(["depth_true", "rate_true"])[col].median().unstack("rate_true").reindex(index=DEPTHS, columns=RATES)
            im = ax.imshow(M.values, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
            for i in range(len(DEPTHS)):
                for j in range(len(RATES)):
                    val = M.values[i, j]; txt = "-" if not np.isfinite(val) else fmt.format(val)
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color="white" if (np.isfinite(val) and val > 0.6 * vmax) else "#222")
            ax.set_xticks(range(len(RATES))); ax.set_xticklabels([f"{r}" for r in RATES], fontsize=8); ax.set_yticks(range(len(DEPTHS))); ax.set_yticklabels([f"{d}" for d in DEPTHS], fontsize=8)
            if ri == 0: ax.set_title(labels[deg], fontsize=9.5)
            if ci == 0: ax.set_ylabel(f"{title}\n真の深さ [cent, 山谷差]", fontsize=8)
            if ri == len(rowspec) - 1: ax.set_xlabel("真の速さ [Hz]", fontsize=8)
            ax.tick_params(length=0)
        for ri in range(len(rowspec)):
            if ci == len(DEGS) - 1: fig.colorbar(axs[ri, ci].images[0], ax=axs[ri, :].tolist(), fraction=0.012, pad=0.01)
    fig.suptitle("較正：合成ビブラート（f0 220/330/440 Hz・持続 0.6/1.0/2.0 s・3 シードを併合）を 録音→pYIN→測定 に通した誤差の中央値", fontsize=10.5)
    fig.savefig(f"{out_dir}/calib_heatmap.png", bbox_inches="tight"); plt.close(fig)

    # 持続×速さ：連鎖が見つかる率・速さの誤差（深さ 40 cent 以上・劣化なし／SNR20）
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.6), dpi=130)
    sub = df[(df["depth_true"] >= 40) & (df["deg"] == "clean")]
    M1 = sub.groupby(["dur_true", "rate_true"])["chain_found"].mean().unstack("rate_true").reindex(index=DURS, columns=RATES)
    M2 = sub.groupby(["dur_true", "rate_true"])["abs_rate_err"].median().unstack("rate_true").reindex(index=DURS, columns=RATES)
    sub2 = df[(df["depth_true"] >= 40) & (df["deg"] == "snr20")]
    M3 = sub2.groupby(["dur_true", "rate_true"])["abs_rate_err"].median().unstack("rate_true").reindex(index=DURS, columns=RATES)
    for ax, M, title, vmin, vmax, cmap, fmt in ((axs[0], M1, "周期的な揺れ（連鎖）が見つかった率（劣化なし）", 0, 1, "YlGn", "{:.2f}"), (axs[1], M2, "速さ |誤差| 中央値 [Hz]（劣化なし）", 0, 1.0, "YlOrRd", "{:.2f}"), (axs[2], M3, "速さ |誤差| 中央値 [Hz]（SNR 20 dB）", 0, 1.0, "YlOrRd", "{:.2f}")):
        ax.imshow(M.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
        for i in range(len(DURS)):
            for j in range(len(RATES)):
                val = M.values[i, j]; ax.text(j, i, "-" if not np.isfinite(val) else fmt.format(val), ha="center", va="center", fontsize=8, color="#222")
        ax.set_xticks(range(len(RATES))); ax.set_xticklabels(RATES, fontsize=8); ax.set_yticks(range(len(DURS))); ax.set_yticklabels(DURS, fontsize=8)
        ax.set_xlabel("真の速さ [Hz]", fontsize=8); ax.set_ylabel("真の持続 [s]", fontsize=8); ax.set_title(title, fontsize=9); ax.tick_params(length=0)
    fig.suptitle("持続が短いときの推定可否（深さ 40 cent 以上・f0 3 種・3 シード併合）", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(f"{out_dir}/calib_duration.png"); plt.close(fig)
    # 補足図: 分析窓ごとの 深さ比（測定/真）× 速さ
    fs = f"{out_dir}/calib_frame_sweep.csv"
    if os.path.exists(fs):
        W = pd.read_csv(fs, encoding="utf-8-sig"); W = W[W["depth_true"] >= 40]
        S["frame_sweep_depth_ratio_by_rate"] = {f"fl{int(fl)}|{deg}": {int(r): round(float(g["depth_ratio"].median()), 3) for r, g in gg.groupby("rate_true")} for (fl, deg), gg in W.groupby(["frame_length", "deg"])}
        S["frame_sweep_depth_within10_by_fl"] = {f"fl{int(fl)}|{deg}": round(float((gg["depth_err"].abs() <= 10).mean()), 3) for (fl, deg), gg in W.groupby(["frame_length", "deg"])}
        fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.6), dpi=130)
        for ax, deg in zip(axs, ("clean", "snr30", "snr20")):
            for fl, col in ((2048, "#C0392B"), (1024, "#E67E22"), (512, "#2F6FDB")):
                g = W[(W["frame_length"] == fl) & (W["deg"] == deg)].groupby("rate_true")["depth_ratio"]
                med = g.median(); q1 = g.quantile(0.25); q3 = g.quantile(0.75)
                ax.plot(med.index, med.values, "-o", ms=4, color=col, label=f"frame {fl}（{fl/SR*1000:.0f} ms）" + ("＝既存設定" if fl == 2048 else ""))
                ax.fill_between(med.index, q1.values, q3.values, color=col, alpha=0.12)
            rr = np.linspace(3.8, 8.2, 100); Wn = 2048 / SR
            ax.plot(rr, np.abs(np.sin(np.pi * Wn * rr) / (np.pi * Wn * rr)), ":", color="#555", lw=1, label="理論 sinc(W·f)（W=93 ms の一様平均）")
            ax.axhline(1.0, color="#999", lw=0.8); ax.set_ylim(0, 1.15); ax.set_xlabel("真の速さ [Hz]", fontsize=8); ax.set_ylabel("深さ比＝測定／真（中央値・Q1〜Q3）", fontsize=8)
            ax.set_title(labels[deg], fontsize=9); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=7, frameon=False)
        fig.suptitle("補足：pYIN の分析窓の長さと深さの過小評価（深さ 40 cent 以上・持続 1.0 s・f0 3 種・3 シード）", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(f"{out_dir}/calib_frame_sweep.png"); plt.close(fig)
        json.dump(S, open(f"{out_dir}/calib_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return S

def analyze_retest(out_dir):
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr
    R = pd.read_csv(f"{out_dir}/retest.csv", encoding="utf-8-sig"); S = {}
    for (w, sh), g in R[R["shift_s"] > 0].groupby(["window", "shift_s"]):
        a_ = R[(R["window"] == w) & (R["shift_s"] == 0.0)].set_index(["video", "t_event"]); b_ = g.set_index(["video", "t_event"])
        j = a_.join(b_, lsuffix="_a", rsuffix="_b", how="inner"); res = dict(n=int(len(j)), shift_s=float(sh))
        for k in ("depth_cents", "rate_hz", "duration_s", "periodicity"):
            x, y = j[f"{k}_a"].values.astype(float), j[f"{k}_b"].values.astype(float); ok = np.isfinite(x) & np.isfinite(y)
            d = y[ok] - x[ok]
            res[k] = dict(n_valid=int(ok.sum()), diff_median=round(float(np.median(d)), 3), absdiff_median=round(float(np.median(np.abs(d))), 3), absdiff_p90=round(float(np.percentile(np.abs(d), 90)), 3),
                          pearson_r=round(float(pearsonr(x[ok], y[ok])[0]), 3) if ok.sum() > 2 else None, spearman_r=round(float(spearmanr(x[ok], y[ok])[0]), 3) if ok.sum() > 2 else None,
                          within=round(float(np.mean(np.abs(d) <= {"depth_cents": 10, "rate_hz": 0.5, "duration_s": 0.1, "periodicity": 0.05}[k])), 3))
        ca, cb = (j["duration_s_a"] > 0), (j["duration_s_b"] > 0)
        res["chain_agreement"] = round(float((ca == cb).mean()), 3); res["chain_found_a"] = round(float(ca.mean()), 3); res["chain_found_b"] = round(float(cb.mean()), 3)
        S[f"{w}|shift{sh}"] = res
    # 既存の feats（CSV 由来の f0・10ms グリッド）による値との比較（窓 [-1.5,+0.3]）
    ev = f"{ROOT}/vibrato_depth_20261008/vibrato_depth_events.csv"
    if os.path.exists(ev):
        E = pd.read_csv(ev, encoding="utf-8-sig"); E = E[E["kind"] == "truth"].copy(); E["t_event"] = E["t_event"].round(3)
        a_ = R[(R["window"] == "w_-1.5_+0.3") & (R["shift_s"] == 0.0)].copy(); a_["t_event"] = a_["t_event"].round(3)
        j = a_.merge(E, on=["video", "t_event"], suffixes=("_new", "_feats")); res = dict(n=int(len(j)))
        for k in ("depth_cents", "rate_hz", "duration_s"):
            x, y = j[f"{k}_new"].values.astype(float), j[f"{k}_feats"].values.astype(float); ok = np.isfinite(x) & np.isfinite(y); d = y[ok] - x[ok]
            res[k] = dict(n_valid=int(ok.sum()), diff_median=round(float(np.median(d)), 3), absdiff_median=round(float(np.median(np.abs(d))), 3), pearson_r=round(float(pearsonr(x[ok], y[ok])[0]), 3) if ok.sum() > 2 else None)
        S["vs_existing_feats_w_-1.5_+0.3"] = res
    json.dump(S, open(f"{out_dir}/retest_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return S

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--analyze_only", action="store_true"); ap.add_argument("--skip_retest", action="store_true"); ap.add_argument("--skip_synth", action="store_true"); ap.add_argument("--retest_shifts", default="0,0.1", help="切り落とし秒（カンマ区切り・0 を含める）。既存 retest.csv に無い shift だけ追加する")
    a = ap.parse_args(); T0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    if not a.analyze_only and not a.skip_synth:
        jobs = [(f0, d, r, du, deg, s) for f0 in F0S for d in DEPTHS for r in RATES for du in DURS for deg in DEGS for s in SEEDS]
        print(f"合成 {len(jobs)} 本  workers={a.workers}", flush=True); rows = []
        with ProcessPoolExecutor(a.workers) as ex:
            for i, r in enumerate(ex.map(job, jobs, chunksize=8)):
                rows.append(r)
                if (i + 1) % 500 == 0: print(f"  {i+1}/{len(jobs)}  {time.time()-T0:.0f}s", flush=True)
        cols = list(rows[0].keys())
        with open(f"{HERE}/calib_results.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        print(f"calib_results.csv {len(rows)} 行  {time.time()-T0:.0f}s", flush=True)
    if not a.analyze_only and not a.skip_retest:
        index = json.load(open(f"{ROOT}/index.json", encoding="utf-8"))
        test = [d for d in index if d["session"] == "20260924" and os.path.exists(d["wav"])]
        shifts = [float(x) for x in a.retest_shifts.split(",")]; rows = []
        if os.path.exists(f"{HERE}/retest.csv"):
            import pandas as pd
            old = pd.read_csv(f"{HERE}/retest.csv", encoding="utf-8-sig"); have = set(old["shift_s"].round(4).unique()); rows = old.to_dict("records"); shifts = [x for x in shifts if round(x, 4) not in have]
        jobs = [(d, s) for d in test for s in shifts]
        print(f"実録音 {len(test)} 曲 × 切り落とし {shifts} s（既存行 {len(rows)}）", flush=True)
        with ProcessPoolExecutor(min(8, a.workers)) as ex:
            for i, rr in enumerate(ex.map(retest_song, jobs)):
                rows += rr; print(f"  {i+1}/{len(jobs)} {jobs[i][0]['video']} shift={jobs[i][1]}  {time.time()-T0:.0f}s", flush=True)
        if rows:
            with open(f"{HERE}/retest.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"retest.csv {len(rows)} 行  {time.time()-T0:.0f}s", flush=True)
    if not a.analyze_only and not a.skip_synth:
        # 補足: pYIN の分析窓（frame_length）を短くしたときの深さの回復（既存設定 2048 ＝ 93 ms が揺れを平均してしまう仮説の検証）
        jobs = [(f0, d, r, 1.0, deg, s, fl) for fl in (2048, 1024, 512) for f0 in F0S for d in DEPTHS for r in RATES for deg in DEGS[:3] for s in SEEDS]
        print(f"補足（分析窓の掃引） {len(jobs)} 本", flush=True); rows = []
        with ProcessPoolExecutor(a.workers) as ex:
            for r in ex.map(job, jobs, chunksize=8): rows.append(r)
        with open(f"{HERE}/calib_frame_sweep.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"calib_frame_sweep.csv {len(rows)} 行  {time.time()-T0:.0f}s", flush=True)
    S = analyze(HERE); print(json.dumps({k: S[k] for k in ("overall_clean_dur>=0.6", "by_depth_clean_dur>=0.6", "by_deg_dur>=0.6", "by_dur_clean")}, ensure_ascii=False, indent=1))
    if os.path.exists(f"{HERE}/retest.csv"):
        R = analyze_retest(HERE); print(json.dumps(R, ensure_ascii=False, indent=1))
    print(f"DONE {time.time()-T0:.0f}s")
