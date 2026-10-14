# -*- coding: utf-8 -*-
r"""録音WAVから「ブレス不足か筋疲労（支えの低下）か」を切り分ける時系列指標を出す
   （2026-10-15・10/14 本人追問「ブレス不足なのか腹筋・背筋疲労なのか」への回答用）。
   入力＝録音フォルダ（mapping.csv で T## → 曲名）・出力＝<out>/疲労切り分け_<session>.csv（＋--append-md で歌声分析_<session>.md に節を追記）
   歌声区間の取り方（重要）＝pyin の有声判定だけでは DAM のオケがマイクに回り込んだ区間も「有声」になるため、
     レベル門（有声フレームの dB p95 −20dB 以上）を併用する。-20dB は有声フレームの dB 分布の谷（二峰のあいだ）で、
     門を −15〜−25dB に振っても有声率の変化は 0.03 未満（T01 で 0.649→0.681）。以降「歌声フレーム」＝有声かつ門を超えたフレーム。
   指標（曲ごと・歌声区間のみ）
     band_24k   ＝ 2〜4kHz ÷ 100〜500Hz（dB・LTAS・歌声フレーム平均）＝「鳴り」。band_all は門なし＝song_metrics.py の band と同一定義
     phr_med/max＝ フレーズ長（子音の短い無声 <0.12s を橋渡しした有声連続長・秒）の中央値／最大
     gap_n_min  ＝ 吸気候補（フレーズ間の無声ギャップ 0.15〜2.0s）の回数／歌唱分
     gap_mean   ＝ 吸気候補の平均長（秒）
     decay      ＝ フレーズ内の音量傾き（dB/s・1.0s以上のフレーズの線形回帰の平均・負＝フレーズ後半で落ちる）
     jitter     ＝ local jitter（%）／shimmer ＝ local shimmer（dB）＝ Praat の local 定義式。周期抽出は自前
                   （parselmouth 不在のため、70〜1000Hz に帯域制限し pyin の F0 から予測周期を与えて正ピークを拾い、
                    予測周期の 0.7〜1.4 倍に入る連続周期だけを使う。持続音区間ごとに 20 周期以上あるものを採用）
     f0_mean    ＝ 有声フレームの平均F0（Hz）／f0_half ＝ 曲後半 − 曲前半の平均F0差（半音）
   使い方: python fatigue_breath_20261014.py --session 20261014 [--append-md]"""
import os, sys, glob, csv, argparse, re
import numpy as np, librosa, soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding='utf-8')
ap = argparse.ArgumentParser(); ap.add_argument("--session", required=True); ap.add_argument("--wavdir", default="")
ap.add_argument("--out", default="G:/研究アーカイブ/成果物/動画解析/song_metrics"); ap.add_argument("--append-md", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
REC = "G:/研究アーカイブ/録音"; wavdir = a.wavdir or f"{REC}/songs_{a.session}_raw"
SR = 22050; HOP = 220; FR = SR / HOP          # 10ms 刻み（song_metrics.py と同じ）
BRIDGE = 0.12                                  # 子音の無声をフレーズ内と見なす上限（秒）
GAP_MIN, GAP_MAX = 0.15, 2.0                   # 吸気候補とする無声ギャップの範囲（秒）
GATE = 20.0                                    # レベル門＝有声フレーム dB p95 からの下げ幅（dB）

names = {}
mp = os.path.join(wavdir, "mapping.csv")
if os.path.exists(mp):
    for r in csv.DictReader(open(mp, encoding="utf-8-sig")):
        k = r.get("renamed") or r.get("wav_file") or ""; song = r.get("song") or r.get("track_name") or ""
        m2 = re.match(r"^(.*?)\s+(\d{2,3}\.\d{1,3})$", song)
        if m2: song = m2.group(1)
        names[os.path.basename(k)] = song
def skey(p):
    b = os.path.basename(p); m = re.search(r"_(\d{8}_\d{6})", b)
    if m: return (0, m.group(1))
    m = re.search(r"\((\d+)\)", b); return (1, int(m.group(1)) if m else 0)

def runs(mask):
    """True の連続区間を [start, end] (両端含む) で返す"""
    out = []; i = 0; n = len(mask)
    while i < n:
        if not mask[i]: i += 1; continue
        j = i
        while j + 1 < n and mask[j + 1]: j += 1
        out.append((i, j)); i = j + 1
    return out

def jitter_shimmer(y, segs, f0, hop=HOP):
    """持続音区間ごとに周期ピーク法で local jitter(%) と local shimmer(dB) を出し、区間長で重み付け平均"""
    sos = butter(4, [70, 1000], btype="bandpass", fs=SR, output="sos")
    yb = sosfiltfilt(sos, y)
    js = []; ss = []; wt = []
    for i, j in segs:
        if (j - i + 1) / FR < 0.2: continue
        f = f0[i:j + 1]; f = f[f > 40]
        if len(f) < 10: continue
        fm = float(np.median(f)); T = SR / fm                      # 予測周期（サンプル）
        s = yb[i * hop:(j + 1) * hop]
        if len(s) < 20 * T: continue
        pk, _ = find_peaks(s, distance=max(2, int(T * 0.6)))
        if len(pk) < 21: continue
        d = np.diff(pk).astype(float); amp = np.abs(s[pk])
        ok = (d > 0.7 * T) & (d < 1.4 * T) & (amp[:-1] > 1e-5)
        # 連続して有効な周期の最長ラン（20周期以上）を使う
        for p, q in runs(ok):
            if q - p + 1 < 20: continue
            dd = d[p:q + 1]
            jit = np.mean(np.abs(np.diff(dd))) / np.mean(dd) * 100
            aa = amp[p:q + 2]
            shi = np.mean(np.abs(20 * np.log10(aa[1:] / aa[:-1])))
            js.append(jit); ss.append(shi); wt.append(len(dd))
    if not js: return np.nan, np.nan, 0
    w = np.array(wt, float)
    return float(np.sum(np.array(js) * w) / w.sum()), float(np.sum(np.array(ss) * w) / w.sum()), len(js)

def analyze(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True); y = y.mean(1)
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    f0p, vf, vp = librosa.pyin(y, fmin=70, fmax=800, sr=SR, frame_length=2048, hop_length=HOP)
    f0p = np.nan_to_num(f0p, nan=0.0); v0 = f0p > 40
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]; db = 20 * np.log10(rms + 1e-6)
    n = min(len(v0), len(db), len(f0p)); v0 = v0[:n]; db = db[:n]; f0p = f0p[:n]
    thr = float(np.percentile(db[v0], 95)) - GATE if v0.sum() > 50 else -np.inf
    v = v0 & (db >= thr)                       # 歌声フレーム＝有声＋レベル門（オケの回り込みを除く）
    st = np.where(v, 69 + 12 * np.log2(np.maximum(f0p, 1) / 440.0), np.nan)
    # --- フレーズ（子音の無声 <BRIDGE を橋渡し）
    vb = v.copy()
    for i, j in runs(~v):
        if (j - i + 1) / FR < BRIDGE: vb[i:j + 1] = True
    ph = [(i, j) for i, j in runs(vb) if (j - i + 1) / FR >= 0.3]
    plen = np.array([(j - i + 1) / FR for i, j in ph]) if ph else np.array([np.nan])
    # --- 吸気候補（フレーズ間の無声ギャップ）と歌唱時間
    gaps = []
    for k in range(len(ph) - 1):
        g = (ph[k + 1][0] - ph[k][1] - 1) / FR
        if GAP_MIN <= g <= GAP_MAX: gaps.append(g)
    sing_min = ((ph[-1][1] - ph[0][0] + 1) / FR / 60.0) if ph else np.nan
    # --- フレーズ内の音量傾き（1.0s以上）
    sl = []
    for i, j in ph:
        if (j - i + 1) / FR < 1.0: continue
        t = np.arange(j - i + 1) / FR; sl.append(np.polyfit(t, db[i:j + 1], 1)[0])
    # --- 帯域（2〜4k ÷ 100〜500Hz・有声フレーム）＝song_metrics と同一
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP)) ** 2; fq = librosa.fft_frequencies(sr=SR, n_fft=2048); S = S[:, :n]
    def bandof(mask):
        Sv = S[:, mask].mean(1) if mask.sum() > 0 else S.mean(1)
        return 10 * np.log10(Sv[(fq >= 2000) & (fq <= 4000)].mean() / (Sv[(fq >= 100) & (fq <= 500)].mean() + 1e-20) + 1e-12)
    band = bandof(v); band_all = bandof(v0)
    # --- 持続音（song_metrics と同じ切り方）→ jitter / shimmer
    sm = st.copy(); idx = np.where(v)[0]
    if len(idx) > 10: sm[idx] = median_filter(sm[idx], size=5, mode="nearest")
    segs = []; i = 0
    while i < n:
        if not v[i]: i += 1; continue
        j = i
        while j + 1 < n and v[j + 1] and abs(np.nanmedian(sm[max(i, j - 4):j + 1]) - np.nanmedian(sm[j + 1:j + 6])) <= 0.8: j += 1
        if (j - i + 1) / FR >= 0.15: segs.append((i, j))
        i = j + 1
    jit, shi, nseg = jitter_shimmer(y, segs, f0p)
    # --- F0 の前後半差
    half = n // 2
    f1 = st[:half][v[:half]]; f2 = st[half:][v[half:]]
    fh = (float(np.nanmean(f2)) - float(np.nanmean(f1))) if (len(f1) > 50 and len(f2) > 50) else np.nan
    fv = f0p[v]
    return dict(
        dur=round(n / FR, 1), voiced=round(float(v.mean()), 3), voiced_raw=round(float(v0.mean()), 3),
        band_24k=round(float(band), 2), band_all=round(float(band_all), 2),
        phr_n=len(ph), phr_med=round(float(np.nanmedian(plen)), 2), phr_max=round(float(np.nanmax(plen)), 2),
        gap_n=len(gaps), gap_n_min=round(len(gaps) / sing_min, 1) if sing_min and sing_min > 0 else np.nan,
        gap_mean=round(float(np.mean(gaps)), 3) if gaps else np.nan,
        decay=round(float(np.mean(sl)), 2) if sl else np.nan,
        jitter=round(jit, 3) if jit == jit else np.nan, shimmer=round(shi, 3) if shi == shi else np.nan, js_seg=nseg,
        f0_mean=round(float(np.mean(fv)), 1) if len(fv) else np.nan, f0_half=round(fh, 2) if fh == fh else np.nan)

files = sorted(glob.glob(os.path.join(wavdir, "*.wav")), key=skey)
rows = []
for k, w in enumerate(files, 1):
    base = os.path.basename(w); song = names.get(base) or os.path.splitext(base)[0].split("_2026")[0]
    mt = re.search(r"_\d{8}_(\d{6})", base)
    try: m = analyze(w)
    except Exception as e: m = dict(error=str(e))
    rows.append(dict(order=k, file=base, song=song, time=mt.group(1) if mt else "", **m))
    print(k, base, song, {q: m.get(q) for q in ("phr_med", "gap_n_min", "jitter", "shimmer", "band_24k", "decay")}, flush=True)
keys = ["order", "file", "song", "time", "dur", "voiced", "voiced_raw", "band_24k", "band_all", "phr_n", "phr_med", "phr_max",
        "gap_n", "gap_n_min", "gap_mean", "decay", "jitter", "shimmer", "js_seg", "f0_mean", "f0_half"]
csvp = f"{a.out}/疲労切り分け_{a.session}.csv"
with open(csvp, "w", newline="", encoding="utf-8-sig") as f:
    wri = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); wri.writeheader(); wri.writerows(rows)
print("DONE", csvp)

# ---- 3分割の平均と曲順との Spearman 相関
ok = [r for r in rows if "error" not in r]
MET = [("phr_med", "フレーズ長中央値s"), ("phr_max", "フレーズ長最大s"), ("gap_n_min", "吸気/分"), ("gap_mean", "吸気平均長s"),
       ("decay", "フレーズ内傾きdB/s"), ("jitter", "ジッター%"), ("shimmer", "シマーdB"), ("band_24k", "2-4k比dB"),
       ("f0_mean", "平均F0Hz"), ("f0_half", "F0前後半差半音"), ("voiced", "有声率")]
def thirds(key):
    xs = [r.get(key) for r in ok]; xs = [np.nan if x in ("", None) else float(x) for x in xs]
    g = len(xs) // 3 or 1
    a3 = [float(np.nanmean(xs[0:g])), float(np.nanmean(xs[g:2 * g])), float(np.nanmean(xs[2 * g:]))]
    o = np.arange(len(xs), dtype=float); m = ~np.isnan(np.array(xs, float))
    rho, p = spearmanr(o[m], np.array(xs, float)[m]) if m.sum() >= 5 else (np.nan, np.nan)
    return a3, float(rho), float(p)
stats = {k: thirds(k) for k, _ in MET}
for k, lab in MET:
    a3, rho, p = stats[k]; print(f"{lab}: {a3[0]:.3f} / {a3[1]:.3f} / {a3[2]:.3f}  rho={rho:+.3f} p={p:.4f}")
import json
open(f"{a.out}/疲労切り分け_{a.session}_stats.json", "w", encoding="utf-8").write(
    json.dumps({"session": a.session, "n": len(ok), "metrics": {k: {"thirds": stats[k][0], "rho": stats[k][1], "p": stats[k][2], "label": lab} for k, lab in MET}}, ensure_ascii=False, indent=1))
print("DONE", f"{a.out}/疲労切り分け_{a.session}_stats.json")
