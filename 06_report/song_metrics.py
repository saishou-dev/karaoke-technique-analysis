# -*- coding: utf-8 -*-
r"""研究セッションの録音WAVから「歌みた収録」と同じ観点の数値を曲ごとに出す（2026-09-29・本人依頼「録音を歌みた収録と同じように分析して毎回」）。
   入力＝録音フォルダ（曲名_日時.wav または mapping.csv でT##→曲名）・出力＝<out>/歌声分析_<session>.md と .csv
   指標＝音程の乗り（12平均律格子からの偏差・持続音のみ）／強弱の幅／鳴りの再現性（フレーズ間の音量ばらつき）／帯域バランス（2〜4kHz÷低域）／
        ビブラート（回数・速さ・深さ）／有声率・音域／（任意）DAM点数（Hearts CSV）
   使い方: python song_metrics.py --session 20260924 [--wavdir G:\研究アーカイブ\録音\songs_20260924_raw]"""
import os, sys, glob, csv, argparse, json, re
import numpy as np, librosa, soundfile as sf
from scipy.ndimage import median_filter
sys.stdout.reconfigure(encoding='utf-8')
ap = argparse.ArgumentParser(); ap.add_argument("--session", required=True); ap.add_argument("--wavdir", default=""); ap.add_argument("--out", default="G:/研究アーカイブ/成果物/動画解析/song_metrics")
ap.add_argument("--dam", default="", help="Hearts CSV（任意・曲名と点数を付ける）")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
REC = "G:/研究アーカイブ/録音"; wavdir = a.wavdir or f"{REC}/songs_{a.session}_raw"
SR = 22050; HOP = 220; FR = SR / HOP
# 曲名の対応（mapping.csv があれば T## → song）
names = {}; mscore = {}  # 2026-10-11 追記: mapping 側の点数（score列／status の score=／8/07 の track_name 末尾）を予備に持つ
mp = os.path.join(wavdir, "mapping.csv")
if os.path.exists(mp):
    for r in csv.DictReader(open(mp, encoding="utf-8-sig")):
        k = r.get("renamed") or r.get("wav_file") or ""; song = r.get("song") or r.get("track_name") or ""; sc = r.get("score") or ""
        if not sc:
            m = re.search(r"score=([\d.]+)", r.get("status") or ""); sc = m.group(1) if m else ""
        m2 = re.match(r"^(.*?)\s+(\d{2,3}\.\d{1,3})$", song)
        if m2: song = m2.group(1); sc = sc or m2.group(2)
        names[os.path.basename(k)] = song
        if sc: mscore[os.path.basename(k)] = float(sc)
def norm(s):  # 曲名の正規化（[生音]・(生音)・括弧書き・空白を除き小文字）
    s = re.sub(r"\[生音\]|[（(]生音[)）]", "", str(s)); s = re.sub(r"[（(\[].*?[)）\]]", "", s); return re.sub(r"\s+", "", s).lower()
def skey(p):  # 録音の時刻順（ファイル名 _YYYYMMDD_HHMMSS → なければ Track(N) の N）
    b = os.path.basename(p); m = re.search(r"_(\d{8}_\d{6})", b)
    if m: return (0, m.group(1))
    m = re.search(r"\((\d+)\)", b); return (1, int(m.group(1)) if m else 0)
def analyze(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True); y = y.mean(1)
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    f0p, vf, vp = librosa.pyin(y, fmin=70, fmax=800, sr=SR, frame_length=2048, hop_length=HOP)  # 有声判定は pyin（既定刻み）
    f0y = librosa.yin(y, fmin=70, fmax=800, sr=SR, frame_length=2048, hop_length=HOP)  # 音高は yin（連続値・量子化なし）
    f0p = np.nan_to_num(f0p, nan=0.0); v = f0p > 40
    f0 = np.where(v & (np.abs(12 * np.log2(np.maximum(f0y, 1) / np.maximum(f0p, 1))) < 0.6), f0y, f0p)  # yin が pyin と半音以内で一致する時だけ連続値を採用
    st = np.where(v, 69 + 12 * np.log2(np.maximum(f0, 1) / 440.0), np.nan)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]; db = 20 * np.log10(rms + 1e-6)
    n = min(len(v), len(db)); v = v[:n]; st = st[:n]; db = db[:n]
    # 持続音（有声ラン≥0.15s・音程跳び<0.8半音で分割）
    sm = st.copy(); idx = np.where(v)[0]
    if len(idx) > 10: sm[idx] = median_filter(sm[idx], size=5, mode="nearest")
    segs = []; i = 0
    while i < n:
        if not v[i]: i += 1; continue
        j = i
        while j + 1 < n and v[j + 1] and abs(np.nanmedian(sm[max(i, j - 4):j + 1]) - np.nanmedian(sm[j + 1:j + 6])) <= 0.8: j += 1
        if (j - i + 1) / FR >= 0.15: segs.append((i, j))
        i = j + 1
    # 音程の乗り＝各持続音の中央値の格子偏差（セント）
    devs = []
    for i, j in segs:
        m = np.nanmedian(sm[i:j + 1]); devs.append(abs(m - round(m)) * 100)
    devs = np.array(devs) if devs else np.array([np.nan])
    # 強弱・再現性（フレーズ＝有声ラン≥0.5s の平均dB）
    ph = []; i = 0
    while i < n:
        if not v[i]: i += 1; continue
        j = i
        while j + 1 < n and v[j + 1]: j += 1
        if (j - i + 1) / FR >= 0.5: ph.append(db[i:j + 1].mean())
        i = j + 1
    ph = np.array(ph) if ph else np.array([np.nan]); vdb = db[v]
    dyn = (np.percentile(vdb, 95) - np.percentile(vdb, 5)) if v.sum() > 100 else np.nan
    # 帯域バランス（LTAS・有声区間）：2〜4kHz ÷ 100〜500Hz（dB）
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP)) ** 2; fq = librosa.fft_frequencies(sr=SR, n_fft=2048); S = S[:, :n]
    Sv = S[:, v].mean(1) if v.sum() > 0 else S.mean(1)
    hi = Sv[(fq >= 2000) & (fq <= 4000)].mean(); lo = Sv[(fq >= 100) & (fq <= 500)].mean(); band = 10 * np.log10(hi / lo + 1e-12)
    # ビブラート（持続音≥0.4s・自己相関 4〜8Hz・深さ≥35cent）
    vib = []
    for i, j in segs:
        s = sm[i:j + 1]; s = s[~np.isnan(s)]
        if len(s) < 40: continue
        d = s - median_filter(s, size=31, mode="nearest"); d -= d.mean(); den = np.dot(d, d)
        if den <= 1e-9: continue
        ac = np.correlate(d, d, "full")[len(d) - 1:] / den; lags = np.arange(12, 26); r = ac[lags]; b = int(np.argmax(r))
        depth = float(np.std(d) * 283); rate = FR / lags[b]
        if r[b] >= 0.3 and depth >= 35: vib.append((rate, depth, len(s) / FR))
    vib = np.array(vib) if vib else np.zeros((0, 3))
    stv = sm[v]; return dict(
        dur=round(n / FR, 1), voiced=round(float(v.mean()), 2), n_notes=len(segs),
        pitch_dev_med=round(float(np.nanmedian(devs)), 1), pitch_within25=round(float(np.mean(devs <= 25)), 2) if len(devs) else np.nan,
        dyn_range=round(float(dyn), 1), phrase_sd=round(float(np.nanstd(ph)), 1), band=round(float(band), 1),
        vib_n=int(len(vib)), vib_rate=round(float(vib[:, 0].mean()), 1) if len(vib) else np.nan, vib_depth=round(float(vib[:, 1].mean()), 0) if len(vib) else np.nan,
        range_lo=round(float(np.nanpercentile(stv, 5)), 1), range_hi=round(float(np.nanpercentile(stv, 95)), 1))
dam = []  # 2026-10-11 置換: 当日の全採点行を (時刻, 正規化曲名, 点, アーティスト, 機種) で持ち、録音ファイルの時刻と最近傍で結合
if a.dam and os.path.exists(a.dam):
    import pandas as pd
    d = pd.read_csv(a.dam, encoding="utf-8-sig", dtype=str)
    col = next((c for c in ("totalScore", "totalPoints", "totalPoint", "point") if c in d.columns), None)
    for _, r in d.iterrows():
        sd = str(r.get("scoringDateTime", ""))
        if sd[:8] != a.session: continue
        try: val = float(r.get(col, 0)) if col else 0.0
        except Exception: val = 0.0
        val = val / 1000 if val > 1000 else val
        dam.append(dict(t=sd[8:14], key=norm(r.get("songName", "")), score=round(val, 3), artist=str(r.get("artistName", "")), machine={"ai": "Ai", "hearts": "WAO"}.get(str(r.get("src", "")), ""), used=False))
def tsec(hhmmss): return int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:6])
files = sorted(glob.glob(os.path.join(wavdir, "*.wav")), key=skey)
match = {}
cands = []
for w in files:
    base = os.path.basename(w); song = names.get(base) or os.path.splitext(base)[0].split("_2026")[0]; key = norm(song)
    m = re.search(r"_\d{8}_(\d{6})", base); ts = m.group(1) if m else None
    for i, drow in enumerate(dam):
        if drow["key"] != key: continue
        if ts: diff = abs(tsec(ts) - tsec(drow["t"]))
        elif base in mscore and abs(mscore[base] - drow["score"]) < 0.002: diff = 0  # 時刻なし（8/07）＝点数一致で結合
        else: continue
        if diff <= 420: cands.append((diff, base, i))
for diff, base, i in sorted(cands):
    if base in match or dam[i]["used"]: continue
    match[base] = dam[i]; dam[i]["used"] = True
rows = []
for w in files:
    base = os.path.basename(w); song = names.get(base) or os.path.splitext(base)[0].split("_2026")[0]
    try: m = analyze(w)
    except Exception as e: m = dict(error=str(e))
    dm = match.get(base); m["dam"] = dm["score"] if dm else mscore.get(base, ""); m["artist"] = dm["artist"] if dm else ""; m["machine"] = dm["machine"] if dm else ""
    mt = re.search(r"_\d{8}_(\d{6})", base); m["time"] = mt.group(1) if mt else (dm["t"] if dm else "")
    rows.append(dict(file=base, song=song, **m)); print(base, song, m.get("dam"), m.get("machine"), {k: v for k, v in m.items() if k in ("pitch_dev_med", "dyn_range", "band", "vib_n")}, flush=True)
# 出力
keys = ["file", "song", "time", "machine", "artist", "dam", "dur", "voiced", "n_notes", "pitch_dev_med", "pitch_within25", "dyn_range", "phrase_sd", "band", "vib_n", "vib_rate", "vib_depth", "range_lo", "range_hi"]
with open(f"{a.out}/歌声分析_{a.session}.csv", "w", newline="", encoding="utf-8-sig") as f:
    wri = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); wri.writeheader(); wri.writerows(rows)
def n2(x): return "" if x is None or (isinstance(x, float) and np.isnan(x)) else x
md = [f"# 歌声分析 {a.session}（録音WAV・{len(rows)}曲）", "",
      "指標＝音程の乗り（持続音の12平均律からの偏差・中央値・セント／±25c以内の割合）・強弱の幅（有声区間のdB p95−p5）・鳴りの再現性（フレーズ平均dBの標準偏差＝小さいほど揃う）・帯域（2〜4kHz÷100〜500Hz・dB＝高いほど明るい）・ビブラート（検出数・速さHz・深さcent）・音域（半音番号 p5〜p95・69=A4）", "",
      "| 曲 | DAM | 長さ | 有声率 | 音程偏差c | ±25c内 | 強弱dB | 再現性dB | 帯域dB | ビブ数 | 速さ | 深さ | 音域 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if "error" in r: md.append(f"| {r['song']} | エラー {r['error'][:40]} |"); continue
    md.append(f"| {r['song']} | {r.get('dam','')} | {r['dur']} | {r['voiced']} | {r['pitch_dev_med']} | {n2(r['pitch_within25'])} | {r['dyn_range']} | {r['phrase_sd']} | {r['band']} | {r['vib_n']} | {n2(r['vib_rate'])} | {n2(r['vib_depth'])} | {r['range_lo']}〜{r['range_hi']} |")
ok = [r for r in rows if "error" not in r]
if ok:
    med = lambda k: round(float(np.nanmedian([r[k] for r in ok])), 1)
    md += ["", f"セッション中央値＝音程偏差 {med('pitch_dev_med')}c／強弱 {med('dyn_range')}dB／再現性 {med('phrase_sd')}dB／帯域 {med('band')}dB／ビブ数 {med('vib_n')}"]
open(f"{a.out}/歌声分析_{a.session}.md", "w", encoding="utf-8").write("\n".join(md)); print("DONE", f"{a.out}/歌声分析_{a.session}.md")
