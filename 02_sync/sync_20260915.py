# -*- coding: utf-8 -*-
"""録画と録音の同期（2026-09-15版・sync_20260911.py の一般化・Windows python・librosa）。方式は9/11と同じ：
   8kHzモノ化→RMS包絡(10ms)の相互相関（主）＋オンセット強度の相互相関（検証）→±0.05s一致で確定、不一致は声帯域(300-3000Hz)RMSで裁定。
   対応付け＝<video dir>/rename_mapping.csv（動画）× 録音/songs_<session>_raw/mapping.csv（録音）を played_at で結合。
   出力: dataset/sync/sync_offsets_<session>.csv（video, wav, offset_sec=wav_time-video_time, ...）
   使い方: python scripts/sync_20260915.py [session]
"""
import os, sys, csv, subprocess
import numpy as np, librosa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915 import Cfg, FFMPEG

C = Cfg(); S = C.S
SR = 8000; HOP = 80
def env(y):
    r = librosa.feature.rms(y=y, frame_length=400, hop_length=HOP)[0]; return (r - r.mean()) / (r.std() + 1e-9)
def onset(y):
    o = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, fmin=200, fmax=3400); return (o - o.mean()) / (o.std() + 1e-9)
def xcorr(a, b):
    n = len(a) + len(b); N = 1 << (n - 1).bit_length(); A = np.fft.rfft(a, N); B = np.fft.rfft(b, N); c = np.fft.irfft(A * np.conj(B), N)
    c = np.concatenate([c[-(len(b) - 1):], c[:len(a)]]); lags = np.arange(-(len(b) - 1), len(a)); k = int(np.argmax(c))
    ov = min(len(a), len(b)); r = c[k] / ov; srt = np.sort(c)[::-1]; ratio = srt[1] / srt[0] if srt[0] > 0 else 1
    return lags[k] * HOP / SR, float(r), float(ratio)
for p, what in ((C.rename_csv, "rename_mapping.csv"), (os.path.join(C.wav_dir, "mapping.csv"), "録音 mapping.csv")):
    if not os.path.exists(p): C.log(f"SYNC STOP: {what} が無い {p}"); raise SystemExit(2)
vids = {r["played_at"]: r["renamed"] for r in csv.DictReader(open(C.rename_csv, encoding="utf-8-sig")) if r.get("played_at")}
wavs = {}
for r in csv.DictReader(open(os.path.join(C.wav_dir, "mapping.csv"), encoding="utf-8-sig")):
    if r.get("played_at"): wavs.setdefault(r["played_at"], []).append(r["renamed"])
tmp = os.path.join(C.root, "dataset", "sync", "tmp_" + S); os.makedirs(tmp, exist_ok=True)
rows = []
for pa, vname in sorted(vids.items()):
    for wname in wavs.get(pa, []):
        va = os.path.join(tmp, os.path.splitext(vname)[0] + "_8k.wav")
        if not os.path.exists(va): subprocess.run([FFMPEG, "-y", "-v", "error", "-i", os.path.join(C.vid_dir, vname), "-vn", "-ac", "1", "-ar", str(SR), va], check=True)
        yv, _ = librosa.load(va, sr=SR, mono=True); yw, _ = librosa.load(os.path.join(C.wav_dir, wname), sr=SR, mono=True)
        lag1, r1, ratio1 = xcorr(env(yv), env(yw)); lag2, r2, ratio2 = xcorr(onset(yv), onset(yw))
        agree = abs(lag1 - lag2) <= 0.05; method = "rms+onset"
        if not agree:
            import scipy.signal as ss
            b, a = ss.butter(4, [300 / (SR / 2), 3000 / (SR / 2)], btype="band"); lag3, r3, _ = xcorr(env(ss.lfilter(b, a, yv)), env(ss.lfilter(b, a, yw)))
            method = "band-arbitrated"; lag = lag3 if abs(lag3 - lag1) <= 0.05 or abs(lag3 - lag2) > 0.05 else lag2
            if abs(lag3 - lag1) > 0.05 and abs(lag3 - lag2) <= 0.05: lag = lag2
        else: lag = lag1
        rows.append(dict(video=os.path.splitext(vname)[0], wav=wname, offset_sec=round(-lag, 3), method=method, r_main=round(r1, 3), r_onset=round(r2, 3),
                         lag_main=round(lag1, 3), lag_onset=round(lag2, 3), agree=agree, peak_ratio=round(ratio1, 3), played_at=pa))
        C.log("SYNC " + str(rows[-1]))
if not rows: C.log("SYNC STOP: played_at で結合できる動画×録音の組が0"); raise SystemExit(2)
os.makedirs(os.path.dirname(C.sync_csv), exist_ok=True)
with open(C.sync_csv, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
C.log(f"SYNC WROTE {C.sync_csv} pairs={len(rows)} band-arbitrated={sum(1 for r in rows if r['method']!='rms+onset')}")
