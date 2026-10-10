# -*- coding: utf-8 -*-
"""スマホ抽出音声（dataset/phone_audio_<S>/<video>.wav・22050Hz mono・伴奏混入＝参考値）の LTAS 帯域比（2026-09-15版）。
   方式は 9/14 の ltas_20260911.csv と同じ：STFT 2048/hop 512・有声フレーム＝RMS > (曲内95pct−25dB, 下限−45dBFS) かつ pYIN 有声。
   pYIN の有声フラグは dataset/pitch_pyin_phone_<S>/<video>_f0.csv（hop 220）を最近傍時刻で引く（無ければ RMS 条件のみ）。
   帯域＝B 0-1k／A 2-4k／C 1-2k／D 4-8k に加え **E 2.5-3.5k（歌手のフォルマント帯）** を算出し、E/B と E/全体(0-8k) を dB で出す。
   使い方: python scripts/ltas_phone_20260915.py [session] → dataset/ltas_phone_<S>.csv（標準出力に表）
"""
import os, sys, csv, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915 import Cfg, ROOT

SR = 22050; NFFT = 2048; HOP = 512
C = Cfg(); S = C.S
AUDIO = os.path.join(ROOT, "dataset", f"phone_audio_{S}")
PYIN = os.path.join(ROOT, "dataset", f"pitch_pyin_phone_{S}")
OUT = os.path.join(ROOT, "dataset", f"ltas_phone_{S}.csv")
BANDS = dict(B=(0, 1000), A=(2000, 4000), C=(1000, 2000), D=(4000, 8000), E=(2500, 3500), ALL=(0, 8000))


def band_power(spec_mean, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi); return float(spec_mean[m].sum())


def analyze(wav):
    import soundfile as sf, librosa
    y, sr = sf.read(wav, dtype="float32")
    if y.ndim > 1: y = y.mean(axis=1)
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR); sr = SR
    peak = 20 * np.log10(max(float(np.abs(y).max()), 1e-9))
    St = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP)) ** 2      # power
    freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    times = librosa.frames_to_time(np.arange(St.shape[1]), sr=SR, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=NFFT, hop_length=HOP)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-9))
    thr = max(float(np.percentile(rms_db, 95)) - 25.0, -45.0)
    voiced = rms_db > thr
    stem = os.path.splitext(os.path.basename(wav))[0]
    f0p = os.path.join(PYIN, stem + "_f0.csv"); f0_med = ""
    if os.path.exists(f0p):
        rows = list(csv.DictReader(open(f0p, encoding="utf-8-sig")))
        pt = np.array([float(r["audio_time_sec"]) for r in rows]); pv = np.array([int(r["voiced"]) for r in rows]); pf = np.array([float(r["f0_hz"]) for r in rows])
        idx = np.clip(np.searchsorted(pt, times), 0, len(pt) - 1)
        voiced &= pv[idx] == 1
        f = pf[idx][voiced]; f = f[f > 0]; f0_med = f"{np.median(f):.1f}" if len(f) else ""
    if voiced.sum() < 10: return None
    L = St[:, voiced].mean(axis=1)
    P = {k: band_power(L, freqs, *v) for k, v in BANDS.items()}
    db = lambda a, b: 10 * np.log10(max(P[a], 1e-20) / max(P[b], 1e-20))
    cent = librosa.feature.spectral_centroid(S=np.sqrt(St[:, voiced]), sr=SR)[0]
    return dict(session=S, video=stem, peak_dBFS=f"{peak:.1f}", voice_thr_dBFS=f"{thr:.1f}", frames=int(voiced.sum()), voiced_s=f"{voiced.sum()*HOP/SR:.1f}",
                **{"A/B_2-4k_over_0-1k_dB": f"{db('A','B'):.2f}", "C/B_1-2k_over_0-1k_dB": f"{db('C','B'):.2f}", "D/B_4-8k_over_0-1k_dB": f"{db('D','B'):.2f}",
                   "E/B_2.5-3.5k_over_0-1k_dB": f"{db('E','B'):.2f}", "E/ALL_2.5-3.5k_over_0-8k_dB": f"{db('E','ALL'):.2f}"},
                centroid_med_Hz=f"{np.median(cent):.0f}", rms_med_dBFS=f"{np.median(rms_db[voiced]):.1f}", f0_med_Hz=f0_med)


if __name__ == "__main__":
    wavs = sorted(glob.glob(os.path.join(AUDIO, "*.wav")))
    if not wavs: print("STOP: 抽出WAVが無い", AUDIO); raise SystemExit(2)
    import concurrent.futures as cf
    rows = [r for r in cf.ProcessPoolExecutor(max_workers=4).map(analyze, wavs) if r]
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    for r in rows: print(f"{r['video'][:40]:40s} E/B={r['E/B_2.5-3.5k_over_0-1k_dB']:>7s} E/ALL={r['E/ALL_2.5-3.5k_over_0-8k_dB']:>7s} A/B={r['A/B_2-4k_over_0-1k_dB']:>7s} voiced={r['voiced_s']}s rms={r['rms_med_dBFS']}")
    print("WROTE", OUT, len(rows))
