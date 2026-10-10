# -*- coding: utf-8 -*-
"""9/24録音（SSL2）のpYIN f0抽出（2026-10-08・pitch_pyin_20260924.py のコピー・Windows python・librosa 1.0）。WSLの pitch_pyin.py と同一パラメータ（SR22050・hop220・fmin70/fmax800・frame2048）。
   出力: dataset/pitch_pyin_20261006/<wav stem>_f0.csv（時間軸＝WAV自身・列 audio_time_sec,f0_hz,conf,voiced）。既存 pitch_pyin/ は触らない。
   使い方: python scripts/pitch_pyin_20261006.py [--workers 6]  ログ: scripts/pitch_pyin_20261006.log"""
import os, sys, csv, glob, argparse, time, concurrent.futures as cf
ROOT = r"G:\研究アーカイブ\成果物\動画解析"; WAV = r"G:\研究アーカイブ\録音\songs_20261006_raw"
OUT = os.path.join(ROOT, "dataset", "pitch_pyin_20261006"); SR = 22050; HOP = 220
def one(p):
    import numpy as np, librosa
    stem = os.path.splitext(os.path.basename(p))[0]; dst = os.path.join(OUT, stem + "_f0.csv")
    if os.path.exists(dst): return stem, "exists"
    t0 = time.time(); y, _ = librosa.load(p, sr=SR, mono=True)
    f0, vflag, vprob = librosa.pyin(y, fmin=70, fmax=800, sr=SR, frame_length=2048, hop_length=HOP)
    times = librosa.times_like(f0, sr=SR, hop_length=HOP)
    tmp = dst + ".part"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["audio_time_sec", "f0_hz", "conf", "voiced"])
        for t, fz, vf, vp in zip(times, f0, vflag, vprob):
            w.writerow([f"{t:.3f}", "0.00" if np.isnan(fz) else f"{fz:.2f}", f"{vp:.3f}", int(bool(vf))])
    os.replace(tmp, dst)
    return stem, f"ok voiced={float(np.mean(vflag)):.0%} median={float(np.nanmedian(f0)):.0f}Hz {time.time()-t0:.0f}s"
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); files = sorted(glob.glob(os.path.join(WAV, "*.wav")))
    log = open(os.path.join(ROOT, "scripts", "pitch_pyin_20261006.log"), "a", encoding="utf-8")
    def L(s): print(s, flush=True); log.write(s + "\n"); log.flush()
    L(f"START {time.strftime('%H:%M:%S')} files={len(files)} workers={a.workers}"); t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for stem, st in ex.map(one, files): L(f"{stem}: {st}")
    L(f"PITCH_DONE {time.strftime('%H:%M:%S')} elapsed={time.time()-t0:.0f}s")
