# -*- coding: utf-8 -*-
"""同曲の「直近の過去演奏」を DTW で今回の時間軸に合わせ、過去のピッチ輪郭（基準メロディの代替）と過去の加点イベントを
   今回の10msグリッドに載せて pairs/<video>.npz に保存する。全セッション対象（過去＝自分より前のセッション）。"""
import os, sys, json, time
import numpy as np, librosa, soundfile as sf
sys.stdout.reconfigure(encoding='utf-8')
ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"; PD = f"{ROOT}/pairs"; os.makedirs(PD, exist_ok=True)
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]; SR = 22050; HOPS = 0.1; HOP = int(round(HOPS * SR)); FR = 100
index = json.load(open(f"{ROOT}/index.json", encoding="utf-8")); titles = json.load(open(f"{ROOT}/titles.json", encoding="utf-8"))
def chroma(d):
    cache = f"{ROOT}/feats/{d['video']}_chroma{int(HOPS*1000)}.npy"
    if os.path.exists(cache): return np.load(cache)
    y, sr = sf.read(d["wav"], dtype="float32", always_2d=True); y = y.mean(1)
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    C = librosa.feature.chroma_stft(y=y, sr=SR, n_fft=4096, hop_length=HOP); rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=HOP)[0]
    X = np.vstack([C, (np.log(rms + 1e-6) / 5.0)[None, :]]).astype(np.float32); np.save(cache, X); return X
by_title = {}
for d in index: by_title.setdefault(titles[d["video"]], []).append(d)
t0 = time.time(); n = 0
for d in index:
    past = [p for p in by_title[titles[d["video"]]] if p["session"] < d["session"]]
    if not past: continue
    p = max(past, key=lambda x: x["session"]); out = f"{PD}/{d['video']}.npz"
    if os.path.exists(out): n += 1; continue
    D, wp = librosa.sequence.dtw(X=chroma(p), Y=chroma(d), metric="cosine"); wp = wp[::-1]
    si = wp[:, 0] * HOPS; dj = wp[:, 1] * HOPS
    u, inv = np.unique(dj, return_inverse=True); m = np.zeros(len(u)); c = np.zeros(len(u)); np.add.at(m, inv, si); np.add.at(c, inv, 1); m /= c  # 今回時刻→過去時刻
    zc = np.load(f"{ROOT}/feats/{d['video']}.npz"); T = len(zc["st"]); t_cur = np.arange(T) / FR
    t_past = np.interp(t_cur, u, m)
    zp = np.load(f"{ROOT}/feats/{p['video']}.npz"); stp = zp["st"].astype(float); vp = zp["voiced"] > 0; stp[~vp] = np.nan
    ip = np.clip(np.round(t_past * FR).astype(int), 0, len(stp) - 1); past_st = stp[ip]; past_v = vp[ip]
    # 過去イベント→今回時刻（past→cur の逆写像）
    u2, inv2 = np.unique(si, return_inverse=True); m2 = np.zeros(len(u2)); c2 = np.zeros(len(u2)); np.add.at(m2, inv2, dj); np.add.at(c2, inv2, 1); m2 /= c2
    lo, hi = p["span"]; ev = [(TYPES.index(cc), float(np.interp(t, u2, m2))) for cc, t in p["events"] if cc in TYPES and lo <= t <= hi]
    ev_type = np.array([e[0] for e in ev], int); ev_time = np.array([e[1] for e in ev], float)
    np.savez_compressed(out, past_video=p["video"], past_st=np.nan_to_num(past_st, nan=0.0).astype(np.float32), past_voiced=past_v.astype(np.float32), t_past=t_past.astype(np.float32), ev_type=ev_type, ev_time=ev_time, cost=float(D[-1, -1] / (D.shape[0] + D.shape[1])))
    n += 1; print(d["video"], "<-", p["video"], f"{time.time()-t0:.0f}s", flush=True)
print("DONE pairs", n)
