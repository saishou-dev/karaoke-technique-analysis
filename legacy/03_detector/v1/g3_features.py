# -*- coding: utf-8 -*-
"""G3 v1 の特徴量関数（学習 g3_train_v1.py と予測 g3_predict_wav.py で共通）。10ms hop 前提。"""
import math
import numpy as np
from scipy.ndimage import median_filter
HOP = 0.01

def to_semitone(f0, conf, conf_min=0.0):  # pYINのvoiced判定（f0>0）を使う。confは0628の録音（旧ゲイン）で全体に低く、しきい値にすると有声の9割を捨てる
    f0 = np.asarray(f0, float); conf = np.asarray(conf, float)
    v = (f0 > 40) & (conf >= conf_min); st = np.full(len(f0), np.nan); st[v] = 69 + 12 * np.log2(f0[v] / 440.0)
    return st, v

def segments(t, st, v):
    sm = st.copy(); ok = ~np.isnan(sm)
    if ok.sum() < 10: return None
    idx = np.where(ok)[0]; sm[idx] = median_filter(sm[idx], size=5, mode="nearest")
    segs = []; i = 0; n = len(t)
    while i < n:
        if not ok[i]: i += 1; continue
        j = i
        while j + 1 < n and ok[j + 1]:
            a = np.nanmedian(sm[max(i, j - 4):j + 1]); b = np.nanmedian(sm[j + 1:j + 6])
            if abs(b - a) > 0.8 and (j - i) >= 8: break
            j += 1
        if (j - i + 1) * HOP >= 0.08: segs.append((i, j))
        i = j + 1
    return (segs, sm) if segs else None

def feats_onset(t, sm, segs, k):
    i, j = segs[k]; n = j - i + 1
    w = lambda a, b: sm[i + max(0, int(a / HOP)): i + min(n, int(b / HOP))]
    p0 = np.nanmedian(sm[i:i + 3]); tgt = np.nanmedian(w(0.2, 0.45)) if n > 25 else np.nanmedian(sm[i:j + 1])
    rise = tgt - p0; first = w(0.0, 0.12); mn = np.nanmin(first) if len(first) else p0
    reach = next((q for q in range(n) if sm[i + q] >= tgt - 0.3), n) * HOP
    slope = np.nanmax(np.diff(sm[i:i + min(n, 30)])) / HOP if n > 2 else 0
    gap = (t[i] - t[segs[k - 1][1]]) if k > 0 else 5.0
    prevp = np.nanmedian(sm[segs[k - 1][0]:segs[k - 1][1] + 1]) if k > 0 else tgt
    return [rise, tgt - mn, reach, slope, gap, tgt - prevp, n * HOP, p0]

def feats_note(t, sm, segs, k):
    i, j = segs[k]; s = sm[i:j + 1]; n = len(s)
    base = median_filter(s, size=min(31, n if n % 2 else n - 1) if n >= 3 else 1, mode="nearest") if n >= 3 else s
    ex = s - base; up = np.nanmax(ex) if n else 0; dn = np.nanmin(ex) if n else 0
    above = ex > 0.5; runs = np.diff(np.concatenate([[0], above.astype(int), [0]]))
    nb = int((runs == 1).sum()); width = 0
    if nb:
        starts = np.where(runs == 1)[0]; ends = np.where(runs == -1)[0]; width = float(np.max(ends - starts)) * HOP
    return [up, dn, nb, width, n * HOP, float(np.nanstd(s)), float(np.nanmax(s) - np.nanmin(s))]

def feats_offset(t, sm, segs, k):
    i, j = segs[k]; n = j - i + 1
    stab = np.nanmedian(sm[max(i, j - 40):max(i + 1, j - 20)]); pend = np.nanmedian(sm[max(i, j - 2):j + 1])
    drop = pend - stab; q = j
    while q > i and sm[q] < stab - 0.3: q -= 1
    ddur = (j - q) * HOP; slope = np.nanmin(np.diff(sm[max(i, j - 30):j + 1])) / HOP if n > 2 else 0
    gap = (t[segs[k + 1][0]] - t[j]) if k + 1 < len(segs) else 5.0
    return [drop, ddur, slope, gap, n * HOP, pend]

def feats_vib(t, sm, i0, i1):
    s = sm[i0:i1]; ok = ~np.isnan(s)
    if ok.sum() < 30: return None
    s = np.interp(np.arange(len(s)), np.where(ok)[0], s[ok]); d = s - median_filter(s, size=31, mode="nearest")
    d = d - d.mean(); den = np.dot(d, d)
    if den <= 1e-9: return None
    ac = np.correlate(d, d, "full")[len(d) - 1:] / den; lags = np.arange(14, 31)
    r = ac[lags]; b = int(np.argmax(r)); rate = 1.0 / (lags[b] * HOP)
    depth = float(np.std(d) * 2 * math.sqrt(2) * 100)
    zc = np.where(np.diff(np.sign(d)) != 0)[0]; reg = float(np.std(np.diff(zc)) * HOP) if len(zc) > 3 else 1.0
    return [float(r[b]), rate, depth, reg, float(np.nanmean(s)), float(ac[1])]

def candidates(t, sm, segs, win=50, step=10):
    """type -> [(anchor_time, feature_list), ...]"""
    out = {"shakuri": [], "kobushi": [], "fall": [], "vibrato": []}
    for k, (i, j) in enumerate(segs):
        out["shakuri"].append((t[i], feats_onset(t, sm, segs, k)))
        out["kobushi"].append((t[i] + (j - i) * HOP / 2, feats_note(t, sm, segs, k)))
        out["fall"].append((t[j], feats_offset(t, sm, segs, k)))
    for i0 in range(0, len(t) - win, step):
        f = feats_vib(t, sm, i0, i0 + win)
        if f is not None: out["vibrato"].append((t[i0 + win // 2], f))
    return out
