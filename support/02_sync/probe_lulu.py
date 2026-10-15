#!/usr/bin/env python3
import numpy as np
import sys
import os

SW = sys.argv[1]
HOP = 80

def envelope(path):
    x = np.fromfile(path, dtype=np.int16).astype(np.float32) / 32768.0
    n = len(x) // HOP
    e = np.sqrt(np.mean(x[:n*HOP].reshape(n, HOP) ** 2, axis=1))
    l = np.log1p(e * 100)
    return l - l.mean()

v = envelope(os.path.join(SW, "vid_lulu._20260628_094340.s16"))
t = envelope(os.path.join(SW, "wav_20260628_take-01.s16"))
region = t[78000:]  # 780s以降
n = len(v) + len(region)
N = 1 << (n - 1).bit_length()
c = np.fft.irfft(np.fft.rfft(region, N) * np.conj(np.fft.rfft(v, N)), N)
valid = len(region) - len(v)
t2 = np.concatenate([[0], np.cumsum(region * region)])
win = t2[len(v):len(v)+valid] - t2[:valid]
r = c[:valid] / (np.sqrt(win) * np.linalg.norm(v) + 1e-9)
i = int(np.argmax(r))
print(f"lulu. in take-01[780s+]: offset={780 + i/100:.2f}s score={r[i]:.3f}")
