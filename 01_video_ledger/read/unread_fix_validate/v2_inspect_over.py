#!/usr/bin/env python3
"""v2_inspect_over.py ＝ 検証2で k＝3 だけ曲末が公式より多くなった曲×技法の中身を見る（2026-10-30・読むだけ）
引数＝--scan <fullscan> --song <曲> --counter <技法>
出力＝基準と k＝3 の値の並び（同じ値が続く区間ごと）・k＝3 の増分の位置・その区間の箱の数と最良見本の類似度と値
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glyph_core as gc
from v2_fullscan import values_from, events_from, FLAT

ap = argparse.ArgumentParser()
ap.add_argument("--scan", required=True); ap.add_argument("--song", required=True); ap.add_argument("--counter", required=True)
a = ap.parse_args()
z = np.load(os.path.join(a.scan, a.song + ".npz"))
fps = float(z["fps"]); ci = gc.COUNTERS.index(a.counter)
vb = values_from(z["nb"], z["bs"], z["bl"], z["es"], z["el"], False)
vk = values_from(z["nb"], z["bs"], z["bl"], z["es"], z["el"], True)
_, ab, bb = events_from(vb, fps); _, ak, bk = events_from(vk, fps)
print("fps", fps, "band base", bb, "band k3", bk)
print("base acc", ab[a.counter]); print("k3 acc", ak[a.counter])
def runs(v):
    out = []; s = 0
    for i in range(1, len(v) + 1):
        if i == len(v) or v[i] != v[s]:
            out.append((s, i - 1, int(v[s]))); s = i
    return out
diff = np.nonzero(vb[:, ci] != vk[:, ci])[0]
print("frames where k3 value differs:", len(diff))
vals_k = {}
for f in diff:
    vals_k.setdefault((int(vb[f, ci]), int(vk[f, ci])), []).append(int(f))
for (b, k), fs in sorted(vals_k.items(), key=lambda x: -len(x[1])):
    print(f"  base={b} -> k3={k}: {len(fs)} frames  first={fs[0]} ({fs[0]/fps:.2f}s) last={fs[-1]} ({fs[-1]/fps:.2f}s)")
# k3 で値11以上が出る区間の箱情報
sel = np.nonzero(vk[:, ci] >= 10)[0] if a.counter else []
print("k3 frames with value>=10:", len(sel))
for f in sel[:: max(1, len(sel) // 15)][:15]:
    f = int(f)
    print(f"  f={f} t={f/fps:.2f}s nb={int(z['nb'][f, ci])} bs={z['bs'][f, ci].round(3).tolist()} bl={z['bl'][f, ci].tolist()} "
          f"es={z['es'][f, ci].round(3).tolist()} el={z['el'][f, ci].tolist()} base={int(vb[f, ci])} k3={int(vk[f, ci])}")
