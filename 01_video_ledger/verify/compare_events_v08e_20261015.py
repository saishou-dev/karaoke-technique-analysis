# -*- coding: utf-8 -*-
"""読取器を差し替えたときの副作用を測る（2026-10-15）。
   2つの results ディレクトリの counter_events.csv を突合し、増加イベント（delta>0）が
   「同じか・時刻が動いたか・消えたか・増えたか」を件数と秒で出す。D1（観測の信頼性）の副作用検査用。
   使い方: python compare_events_v08e_20261015.py <旧results> <新results>
     例: python compare_events_v08e_20261015.py results_wao_20261014 results_wao_20261014_v08e
   （相対名は G:\\研究アーカイブ\\成果物\\動画解析 の下として解釈する）
"""
import csv, io, os, re, sys

if sys.stdout is not None: sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\研究アーカイブ\成果物\動画解析"
A = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(ROOT, sys.argv[1])
B = sys.argv[2] if os.path.isabs(sys.argv[2]) else os.path.join(ROOT, sys.argv[2])


def load(d):
    out = {}
    for n in sorted(os.listdir(d)):
        p = os.path.join(d, n, "counter_events.csv")
        if not os.path.exists(p): continue
        fps = 59.2
        lg = os.path.join(d, n, "read_log.txt")
        if os.path.exists(lg):
            m = re.search(r"fps=([\d.]+)", io.open(lg, encoding="utf-8", errors="replace").read())
            if m: fps = float(m.group(1))
        out[n] = ({(x["counter"], int(x["value"])): int(x["event_frame_est"])
                   for x in csv.DictReader(io.open(p, encoding="utf-8-sig")) if int(x["delta"]) > 0}, fps)
    return out


a, b = load(A), load(B)
common = sorted(set(a) & set(b))
shifts, rem, add = [], [], []
for n in common:
    sa, fps = a[n]; sb, _ = b[n]
    for k in set(sa) | set(sb):
        if k not in sb: rem.append((n, k, sa[k]))
        elif k not in sa: add.append((n, k, sb[k]))
        elif sa[k] != sb[k]: shifts.append((abs(sa[k] - sb[k]) / fps, n, k, sa[k], sb[k]))
na = sum(len(a[n][0]) for n in common); nb = sum(len(b[n][0]) for n in common)
print(f"曲={len(common)}  増加イベント  旧={na}  新={nb}")
print(f"完全一致={na - len(shifts) - len(rem)}  時刻が動いた={len(shifts)}  消失={len(rem)}  追加={len(add)}")
d = sorted(s for s, *_ in shifts)
if d:
    print("ずれ(秒): 中央=%.3f 平均=%.3f 最大=%.3f" % (d[len(d) // 2], sum(d) / len(d), d[-1]))
    for th in (0.1, 0.5, 1.0, 2.0, 5.0):
        k = sum(1 for x in d if x > th)
        print("  |ずれ| > %.1fs : %d 件 (%.2f%%)" % (th, k, 100 * k / na))
shifts.sort(reverse=True)
print("\n| 曲 | カウンター | 値 | 旧frame | 新frame | ずれ(秒) |")
print("|---|---|---|---|---|---|")
for s, n, k, fa, fb in shifts[:15]:
    print("| %s | %s | %d | %d | %d | %.2f |" % (n.split("_2026")[0][:26], k[0], k[1], fa, fb, s))
print(f"\n消失（{len(rem)}件）:")
for n, k, f in rem: print("  %-28s %-10s value=%d frame=%d" % (n.split("_2026")[0][:28], k[0], k[1], f))
print(f"\n追加（{len(add)}件）:")
for n, k, f in add: print("  %-28s %-10s value=%d frame=%d" % (n.split("_2026")[0][:28], k[0], k[1], f))
