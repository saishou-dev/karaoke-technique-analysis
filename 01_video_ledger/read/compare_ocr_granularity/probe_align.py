#!/usr/bin/env python3
"""probe_align.py ＝ 桁の位置が固定か（左詰め／中央揃え／右詰め）を正解値つきで実測（2026-10-29）

probe_geometry.py の中央値は「どの数字が出ていたか」が混ざるため、桁の揃え方を決められない。
ここでは正解値（gt.csv）で 1桁・2桁に分けたうえで、数字全体の左端・中心・右端を測り、
  左詰め  ＝ 1桁と2桁の左端が一致
  中央揃え ＝ 1桁と2桁の中心が一致
  右詰め  ＝ 1桁と2桁の右端が一致
のどれかを曲×技法ごとに判定する。これが「固定ピッチで桁ごとに切る」条件の設計根拠になる。
"""
import argparse
import csv
import json
import os
import sys
import collections

import numpy as np

PREV = "/mnt/g/研究アーカイブ/成果物/動画解析/ocr_compare_20261018"
sys.path.insert(0, PREV)
import glyph_core as gc

COUNTERS = gc.COUNTERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev-work", default=os.path.join(PREV, "work"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    gt = {}
    with open(os.path.join(a.prev_work, "gt.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["certain"] == "1":
                gt[(r["video"], int(r["sample_idx"]), r["counter"])] = int(r["gt_value"])

    songs = [d for d in sorted(os.listdir(a.prev_work))
             if os.path.isdir(os.path.join(a.prev_work, d))]
    rows = []
    perdigit = collections.defaultdict(list)   # (song,counter,digitchar) -> [w]
    for s in songs:
        for c in COUNTERS:
            arr = np.load(os.path.join(a.prev_work, s, f"roi_{c}.npy"), mmap_mode="r")
            # 正解値ごとに、箱から求めた「数字全体の左端・右端」を集める
            by = collections.defaultdict(lambda: {"l": [], "r": [], "h": [], "y": [], "w": []})
            for i in range(arr.shape[0]):
                k = (s, i, c)
                if k not in gt:
                    continue
                g = np.asarray(arr[i]).max(axis=2)
                b = gc.glyph_boxes(g)
                if b is None:
                    continue
                v = gt[k]
                nd = len(str(v))
                if len(b) != nd:          # 箱の数が桁数と合わない＝切り出し自体が違う→除く
                    continue
                l = b[0][0]
                r = b[-1][0] + b[-1][2]
                by[v]["l"].append(l)
                by[v]["r"].append(r)
                by[v]["h"].append(float(np.median([x[3] for x in b])))
                by[v]["y"].append(float(np.median([x[1] for x in b])))
                by[v]["w"].append(float(np.median([x[2] for x in b])))
                for d, bb in zip(str(v), b):
                    perdigit[(s, c, d)].append(bb[2])
            for v in sorted(by):
                d = by[v]
                if len(d["l"]) < 20:
                    continue
                L = float(np.median(d["l"])); R = float(np.median(d["r"]))
                rows.append([s, c, v, len(str(v)), len(d["l"]),
                             round(L, 1), round(R, 1), round((L + R) / 2.0, 1),
                             round(R - L, 1),
                             round(float(np.median(d["h"])), 1),
                             round(float(np.median(d["y"])), 1),
                             round(float(np.percentile(d["l"], 95) - np.percentile(d["l"], 5)), 1),
                             round(float(np.percentile(d["r"], 95) - np.percentile(d["r"], 5)), 1)])

    with open(os.path.join(a.out, "align_report.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "counter", "gt_value", "ndigits", "n",
                    "left_med", "right_med", "center_med", "span",
                    "digit_h_med", "digit_y_med", "left_p95_p5", "right_p95_p5"])
        w.writerows(rows)

    # 判定＝曲×技法ごとに、1桁群の代表と2桁群の代表で左端・中心・右端の差を比べる
    jud = []
    agg = collections.defaultdict(lambda: {1: [], 2: []})
    for r in rows:
        agg[(r[0], r[1])][r[3]].append(r)
    for k, v in sorted(agg.items()):
        if not v[1] or not v[2]:
            continue
        def wmed(lst, col):
            vals = []
            for r in lst:
                vals += [r[col]] * r[4]
            return float(np.median(vals))
        l1, c1, r1 = wmed(v[1], 5), wmed(v[1], 7), wmed(v[1], 6)
        l2, c2, r2 = wmed(v[2], 5), wmed(v[2], 7), wmed(v[2], 6)
        dl, dc, dr = abs(l1 - l2), abs(c1 - c2), abs(r1 - r2)
        best = min((dl, "left"), (dc, "center"), (dr, "right"))[1]
        jud.append([k[0], k[1], round(l1, 1), round(c1, 1), round(r1, 1),
                    round(l2, 1), round(c2, 1), round(r2, 1),
                    round(dl, 1), round(dc, 1), round(dr, 1), best])
        print(f"{k[0][:10]} {k[1]:10s} 1桁(l,c,r)=({l1:5.1f},{c1:5.1f},{r1:5.1f}) "
              f"2桁=({l2:5.1f},{c2:5.1f},{r2:5.1f}) 差 l={dl:5.1f} c={dc:5.1f} r={dr:5.1f} → {best}")
    with open(os.path.join(a.out, "align_judgement.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "counter", "l1", "c1", "r1", "l2", "c2", "r2",
                    "d_left", "d_center", "d_right", "alignment"])
        w.writerows(jud)

    pd = []
    for (s, c, d), ws in sorted(perdigit.items()):
        pd.append([s, c, d, len(ws), round(float(np.median(ws)), 1),
                   round(float(np.percentile(ws, 5)), 1), round(float(np.percentile(ws, 95)), 1)])
    with open(os.path.join(a.out, "digit_width.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "counter", "digit", "n", "w_med", "w_p5", "w_p95"])
        w.writerows(pd)
    # 数字ごとの幅（全曲まとめ）
    allw = collections.defaultdict(list)
    for (s, c, d), ws in perdigit.items():
        allw[d] += list(ws)
    print("\n数字ごとの箱の幅（全曲・px）:")
    for d in sorted(allw):
        ws = allw[d]
        print(f"  {d}: n={len(ws):6d} 中央値={np.median(ws):5.1f} 5%={np.percentile(ws,5):5.1f} 95%={np.percentile(ws,95):5.1f}")
    json.dump({d: {"n": len(v), "w_med": float(np.median(v)),
                   "w_p5": float(np.percentile(v, 5)), "w_p95": float(np.percentile(v, 95))}
               for d, v in allw.items()},
              open(os.path.join(a.out, "digit_width_all.json"), "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
