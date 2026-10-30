#!/usr/bin/env python3
"""build_slots.py ＝ 条件C（固定ピッチの桁窓）の校正値を実測から決める（2026-10-29）

probe_align.py の実測で、数字は技法別の窓の中で**中央揃え**（1桁と2桁で中心が一致・差0.0〜5.0px、
左端の差10〜21px・右端の差11〜20px）であり、桁の箱の幅は数字によらずほぼ一定
（中央値17〜24px・95%点22〜27px）であることが分かった。
したがって「桁ごとの固定窓」は次の2つの形でしか作れない。

  条件C  ＝ 固定ピッチ・2枠固定。数字の中心 cx を基準に cx±P/2 の2枠を**桁数に関わらず常に**置く。
            先生の指摘（数字の大きさも窓の大きさも固定できる）をそのまま実装した条件。
  条件C2 ＝ 桁ごとの窓。先に桁数（1 or 2）をインクの横幅から決め、その桁数に応じた固定窓を置く。
            中央揃えを踏まえて条件Cを直した条件。

この台本は cx・cy・ピッチ P・窓の大きさ（w,h）を、曲×技法ごとに正解値つきの実測から出す。

出力: --out/slots.json
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

PREV = "/mnt/g/研究アーカイブ/成果物/動画解析/ocr_compare_20261018"
sys.path.insert(0, PREV)
import glyph_core as gc

COUNTERS = gc.COUNTERS
PAD_X = 4       # 桁窓の横の余白（片側）
PAD_Y = 4       # 桁窓の縦の余白（片側）


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

    stat = collections.defaultdict(lambda: {"cx": [], "cy": [], "dw": [], "dh": [],
                                            "pitch": [], "ink1": [], "ink2": []})
    for s in songs:
        for c in COUNTERS:
            arr = np.load(os.path.join(a.prev_work, s, f"roi_{c}.npy"), mmap_mode="r")
            for i in range(arr.shape[0]):
                k = (s, i, c)
                if k not in gt:
                    continue
                g = np.asarray(arr[i]).max(axis=2)
                b = gc.glyph_boxes(g)
                if b is None:
                    continue
                nd = len(str(gt[k]))
                if len(b) != nd:
                    continue
                d = stat[(s, c)]
                l = b[0][0]
                r = b[-1][0] + b[-1][2]
                d["cx"].append((l + r) / 2.0)
                ys = [x[1] for x in b]
                hs = [x[3] for x in b]
                d["cy"].append(float(np.mean([y + h / 2.0 for y, h in zip(ys, hs)])))
                d["dw"] += [x[2] for x in b]
                d["dh"] += hs
                if nd == 2:
                    d["pitch"].append((b[1][0] + b[1][2] / 2.0) - (b[0][0] + b[0][2] / 2.0))
                    d["ink2"].append(r - l)
                else:
                    d["ink1"].append(r - l)

    # 全曲まとめの代表値（桁窓の大きさとピッチは全体で1つに固定する＝「窓の大きさを決める」）
    allw = [x for k in stat for x in stat[k]["dw"]]
    allh = [x for k in stat for x in stat[k]["dh"]]
    allp = [x for k in stat for x in stat[k]["pitch"]]
    ink1 = [x for k in stat for x in stat[k]["ink1"]]
    ink2 = [x for k in stat for x in stat[k]["ink2"]]
    SW = int(round(np.percentile(allw, 95))) + 2 * PAD_X
    SH = int(round(np.percentile(allh, 95))) + 2 * PAD_Y
    P = float(np.median(allp))
    # 桁数の判定しきい値（条件C2）＝1桁のインク幅と2桁のインク幅の中点
    thr = (float(np.percentile(ink1, 95)) + float(np.percentile(ink2, 5))) / 2.0
    print(f"桁窓 w={SW} h={SH}  ピッチ P={P:.1f}  桁幅95%={np.percentile(allw,95):.1f} "
          f"桁高95%={np.percentile(allh,95):.1f}")
    print(f"インク幅 1桁 中央値={np.median(ink1):.1f} 95%={np.percentile(ink1,95):.1f} / "
          f"2桁 5%={np.percentile(ink2,5):.1f} 中央値={np.median(ink2):.1f} → 桁数判定しきい値={thr:.1f}")

    slots = {"slot_w": SW, "slot_h": SH, "pitch": P, "ndigit_thr": thr,
             "pad_x": PAD_X, "pad_y": PAD_Y, "per": {}}
    for s in songs:
        slots["per"][s] = {}
        for c in COUNTERS:
            d = stat[(s, c)]
            if not d["cx"]:
                print(f"  !! {s} {c} 実測なし")
                continue
            cx = float(np.median(d["cx"]))
            cy = float(np.median(d["cy"]))
            slots["per"][s][c] = {"cx": cx, "cy": cy, "n": len(d["cx"]),
                                  "n2": len(d["pitch"])}
            print(f"  {s[:10]} {c:10s} cx={cx:5.1f} cy={cy:5.1f} n={len(d['cx']):5d} n2={len(d['pitch']):5d}")
    json.dump(slots, open(os.path.join(a.out, "slots.json"), "w"), ensure_ascii=False, indent=1)
    print("wrote slots.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
