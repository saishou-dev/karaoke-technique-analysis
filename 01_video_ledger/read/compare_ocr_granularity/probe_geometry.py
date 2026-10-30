#!/usr/bin/env python3
"""probe_geometry.py ＝ 数字の桁位置が固定かを実測し、条件Cの「固定ピッチの桁窓」を決める（2026-10-29）

背景＝2026-10-29 のゼミで「数字の大きさも規定、窓の大きさも決められる」という指摘を受けた。
    桁の位置が本当に固定なら、技法別の数字窓をさらに桁ごとに割った窓を作れる。
    まずそれが事実かを、2026-10-18 の比較実験で作った入力（技法別の数字窓）の上で実測する。

やること
  1. 既存の入力 ocr_compare_20261018/work/<曲>/roi_<技法>.npy を読む（書き換えない）。
  2. v0.8e と同一の glyph_boxes で数字の箱を切り、1箱（1桁）と2箱（2桁）の場合に分けて
     箱の x・y・幅・高さの分布を曲×技法ごとに集める。
  3. 桁の位置が固定か（分布の広がり）を report に出し、固定ピッチの桁窓を geometry.json に書く。

桁窓の決め方（条件C）
  左詰めか中央揃えかを 1桁の箱の中心と2箱の左桁の中心の差で判定し、
  2桁の場合の左桁・右桁の中央値位置を基準に、幅 = 桁幅の中央値 + 2*pad の窓を2枠置く。
  1桁の場合がどちらの枠に入るかも実測して記録する（＝固定ピッチの前提が成り立つかの証拠）。

出力（--out 配下）
  geometry.json         … 曲×技法ごとの桁窓（条件Cが使う）
  geometry_report.csv   … 箱の分布（中央値・5%点・95%点・件数）
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

PREV = "/mnt/g/研究アーカイブ/成果物/動画解析/ocr_compare_20261018"
sys.path.insert(0, PREV)
import glyph_core as gc

COUNTERS = gc.COUNTERS
PAD = 2


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev-work", default=os.path.join(PREV, "work"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    songs = [d for d in sorted(os.listdir(a.prev_work))
             if os.path.isdir(os.path.join(a.prev_work, d))]
    rep = []
    geo = {}
    for s in songs:
        geo[s] = {}
        for c in COUNTERS:
            arr = np.load(os.path.join(a.prev_work, s, f"roi_{c}.npy"), mmap_mode="r")
            h, w = arr.shape[1], arr.shape[2]
            one = []   # 1桁のとき、その箱
            two_l = []  # 2桁のとき、左の箱
            two_r = []  # 2桁のとき、右の箱
            for i in range(arr.shape[0]):
                g = np.asarray(arr[i]).max(axis=2)
                b = gc.glyph_boxes(g)
                if b is None:
                    continue
                if len(b) == 1:
                    one.append(b[0])
                else:
                    two_l.append(b[0])
                    two_r.append(b[1])
            one = np.asarray(one, dtype=np.float64).reshape(-1, 4)
            two_l = np.asarray(two_l, dtype=np.float64).reshape(-1, 4)
            two_r = np.asarray(two_r, dtype=np.float64).reshape(-1, 4)
            # 代表値
            def med(arrv, k):
                return q(arrv[:, k], 50) if len(arrv) else float("nan")
            dw = np.nanmedian([med(one, 2), med(two_l, 2), med(two_r, 2)])
            dh = np.nanmedian([med(one, 3), med(two_l, 3), med(two_r, 3)])
            yc = np.nanmedian([med(one, 1), med(two_l, 1), med(two_r, 1)])
            # 1桁の中心と2桁左桁の中心の差（＞0 なら1桁は右寄り＝中央揃えの疑い）
            cen_one = med(one, 0) + med(one, 2) / 2.0
            cen_l = med(two_l, 0) + med(two_l, 2) / 2.0
            cen_r = med(two_r, 0) + med(two_r, 2) / 2.0
            rep.append([s, c, h, w, len(one), len(two_l),
                        round(med(one, 0), 1), round(q(one[:, 0], 5), 1) if len(one) else "",
                        round(q(one[:, 0], 95), 1) if len(one) else "",
                        round(med(two_l, 0), 1), round(q(two_l[:, 0], 5), 1) if len(two_l) else "",
                        round(q(two_l[:, 0], 95), 1) if len(two_l) else "",
                        round(med(two_r, 0), 1), round(q(two_r[:, 0], 5), 1) if len(two_r) else "",
                        round(q(two_r[:, 0], 95), 1) if len(two_r) else "",
                        round(dw, 1), round(dh, 1), round(yc, 1),
                        round(cen_one - cen_l, 1) if len(one) and len(two_l) else "",
                        round(cen_r - cen_l, 1) if len(two_l) else ""])
            # 桁窓（左詰め前提＝2桁の左桁位置に左枠、右桁位置に右枠）。1桁しか出ない技法は
            # その位置を左枠にし、右枠はピッチの中央値（他技法から）で後から埋める。
            geo[s][c] = {
                "roi_h": int(h), "roi_w": int(w),
                "digit_w": None if np.isnan(dw) else float(dw),
                "digit_h": None if np.isnan(dh) else float(dh),
                "y": None if np.isnan(yc) else float(yc),
                "x_one": None if np.isnan(med(one, 0)) else float(med(one, 0)),
                "x_two_l": None if np.isnan(med(two_l, 0)) else float(med(two_l, 0)),
                "x_two_r": None if np.isnan(med(two_r, 0)) else float(med(two_r, 0)),
                "n_one": int(len(one)), "n_two": int(len(two_l)),
            }
            print(f"{s[:10]} {c:10s} h={h} w={w} one={len(one):5d} two={len(two_l):5d} "
                  f"x1={med(one,0):5.1f} xl={med(two_l,0):5.1f} xr={med(two_r,0):5.1f} "
                  f"dw={dw:4.1f} dh={dh:4.1f} y={yc:4.1f} "
                  f"cen(one-l)={cen_one-cen_l if len(one) and len(two_l) else float('nan'):6.1f} "
                  f"pitch={cen_r-cen_l if len(two_l) else float('nan'):5.1f}", flush=True)

    with open(os.path.join(a.out, "geometry_report.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "counter", "roi_h", "roi_w", "n_one", "n_two",
                    "x_one_med", "x_one_p5", "x_one_p95",
                    "x_two_l_med", "x_two_l_p5", "x_two_l_p95",
                    "x_two_r_med", "x_two_r_p5", "x_two_r_p95",
                    "digit_w_med", "digit_h_med", "y_med",
                    "center_one_minus_left", "pitch_center"])
        w.writerows(rep)
    json.dump(geo, open(os.path.join(a.out, "geometry_raw.json"), "w"), ensure_ascii=False, indent=1)
    print("wrote geometry_report.csv / geometry_raw.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
