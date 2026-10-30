#!/usr/bin/env python3
"""fix_b_boxes.py ＝ glyph_boxes（v0.8e の箱の切り出し）を、値を差し替えられる形と、段階の診断付きの形で置き直したもの（2026-10-30）

既定値（DEFAULT）は glyph_core.glyph_boxes の定数そのまま。既定値で gc.glyph_boxes と全件同じ箱を返すことを
fix_b_diag.py が実測して記録する。正本・glyph_core は書き換えない。

段階（1つの明るさの区切りごと・処理の順番）
  0 二値化      ＝ 白黒にした後、面積30を超える白の塊が1つも無い
  1 大きさ      ＝ 塊はあるが、数字の大きさ（高さ28〜58・幅3〜44・面積50以上）の塊が0個
  2 個数        ＝ 数字の大きさの塊が3個以上
  3 ごみ上限    ＝ 数字の大きさ以外の塊（面積30超）の面積合計が1200を超える
  4 2桁の並び   ＝ 2個取れたが、隙間（5〜14）・高さの差（8以下）・縦のずれ（6以下）のどれかを外れる
  5 成功
"""
import cv2
import numpy as np

DEFAULT = dict(thresholds=(215, 200, 185, 228, 170, 150, 130), hmin=28, hmax=58, wmin=3, wmax=44,
               amin=50, junk_area=30, junk_max=1200, gap_max=14, gap_min=5, hdiff=8, ydiff=6)
STAGES = ["0 二値化", "1 大きさ", "2 個数", "3 ごみ上限", "4 2桁の並び", "5 成功"]


def glyph_boxes_p(roi_gray, P=DEFAULT, diag=False):
    best1 = None
    best2 = None
    st_list = []
    comps = []
    for T in P["thresholds"]:
        m = (roi_gray > T).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        boxes = []
        junk = 0
        nbig = 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area > P["junk_area"]:
                nbig += 1
                if diag:
                    comps.append((int(T), int(x), int(y), int(w), int(h), int(area)))
            if P["hmin"] <= h <= P["hmax"] and P["wmin"] <= w <= P["wmax"] and area >= P["amin"]:
                boxes.append((x, y, w, h))
            elif area > P["junk_area"]:
                junk += area
        if not (1 <= len(boxes) <= 2 and junk <= P["junk_max"]):
            if diag:
                if len(boxes) == 0:
                    st_list.append((0 if nbig == 0 else 1, len(boxes), junk))
                elif len(boxes) > 2:
                    st_list.append((2, len(boxes), junk))
                else:
                    st_list.append((3, len(boxes), junk))
            continue
        boxes.sort(key=lambda b: b[0])
        if len(boxes) == 2:
            gap = boxes[1][0] - (boxes[0][0] + boxes[0][2])
            hd = abs(boxes[0][3] - boxes[1][3])
            yd = abs(boxes[0][1] - boxes[1][1])
            if gap > P["gap_max"] or gap < P["gap_min"] or hd > P["hdiff"] or yd > P["ydiff"]:
                if diag:
                    st_list.append((4, len(boxes), junk))
                continue
            if best2 is None:
                best2 = boxes
        elif best1 is None:
            best1 = boxes
        if diag:
            st_list.append((5, len(boxes), junk))
    res = best2 if best2 is not None else best1
    if diag:
        return res, st_list, comps
    return res


def size_reasons(c, P=DEFAULT):
    """数字の大きさの条件のうち、外れたものの一覧（塊1つ）。"""
    _T, _x, _y, w, h, area = c
    r = []
    if h < P["hmin"]:
        r.append("高さ<28")
    if h > P["hmax"]:
        r.append("高さ>58")
    if w < P["wmin"]:
        r.append("幅<3")
    if w > P["wmax"]:
        r.append("幅>44")
    if area < P["amin"]:
        r.append("面積<50")
    return r
