#!/usr/bin/env python3
"""gran_core.py ＝ 切り出し粒度を変えた比較の共通部品（2026-10-29）

条件
  A ＝ 帯（画面下部の加点表示領域まるごと・2480×120）。1コマに6つの数字が同時に写る。
  B ＝ 技法別の数字窓（カウンターごとに1枠・67×70〜90）。2026-10-18 の比較実験と同一の入力。
  C ＝ 固定ピッチの桁窓（33×47 を2枠・桁数に関わらず数字の中心 cx から ±P/2 に固定）。
  C2＝ 桁ごとの窓（先に桁数を測り、1桁なら中心に1枠・2桁なら ±P/2 に2枠）。

条件Aで「どの数字がどのカウンターのものか」を決める規則
  位置が取れる手法（見本照合・生ピクセル最近傍・Tesseract の tsv・EasyOCR readtext・PaddleOCR 一式）
    ＝ 数字のかたまりの中心 x が、そのカウンターの数字窓の x 範囲に入るものを採る。
      どのカウンターの範囲にも入らないかたまりは「場所外の読み」として数える。
  位置が取れない手法（Tesseract の文字列・EasyOCR recognize・PaddleOCR 認識のみ）
    ＝ 出力文字列の数字の並びを左から順に6つのカウンターへ割り当てる。
      7個目以降は「場所外の読み」。足りない分は「読まない」。

正本（read_counters*.py）と 2026-10-18 の比較実験の台本は読み取り専用で参照し、書き換えない。
"""
import os
import re
import sys

import cv2
import numpy as np

PREV = os.environ.get("KV_PREV", "/mnt/g/研究アーカイブ/成果物/動画解析/ocr_compare_20261018")
sys.path.insert(0, PREV)
import glyph_core as gc   # noqa: E402

COUNTERS = gc.COUNTERS
THRESHOLDS = gc.THRESHOLDS
NORM = gc.NORM
DIGITS = re.compile(r"\d+")

# v0.8e の「ごみ上限 1200」は数字窓1枚（67x70〜90）の中の話で、帯には当てはめられない。
# 実測（2026-10-29・スターダスト f14000）＝帯では背景のパネルそのものが閾値を超えて1つの大きな
# 連結成分になり、ごみの合計が T=200 で 38,472・T=170 で 63,166 に達する。面積比で比例配分した
# 上限（1200 x 297,600/32,160 ≒ 11,100）を置くと、数字が現れる低い閾値（T=130）が全部弾かれる。
# したがって条件A ではごみ上限を外す（＝現行手法に最も有利な形で帯を読ませる）。
BAND_JUNK_LIMIT = None


def prep(crop, scale=3):
    """既製OCR向けの前処理＝V → 3倍拡大 → 大津の二値化 → 反転（白地に黒字）。2026-10-18 と同一。"""
    g = crop.max(axis=2) if crop.ndim == 3 else crop
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 255 - b


def parse_runs(s):
    """文字列から数字の並びを左から順に取り出す。['10','2','0',...]"""
    if s is None:
        return []
    return DIGITS.findall(str(s))


def parse_digits(s, limit=6):
    """2026-10-18 と同一＝数字を全部つないで整数化。空なら None。"""
    m = parse_runs(s)
    if not m:
        return None
    t = "".join(m)
    if len(t) > limit:
        t = t[:limit]
    return int(t)


# ---------------- 条件A：帯の上で数字のかたまりを探す ----------------

def band_groups(band_v, junk_limit=BAND_JUNK_LIMIT):
    """帯の V 画像から、閾値ごとに「数字のかたまり」を返す。
    箱の条件・桁間の条件は v0.8e と同一。ごみ上限だけ帯の広さに比例配分した。
    返り値 ＝ [(T, [ [box,...], ... ]), ...]（閾値の優先順）"""
    out = []
    for T in THRESHOLDS:
        m = (band_v > T).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        boxes = []
        junk = 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if 28 <= h <= 58 and 3 <= w <= 44 and area >= 50:
                boxes.append((int(x), int(y), int(w), int(h)))
            elif area > 30:
                junk += int(area)
        if (junk_limit is not None and junk > junk_limit) or not boxes:
            continue
        boxes.sort(key=lambda b: b[0])
        groups = []
        cur = [boxes[0]]
        for b in boxes[1:]:
            p = cur[-1]
            gap = b[0] - (p[0] + p[2])
            if gap <= 14 and abs(p[3] - b[3]) <= 8 and abs(p[1] - b[1]) <= 6 and gap >= gc.MIN_GAP2:
                cur.append(b)
            else:
                groups.append(cur)
                cur = [b]
        groups.append(cur)
        out.append((T, [g for g in groups if len(g) <= 2]))
    return out


def band_assign(groups, roi):
    """かたまりの中心 x を、そのカウンターの数字窓の x 範囲に当てる。
    返り値 ＝ ({counter: [box,...]}, 場所外のかたまり数)"""
    hit = {}
    stray = 0
    for g in groups:
        l = g[0][0]
        r = g[-1][0] + g[-1][2]
        cx = (l + r) / 2.0
        put = None
        for c in COUNTERS:
            x1, _y1, x2, _y2 = roi[c]
            if x1 <= cx < x2:
                put = c
                break
        if put is None:
            stray += 1
        elif put not in hit:
            hit[put] = g
    return hit, stray


def read_band_tpl(band_bgr, roi, tpl, how="ncc"):
    """条件A＝帯の上で現行手法（または生ピクセル最近傍）を動かす。
    閾値の優先順は v0.8e と同じで、カウンターごとに「2桁が取れた閾値」を優先する。
    返り値 ＝ ({counter: value or None}, 場所外のかたまり数)"""
    v = band_bgr.max(axis=2) if band_bgr.ndim == 3 else band_bgr
    best1, best2 = {}, {}
    stray = 0
    for ti, (_T, groups) in enumerate(band_groups(v)):
        hit, st = band_assign(groups, roi)
        if ti == 0:
            stray = st
        for c, g in hit.items():
            if len(g) == 2 and c not in best2:
                best2[c] = g
            elif len(g) == 1 and c not in best1:
                best1[c] = g
    out = {}
    for c in COUNTERS:
        g = best2.get(c) or best1.get(c)
        if g is None:
            out[c] = None
            continue
        ds = []
        for b in g:
            if how == "ncc":
                d = tpl.classify_ncc(gc.norm_glyph(v, b))
            else:
                d = tpl.classify_nn_l2(gc.raw_glyph(v, b))
            if d is None:
                ds = None
                break
            ds.append(d)
        out[c] = None if not ds else (ds[0] if len(ds) == 1 else ds[0] * 10 + ds[1])
    return out, stray


def assign_by_x(items, roi, scale=1.0):
    """items ＝ [(cx, text), ...]（cx は帯の座標・scale で割って戻す）。
    返り値 ＝ ({counter: text}, 場所外の数)"""
    hit = {}
    stray = 0
    for cx, txt in sorted(items, key=lambda t: t[0]):
        x = cx / scale
        put = None
        for c in COUNTERS:
            x1, _y1, x2, _y2 = roi[c]
            if x1 <= x < x2:
                put = c
                break
        if put is None:
            stray += 1
        elif put not in hit:
            hit[put] = txt
        else:
            hit[put] = str(hit[put]) + str(txt)
    return hit, stray


def assign_in_order(text):
    """位置が取れない手法＝数字の並びを左から順に6カウンターへ。"""
    runs = parse_runs(text)
    hit = {}
    for j, t in enumerate(runs[:len(COUNTERS)]):
        hit[COUNTERS[j]] = t
    return hit, max(0, len(runs) - len(COUNTERS))


# ---------------- 条件C・C2：桁窓 ----------------

def slot_boxes(cx, cy, sw, sh, pitch, ndigits):
    """桁窓の座標（整数・左上と右下）。中央揃えなので桁数で位置が変わる。"""
    if ndigits == 1:
        cs = [cx]
    else:
        cs = [cx - pitch / 2.0, cx + pitch / 2.0]
    out = []
    for c in cs:
        x1 = int(round(c - sw / 2.0))
        y1 = int(round(cy - sh / 2.0))
        out.append((x1, y1, x1 + sw, y1 + sh))
    return out


def crop_pad(img, box):
    """はみ出す分は端の値で埋めて切り出す（固定の大きさを必ず返す）。"""
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    px1, py1 = max(0, -x1), max(0, -y1)
    px2, py2 = max(0, x2 - w), max(0, y2 - h)
    sub = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if px1 or py1 or px2 or py2:
        pad = [(py1, py2), (px1, px2)] + ([(0, 0)] if sub.ndim == 3 else [])
        sub = np.pad(sub, pad, mode="edge")
    return sub


def ink_extent(roi_v):
    """数字のインクの横幅（条件C2の桁数判定に使う）。閾値は v0.8e と同じ並びで上から試す。"""
    for T in THRESHOLDS:
        m = roi_v > T
        cols = m.sum(axis=0)
        idx = np.nonzero(cols >= 3)[0]
        if len(idx) == 0:
            continue
        ext = int(idx[-1] - idx[0] + 1)
        if 8 <= ext <= 70:
            return ext
    return None


def ndigits_of(roi_v, thr):
    e = ink_extent(roi_v)
    if e is None:
        return 1
    return 2 if e >= thr else 1


def classify_slot_tpl(slot_bgr, tpl, how="ncc"):
    """条件C・C2の現行手法＝桁窓をまるごと正規化して見本照合（連結成分での切り出しをしない）。
    「窓が固定なら切り出しは要らないのか」を測るための条件。"""
    v = slot_bgr.max(axis=2) if slot_bgr.ndim == 3 else slot_bgr
    g = v.astype(np.float32)
    if how == "ncc":
        g = (g - g.min()) / max(1.0, (g.max() - g.min()))
    else:
        g = g / 255.0
    g = cv2.resize(g, NORM)
    return tpl.classify_ncc(g) if how == "ncc" else tpl.classify_nn_l2(g)


def combine_slots(digits):
    """桁窓ごとの読み（None＝読まない）を左から並べて整数にする。全部 None なら None。"""
    ds = [d for d in digits if d is not None]
    if not ds:
        return None
    v = 0
    for d in ds:
        v = v * 10 + int(d)
    return v
