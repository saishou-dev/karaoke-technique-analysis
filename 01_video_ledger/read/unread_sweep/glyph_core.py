#!/usr/bin/env python3
"""glyph_core.py ＝ 読取器 v0.8e の「数字の読み取り」部分を比較実験用に切り出した参照実装（2026-10-18）

目的＝既製OCRと同じ土俵で現行手法（自作の見本照合＝NCC）を測るため、v0.8e の
  glyph_boxes（数字の箱を切る）→ norm_glyph（箱を正規化）→ classify（0〜9へ判定）
  → band_frame_range（帯消失ガード）→ detect_increments（6コマ安定＋非減少で軌跡を1本選ぶ）
を、動画入出力から切り離した関数として置き直したもの。

**正本は書き換えない**。本ファイルは
  01_video_ledger/read/read_counters_wao_v08e_20261015.py（皮）
  01_video_ledger/read/read_counters_wao_single_20261015.py（単一ファイル版・293行）
から定数・判定式をそのまま写した読み取り専用の複製であり、正本の挙動は一切変えない。
同一性は identity_check.py が1曲の全コマ走査で counter_events.csv の一致を実測して示す。

用語（外向きの標準語＝research/用語対応表_2026-10-15.md）
  帯           ＝ 画面下部の加点表示領域（関心領域 ROI）
  見本照合NCC  ＝ 正規化相互相関によるテンプレートマッチング
  軌跡の確定   ＝ 時系列の整数列から単調非減少の1本を選ぶ後処理
"""
import os
import cv2
import numpy as np

COUNTERS = ["shakuri", "kobushi", "fall", "vibrato", "accent", "hammering"]

# ---- v0.8e の定数（正本からの写し）----
THRESHOLDS = (215, 200, 185, 228, 170, 150, 130)
NORM = (28, 40)          # 正規化後の大きさ（横28・縦40）
NCC_MIN = 0.75           # 見本照合の下限（これに届かなければ「読まない」）
STABLE_MIN = 6           # 同一値がこの回数読めたら安定値
MIN_GAP2 = 5             # v0.8e(A) 2桁として採用する桁間の最小隙間(px)
BAND_MIN = 0.5           # v0.8e(B) 帯ありと見なす同時読取数の下限
BAND_W_FRAMES = 150      # v0.8e(B) 移動平均の半幅（元コマ単位・150≒5秒）


def glyph_boxes(roi_gray, min_gap2=MIN_GAP2):
    """関心領域を7段階の閾値で白黒化し、数字の大きさの連結成分だけを拾う。
    2桁（2箱）が取れた閾値を優先し、取れなければ1桁（1箱）を使う。
    桁間の隙間が min_gap2 未満の2箱は「1文字が割れた形」として採らない（v0.8e(A)）。"""
    best1 = None
    best2 = None
    for T in THRESHOLDS:
        m = (roi_gray > T).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        boxes = []
        junk = 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if 28 <= h <= 58 and 3 <= w <= 44 and area >= 50:
                boxes.append((x, y, w, h))
            elif area > 30:
                junk += area
        if not (1 <= len(boxes) <= 2 and junk <= 1200):
            continue
        boxes.sort(key=lambda b: b[0])
        if len(boxes) == 2:
            gap = boxes[1][0] - (boxes[0][0] + boxes[0][2])
            hdiff = abs(boxes[0][3] - boxes[1][3])
            ydiff = abs(boxes[0][1] - boxes[1][1])
            if gap > 14 or gap < min_gap2 or hdiff > 8 or ydiff > 6:
                continue
            if best2 is None:
                best2 = boxes
        elif best1 is None:
            best1 = boxes
    return best2 if best2 is not None else best1


def norm_glyph(roi_gray, box):
    """箱の周り2pxを足して切り、min-max で 0〜1 に伸ばしてから 28×40 に縮める。"""
    x, y, w, h = box
    pad = 2
    y0, y1 = max(0, y - pad), min(roi_gray.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(roi_gray.shape[1], x + w + pad)
    g = roi_gray[y0:y1, x0:x1].astype(np.float32)
    g = (g - g.min()) / max(1.0, (g.max() - g.min()))
    return cv2.resize(g, NORM)


def raw_glyph(roi_gray, box):
    """min-max 正規化をしない版（対照5＝最近傍の入力）。0〜1に割るだけで伸ばさない。"""
    x, y, w, h = box
    pad = 2
    y0, y1 = max(0, y - pad), min(roi_gray.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(roi_gray.shape[1], x + w + pad)
    g = roi_gray[y0:y1, x0:x1].astype(np.float32) / 255.0
    return cv2.resize(g, NORM)


class Templates:
    """0〜9の見本（テンプレ）。NCC 用（平均0・ノルム1）と生ピクセル用の両方を持つ。"""

    def __init__(self, reps_npy, labels_json=None):
        import json
        reps = np.load(reps_npy)
        self.reps = reps
        flat = []
        for r in reps:
            v = r.ravel() - r.mean()
            flat.append(v / max(1e-6, np.linalg.norm(v)))
        self.reps_flat = np.stack(flat)
        self.reps_raw = np.stack([r.ravel() for r in reps])
        if labels_json is None:
            labels_json = os.path.splitext(reps_npy)[0] + "_labels.json"
        L = json.load(open(labels_json))
        self.labels = {int(k): v for k, v in L.items()}

    def classify_ncc(self, gl):
        """v0.8e と同一：NCC 最良が NCC_MIN 未満なら None（＝読まない）。"""
        v = gl.ravel() - gl.mean()
        n = np.linalg.norm(v)
        if n < 1e-6:
            return None
        scores = self.reps_flat @ (v / n)
        bi = int(np.argmax(scores))
        if scores[bi] < NCC_MIN:
            return None
        return self.labels.get(bi)

    def classify_nn_l2(self, gl):
        """対照5：生ピクセルのユークリッド距離で最近傍。棄権しない（必ず答える）。"""
        v = gl.ravel()
        d = ((self.reps_raw - v) ** 2).sum(axis=1)
        bi = int(np.argmin(d))
        return self.labels.get(bi)


def read_value_ncc(roi_bgr, tpl):
    """1枚の関心領域から v0.8e と同一手順で整数を読む。読めなければ None。"""
    g = roi_bgr.max(axis=2)          # V=max(B,G,R)（色付き数字に強い・v0.8d）
    boxes = glyph_boxes(g)
    if boxes is None:
        return None
    digits = []
    for b in boxes:
        d = tpl.classify_ncc(norm_glyph(g, b))
        if d is None:
            return None
        digits.append(d)
    return digits[0] if len(digits) == 1 else digits[0] * 10 + digits[1]


def read_value_nn(roi_bgr, tpl):
    """対照5：箱の切り出しは同じ・分類だけ生ピクセル最近傍（正規化なし・棄権なし）。"""
    g = roi_bgr.max(axis=2)
    boxes = glyph_boxes(g)
    if boxes is None:
        return None                   # 箱が切れない場合だけは答えようがない
    digits = []
    for b in boxes:
        d = tpl.classify_nn_l2(raw_glyph(g, b))
        if d is None:
            return None
        digits.append(d)
    return digits[0] if len(digits) == 1 else digits[0] * 10 + digits[1]


# ---------------- 後処理（全候補に同一適用） ----------------

def band_trim(values, nsamples, band_w, band_min=BAND_MIN):
    """v0.8e(B) 帯消失ガード。values: {counter: [(idx, val), ...]}（idx＝抽出列の添字）。"""
    cnt = {}
    for name in values:
        for i, _v in values[name]:
            cnt[i] = cnt.get(i, 0) + 1
    cum = [0] * (nsamples + 1)
    for i in range(nsamples):
        cum[i + 1] = cum[i] + cnt.get(i, 0)

    def sm(i):
        a = max(0, i - band_w)
        b = min(nsamples, i + band_w + 1)
        return (cum[b] - cum[a]) / max(1, b - a)

    bs = 0
    while bs < nsamples and sm(bs) < band_min:
        bs += 1
    be = nsamples - 1
    while be >= 0 and sm(be) < band_min:
        be -= 1
    if bs > be:
        return values, (None, None)
    out = {k: [(i, v) for i, v in vs if bs <= i <= be] for k, vs in values.items()}
    return out, (bs, be)


def detect_increments(seq, stable_min=STABLE_MIN):
    """6コマ安定＋減らない制約（重み付き最長非減少部分列）で軌跡を1本選ぶ。
    seq: [(idx, value), ...]（読めたコマだけ）。返り値＝[(idx, value, delta), ...], 安定ラン数"""
    runs = []
    run_v = None
    for fr, v in seq:
        if run_v is not None and v == run_v:
            runs[-1][2] += 1
        else:
            runs.append([fr, v, 1])
            run_v = v
    runs = [r for r in runs if r[2] >= stable_min]
    n = len(runs)
    accepted = []
    if n:
        dp = [0.0] * n
        prev = [-1] * n
        for i in range(n):
            dp[i] = runs[i][2]
            for j in range(i):
                if runs[j][1] <= runs[i][1] and dp[j] + runs[i][2] > dp[i]:
                    dp[i] = dp[j] + runs[i][2]
                    prev[i] = j
        best = int(np.argmax(dp))
        path = []
        while best != -1:
            path.append(best)
            best = prev[best]
        path.reverse()
        cur = None
        for idx in path:
            fr, v, wgt = runs[idx]
            if cur is None:
                accepted.append((fr, v, 0))
                cur = v
            elif v > cur:
                accepted.append((fr, v, v - cur))
                cur = v
    return accepted, n
