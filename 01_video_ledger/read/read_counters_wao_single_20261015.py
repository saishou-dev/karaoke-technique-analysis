#!/usr/bin/env python3
"""読取器 WAO版 v0.8e 単一ファイル版（2026-10-15・説明用）

これ1本で「録画の加点カウンターを読む」工程が全部読める形に書き下ろしたもの。
出力は read_counters_wao_v08e_20261015.py（文字列置換で本体 read_counters.py を改変して exec する皮）と
同一（counter_events.csv／.json のバイト一致を 10/14 の帯動画で確認済み・下の「同一性」参照）。

中身の出自＝3つを素直に1本へ合わせたもの:
  (1) 本体 read_counters.py（v0.8・2026-08-08）… 数字の切り出し→NCC分類→整数タイムライン→WLISで軌跡確定
  (2) WAO版 v0.8d（read_counters_wao.py・2026-09-12）… ラベルjson差し替え・V=max(B,G,R)・低閾値150/130・2桁優先
  (3) v0.8e の歯止め2つ（2026-10-15）… (A) 2桁の桁間下限 MIN_GAP2  (B) 帯が消えた後を読まない帯消失ガード

読む順＝glyph_boxes（数字の箱を切る）→ norm_glyph（箱を正規化）→ classify（0〜9に判定）→ detect_increments（増えた瞬間＝加点）。
band_frame_range は (B) の歯止め。

使い方: python read_counters_wao_single_20261015.py <band.mp4> <outdir> <reps.npy> [roi.json]
  reps.npy と同じ場所の <reps>_labels.json があればクラスタ→数字の対応をそれで置き換える。
  環境変数 KV_MIN_GAP2（既定5）・KV_BAND_MIN（既定0.5）・KV_BAND_W（既定150）で歯止めを調整できる。

同一性（2026-10-15 実測）＝10/14 の帯動画で v0.8e（皮）と本ファイルの counter_events.csv・counter_events.json の
md5 が一致することを確認した（記録＝hisho\research\説明_録画からの検出_2026-10-15.md）。
"""
import cv2
import numpy as np
import sys
import os
import csv
import json

VIDEO = sys.argv[1]
OUTDIR = sys.argv[2]
REPS_NPY = sys.argv[3]
ROI_OVERRIDE = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else {}

# テンプレのクラスタ index → 数字（None=ノイズ）。harvest.py が作ったモンタージュに人手でラベルを付けたもの。
# <reps>_labels.json があればそれで置き換える（WAO版＝cluster_reps_wao_ext_labels.json）。
LABELS_DEFAULT = {0: 0, 1: 2, 2: 3, 3: 1, 4: 2, 5: 6, 6: 1, 7: 2, 8: 1, 9: 7,
                  10: 4, 11: 5, 12: 6, 13: 9, 14: None, 15: 5, 16: 0, 17: 0, 18: None,
                  19: 2, 20: 1, 21: 6, 22: 1, 23: 0, 24: 2, 25: 2, 26: 2, 27: 2,
                  28: None, 29: 6, 30: 2, 31: 8, 32: 2, 33: 4, 34: 0, 35: 4,
                  36: None, 37: 7, 38: None, 39: 8}
_LABELS_JSON = os.path.splitext(REPS_NPY)[0] + "_labels.json"
if os.path.exists(_LABELS_JSON):
    LABELS = {int(k): v for k, v in json.load(open(_LABELS_JSON)).items()} or LABELS_DEFAULT
else:
    LABELS = LABELS_DEFAULT

# 帯 2480×120 内の各カウンターの数字の位置（既定値）。曲ごとの較正値 roi.json があればそれで上書きする。
DIGIT_ROIS = {
    "shakuri": (245, 85, 350, 148),
    "kobushi": (565, 85, 690, 148),
    "fall":    (925, 85, 1020, 148),
    "vibrato": (1230, 85, 1460, 148),
}
for k, v in ROI_OVERRIDE.items():
    DIGIT_ROIS[k] = tuple(v)

THRESHOLDS = (215, 200, 185, 228, 170, 150, 130)  # v0.8d で低閾値 150・130 を追加
NORM = (28, 40)        # 正規化後の大きさ（横28・縦40）
NCC_MIN = 0.75         # テンプレ照合の下限
STABLE_MIN = 6         # 同一値がこの回数読めたら安定値
MIN_GAP2 = int(os.environ.get("KV_MIN_GAP2", "5"))      # v0.8e(A) 2桁として採用する桁間の最小隙間(px)
BAND_MIN = float(os.environ.get("KV_BAND_MIN", "0.5"))  # v0.8e(B) 帯ありと見なす同時読取数の下限
BAND_W = int(os.environ.get("KV_BAND_W", "150"))        # v0.8e(B) 移動平均の半幅（フレーム・150≒5秒）


def glyph_boxes(roi_gray):
    # 関心領域を7段階の閾値で白黒化し、連結成分のうち数字の大きさの箱だけを拾う。
    # 2桁（2箱）が取れた閾値を優先し、取れなければ1桁（1箱）を使う（十の位の取りこぼし対策・v0.8d）。
    # 桁間の隙間が MIN_GAP2 未満の2箱は「1文字が割れた形」として採らない（v0.8e(A)）。
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
            # v0.8d は gap>=-4 を全て2桁として採用していた。gap が 0 付近＝1文字が2つの連結成分に
            # 割れた形なので、この閾値は2桁として採用せず、他の閾値の1箱候補へ委ねる
            # （10/14 きらり fall=11 の原因）。連結して1文字として読み直す案も試したが、
            # 加点フラッシュ中の字形はどの閾値でも NCC 0.29〜0.50 でテンプレに合わず不採用（10/15 実測）。
            if gap > 14 or gap < MIN_GAP2 or hdiff > 8 or ydiff > 6:
                continue
            if best2 is None:
                best2 = boxes
        elif best1 is None:
            best1 = boxes
    return best2 if best2 is not None else best1


def norm_glyph(roi_gray, box):
    # 箱の周り2pxを足して切り、min-max で 0〜1 に伸ばしてから 28×40 に縮める。
    # 画面の明るさ・加点フラッシュの明暗に依らない形にするための正規化。
    x, y, w, h = box
    pad = 2
    y0, y1 = max(0, y - pad), min(roi_gray.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(roi_gray.shape[1], x + w + pad)
    g = roi_gray[y0:y1, x0:x1].astype(np.float32)
    g = (g - g.min()) / max(1.0, (g.max() - g.min()))
    return cv2.resize(g, NORM)


# 0〜9の見本（テンプレ）を平均0・ノルム1に揃えておく。以後の照合は内積＝NCC で済む。
reps = np.load(REPS_NPY)
reps_flat = []
for r in reps:
    v = r.ravel() - r.mean()
    reps_flat.append(v / max(1e-6, np.linalg.norm(v)))
reps_flat = np.stack(reps_flat)


def classify(gl):
    # 正規化した1文字を全テンプレと NCC（正規化相互相関）で照合し、最良のクラスタの数字を返す。
    # 最良でも NCC_MIN=0.75 に届かなければ「読めない」として None（閾値は明るさに依存しない）。
    v = gl.ravel() - gl.mean()
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    scores = reps_flat @ (v / n)
    bi = int(np.argmax(scores))
    if scores[bi] < NCC_MIN:
        return None
    return LABELS.get(bi)


def band_frame_range(values, nframes):
    # v0.8e(B) 帯消失ガード：「同じフレームで何個のカウンターが読めたか」の移動平均（±BAND_W）を取り、
    # BAND_MIN 以上の区間を「帯が画面に出ている区間」とみなして、その最初と最後のフレーム番号を返す。
    # 曲が終わって帯が消えた後に背景の人物・輪郭を数字と誤認する分（頭と尻尾）だけを落とす。
    cnt = {}
    for name in values:
        for fr, _v in values[name]:
            cnt[fr] = cnt.get(fr, 0) + 1
    cum = [0] * (nframes + 1)
    for i in range(nframes):
        cum[i + 1] = cum[i] + cnt.get(i, 0)

    def smoothed(i):
        a = max(0, i - BAND_W)
        b = min(nframes, i + BAND_W + 1)
        return (cum[b] - cum[a]) / max(1, b - a)

    bs = 0
    while bs < nframes and smoothed(bs) < BAND_MIN:
        bs += 1
    be = nframes - 1
    while be >= 0 and smoothed(be) < BAND_MIN:
        be -= 1
    return bs, be


def detect_increments(seq):
    # 増分検出：同じ値が STABLE_MIN 回以上続いた「安定ラン」だけを残し（滞在フレーム数＝重み）、
    # 重み付き最長非減少部分列（WLIS）で本当の軌跡を1本選ぶ（カウンターは減らないという制約）。
    # 選ばれた軌跡で値が増えた点が加点＝(フレーム, 値, 増分) を返す。第2戻り値は安定ランの総数。
    runs = []  # [最初のフレーム, 値, 重み]
    run_v = None
    for fr, v in seq:
        if run_v is not None and v == run_v:
            runs[-1][2] += 1
        else:
            runs.append([fr, v, 1])
            run_v = v
    runs = [r for r in runs if r[2] >= STABLE_MIN]
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


# ---- 1) 帯動画を1コマずつ読み、各カウンターの数字を読む ----
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"fps={fps:.3f} frames={nframes} dur={nframes/fps:.1f}s")

values = {k: [] for k in DIGIT_ROIS}  # カウンター名 → [(フレーム, 読めた値)]
fidx = -1
while True:
    ok, img = cap.read()
    if not ok:
        break
    fidx += 1
    g = img.max(axis=2)  # v0.8d: グレースケール化の代わりに V=max(B,G,R)（色付き数字に強い）
    for name, roi in DIGIT_ROIS.items():
        x1, y1, x2, y2 = roi
        rg = g[y1:y2, x1:x2]
        boxes = glyph_boxes(rg)
        if boxes is None:
            continue
        digits = []
        okread = True
        for b in boxes:
            d = classify(norm_glyph(rg, b))
            if d is None:
                okread = False
                break
            digits.append(d)
        if not okread:
            continue
        val = digits[0] if len(digits) == 1 else digits[0] * 10 + digits[1]
        values[name].append((fidx, val))

for name in DIGIT_ROIS:
    print(f"{name}: readable {len(values[name])}/{nframes} ({100*len(values[name])/nframes:.0f}%)")

# ---- 2) v0.8e(B) 帯のある区間の外の読みを捨てる ----
_bs, _be = band_frame_range(values, nframes)
if _bs <= _be:
    print(f"band: present {_bs}..{_be} / {nframes} (BAND_MIN={BAND_MIN} W={BAND_W})")
    for name in DIGIT_ROIS:
        before = len(values[name])
        values[name] = [(f, v) for f, v in values[name] if _bs <= f <= _be]
        if len(values[name]) != before:
            print(f"{name}: band-trim {before} -> {len(values[name])}")
else:
    print("band: not detected - trim skipped")

# ---- 3) 増えた瞬間＝加点を取り出す ----
results = {}
for name in DIGIT_ROIS:
    accepted, n_runs = detect_increments(values[name])
    results[name] = accepted
    ev = sum(d for _, _, d in accepted)
    ini = accepted[0][1] if accepted else None
    fin = accepted[-1][1] if accepted else None
    rej = n_runs - len(accepted)
    print(f"{name}: initial={ini} final={fin} increments={ev} runs={n_runs} off_path={rej}")

# ---- 4) 加点の時刻を書き出す ----
# 遷移窓: 真のイベント時刻はカウンターフラッシュ開始≒「旧値が最後に読めたフレーム」直後
# (較正実測 2026/8/8: 数字は加点の瞬間にフラッシュ付きで変化し、安定読取はフラッシュ明け≈+0.5s)
with open(os.path.join(OUTDIR, "counter_events.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["counter", "frame", "time_sec", "value", "delta",
                "event_frame_est", "event_time_est"])
    rows = []
    for name, acc in results.items():
        seq = values[name]
        for i, (fr, v, d) in enumerate(acc):
            ev_fr = fr
            if i > 0:
                pv = acc[i - 1][1]
                last_old = None
                for f2, v2 in seq:
                    if f2 >= fr:
                        break
                    if v2 == pv:
                        last_old = f2
                if last_old is not None:
                    ev_fr = last_old + 1
            rows.append((fr, name, v, d, ev_fr))
    for fr, name, v, d, ev_fr in sorted(rows):
        w.writerow([name, fr, f"{fr/fps:.3f}", v, d, ev_fr, f"{ev_fr/fps:.3f}"])
json.dump({k: [[fr, v, d] for fr, v, d in acc] for k, acc in results.items()},
          open(os.path.join(OUTDIR, "counter_events.json"), "w"))
print("done")
