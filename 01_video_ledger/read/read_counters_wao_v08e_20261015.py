# 読取器 WAO版 v0.8e（2026-10-15）：v0.8d（read_counters_wao.py）に 10/14 の fall 偽陽性2種への歯止めを足したもの。
# v0.8d は無変更のまま残す（既存の 20260915／20260924／20261006 の結果は再現できる）。
#
# 10/14 verify＝PERFECT 16/30。過剰読み 15件のうち 14件が fall。証拠タイルで原因を2つに切り分けた：
#   (A) 1桁の「1」が2つの連結成分に割れ、v0.8d の「2桁優先」がそれを拾って "11" と読む。
#       例＝きらり f13880（DAM CSV の真値 1 → 読値 11）。画面は一貫して「1」1文字。
#       対策＝2桁として採用する条件に「桁間の隙間 gap >= KV_MIN_GAP2（既定5px）」を加える。
#       根拠＝10/14 の20曲・2箱フレーム 29,910 枚の実測で、真の2桁は gap 中央値9（p5=6）、偽の2箱は中央値1（p95=4）。
#       gap>=5 で真の2桁の 99.46% を残し、偽の2箱の 91.4% を落とす。落ちた枠は1箱候補へフォールバックする。
#       既定を6でなく5にしたのは実測比較による：6 だと本物の2桁（RPG accent 22・言葉のいらない約束 accent 21/22）を
#       3件落とすが、5 なら偽陽性4件は同じく全部消えたうえでその3件が残る（10/15 実測）。
#   (B) 曲が終わってカウンター帯が画面から消えた後、背景の映像（人物・輪郭）を数字と誤認する。
#       例＝怪獣の花唄 f16362（帯は f13011 で消えている）・ホワイトノイズ f14520・Habit f11638。
#       対策＝「同時に数字が読めたカウンター数」の移動平均（±150フレーム≒5秒）が
#       KV_BAND_MIN（既定0.5）以上の区間＝帯あり、とみなし、その前後のフレームの読みを捨てる。
#       頭と尻尾を削るだけなので曲中の読みは一切変えない。
#       根拠＝怪獣の花唄は曲中 3.56 に対し問題フレームで 0.07。きらりは問題フレームでも 1.42（＝Aの事例で、こちらでは落ちない）。
#
# 使い方は v0.8d と同じ: python read_counters_wao_v08e_20261015.py <band.mp4> <outdir> <reps.npy> [roi.json]
import json, os, sys
reps = sys.argv[3]; lab = os.path.splitext(reps)[0] + "_labels.json"
SRC = os.environ.get("KV_READER_SRC") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "read_counters.py")
src = open(SRC, encoding="utf-8").read()
g = {"__name__": "__main__"}
if os.path.exists(lab):
    L = json.load(open(lab)); g["__WAO_LABELS__"] = {int(k): v for k, v in L.items()}
    src = src.replace("LABELS = {", "LABELS = __WAO_LABELS__ or {", 1)
src = src.replace("THRESHOLDS = (215, 200, 185, 228, 170)", "THRESHOLDS = (215, 200, 185, 228, 170, 150, 130)")
src = src.replace("g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)", "g = img.max(axis=2)")

# ---- (A) 2桁ガード付きの glyph_boxes（v0.8d の形＋gap 下限）----
i = src.index("def glyph_boxes(roi_gray):"); j = src.index("def norm_glyph(roi_gray, box):")
MIN_GAP2 = int(os.environ.get("KV_MIN_GAP2", "5"))
new_fn = '''MIN_GAP2 = %d  # v0.8e: 2桁として採用する桁間の最小隙間(px)

def glyph_boxes(roi_gray):
    best1 = None; best2 = None
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
            if best2 is None: best2 = boxes
        elif best1 is None:
            best1 = boxes
    return best2 if best2 is not None else best1

''' % MIN_GAP2
src = src[:i] + new_fn + src[j:]

# ---- (B) 帯消失ガード：安定ラン抽出の直前に、帯のある区間の外の読みを捨てる ----
anchor = "# 安定ラン抽出"
k = src.index(anchor)
trim = '''# ---- v0.8e 帯消失ガード（2026-10-15）----
# 「同じフレームで何個のカウンターが読めたか」の移動平均で帯の在／不在を測り、帯のある区間の外を捨てる。
_BAND_MIN = float(os.environ.get("KV_BAND_MIN", "0.5")); _BAND_W = int(os.environ.get("KV_BAND_W", "150"))
_cnt = {}
for _n in DIGIT_ROIS:
    for _fr, _v in values[_n]:
        _cnt[_fr] = _cnt.get(_fr, 0) + 1
_cum = [0] * (nframes + 1)
for _i in range(nframes):
    _cum[_i + 1] = _cum[_i] + _cnt.get(_i, 0)
def _sm(_i):
    _a = max(0, _i - _BAND_W); _b = min(nframes, _i + _BAND_W + 1)
    return (_cum[_b] - _cum[_a]) / max(1, _b - _a)
_bs = 0
while _bs < nframes and _sm(_bs) < _BAND_MIN: _bs += 1
_be = nframes - 1
while _be >= 0 and _sm(_be) < _BAND_MIN: _be -= 1
if _bs <= _be:
    print(f"band: present {_bs}..{_be} / {nframes} (BAND_MIN={_BAND_MIN} W={_BAND_W})")
    for _n in DIGIT_ROIS:
        _b4 = len(values[_n])
        values[_n] = [(_f, _v) for _f, _v in values[_n] if _bs <= _f <= _be]
        if len(values[_n]) != _b4:
            print(f"{_n}: band-trim {_b4} -> {len(values[_n])}")
else:
    print("band: not detected - trim skipped")

'''
src = src[:k] + trim + src[k:]

assert "img.max(axis=2)" in src and "150, 130" in src and "best2" in src and "MIN_GAP2" in src and "band-trim" in src
exec(compile(src, "read_counters.py", "exec"), g)
