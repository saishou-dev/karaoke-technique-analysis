# 読取器 WAO版 v0.9 候補（2026-10-15）：v0.8e に「反転二値化フォールバック」を足したもの。
# v0.8e（read_counters_wao_v08e_20261015.py）と v0.8d（read_counters_wao.py）はどちらも無変更で残す。
#
# 追加した歯止め (C) ＝反転二値化フォールバック
#   読取器の二値化は `ROI > T`＝「暗い背景に明るい数字」専用。帯の背景が明るくなる演出では数字のほうが暗くなり、
#   連結成分が取れず「不可読」になる（10/15 実測＝Mela!(生音) 20260924 しゃくりで 593 フレーム＝10 秒が不可読）。
#   → 通常の極性で読めなかったフレームだけ、**反転（255 − ROI）**でもう一度読む。
#   出典＝報告書 `2026-10-15_1115_found_fill.md` の 3-2（10/6 WAO の目視追記 11 件のうち 8 件は反転読みで自動的に拾えていた）。
#   実装は found_pin_20261015.py の両極性読取と同じ形（通常→失敗時だけ反転・成功したらそれを採用）。
#
# 切替＝環境変数 KV_INVERT_FB（既定 1＝有効）。**KV_INVERT_FB=0 にすると v0.8e と完全に同じ挙動**になる
#   （(C) のパッチを当てないので、生成コードが v0.8e のものと1バイトも違わない。下の assert で担保）。
#   (A)(B) の閾値も v0.8e と同じ環境変数（KV_MIN_GAP2・KV_BAND_MIN・KV_BAND_W）で動かせる。
#
# (A)(B) のロジックは v0.8e のファイルから**そのまま取り込む**（二重管理をしないため）。
#   仕組み＝v0.8e は最後に `exec(compile(src, "read_counters.py", "exec"), g)` で組み立てた src を実行するので、
#   その compile を差し替えて src だけ受け取る（v0.8e のファイルは読むだけ・一切書き換えない）。
#
# 使い方は v0.8d / v0.8e と同じ: python read_counters_wao_v09_20261015.py <band.mp4> <outdir> <reps.npy> [roi.json]
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
V08E = os.environ.get("KV_V08E_SRC") or os.path.join(HERE, "read_counters_wao_v08e_20261015.py")

# ---- v0.8e が組み立てた読取器ソースを取り出す（v0.8e は実行しない）----
_cap = {}


def _compile(src, filename, mode, *a, **kw):
    if filename == "read_counters.py":
        _cap["src"] = src
        return compile("pass", filename, mode)
    return compile(src, filename, mode, *a, **kw)


_g = {"__name__": "__main__", "__file__": V08E, "compile": _compile}
exec(compile(open(V08E, encoding="utf-8").read(), V08E, "exec"), _g)
src = _cap["src"]
assert "MIN_GAP2" in src and "band-trim" in src, "v0.8e のソースを取り込めていない: " + V08E

INVERT_FB = os.environ.get("KV_INVERT_FB", "1") != "0"

# ---- (C) 反転二値化フォールバック（1フレーム1ROIの読取を関数に出し、失敗時だけ反転で再試行）----
OLD = """        rg = g[y1:y2, x1:x2]
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
"""
NEW = """        rg = g[y1:y2, x1:x2]
        val = _read_roi_once(rg)
        if val is None:
            val = _read_roi_once(255 - rg)   # v0.9 (C): 明背景＝数字が暗い区間を反転して読み直す
            if val is not None:
                _inv_hits[name] = _inv_hits.get(name, 0) + 1
        if val is None:
            continue
        values[name].append((fidx, val))
"""
HELPER = """
# ---- v0.9 (C) 1フレーム1ROIの読取（v0.8d/v0.8e の中身をそのまま関数に出したもの）----
_inv_hits = {}


def _read_roi_once(rg):
    boxes = glyph_boxes(rg)
    if boxes is None:
        return None
    digits = []
    for b in boxes:
        d = classify(norm_glyph(rg, b))
        if d is None:
            return None
        digits.append(d)
    return digits[0] if len(digits) == 1 else digits[0] * 10 + digits[1]


"""
REPORT = """
for name in DIGIT_ROIS:
    if _inv_hits.get(name):
        print(f"{name}: invert-fallback hits {_inv_hits[name]}")
"""

if INVERT_FB:
    assert src.count(OLD) == 1, "v0.9: 読取ループの差し替え位置が見つからない（v0.8e/ベースの変更？）"
    src = src.replace(OLD, NEW, 1)
    anchor = "cap = cv2.VideoCapture(VIDEO)"
    assert src.count(anchor) == 1
    src = src.replace(anchor, HELPER + anchor, 1)
    i = src.index("# 安定ラン抽出")   # 全フレーム走査が終わった直後
    src = src[:i] + REPORT + "\n" + src[i:]
    assert "_read_roi_once" in src and "invert-fallback" in src

# v0.8e が用意した globals（__WAO_LABELS__ を含む）をそのまま使う
g = dict(_g["g"])
g["__name__"] = "__main__"
exec(compile(src, "read_counters.py", "exec"), g)
