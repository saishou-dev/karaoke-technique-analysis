# ゼミ補助資料 #52 用（2026-11-04）：読み取り本体 v0.8e の「1フレームの読み」だけを図のために呼び出す薄い読み込み口。
# 既存の read_counters.py と read_counters_wao_v08e_20261015.py は書き換えない。
# v0.8e と同じ置き換え（ラベル・明るさ段階7つ・各色の最大で灰色化・2桁ガード MIN_GAP2=5）を当てた本体のうち、
# 動画を読む前までの部分（関数と見本の読み込み）だけを実行する。
# 1フレームの読み＝read_frame(img, roi) → (値 or None, 各桁の (箱, 数字, 類似度))。
import json, os, sys
import numpy as np

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
CORE = os.path.join(ROOT, "scripts", "wsl_karaoke_vision", "read_counters.py")
V08E = os.path.join(ROOT, "scripts", "read_counters_wao_v08e_20261015.py")
REPS = os.path.join(ROOT, "templates", "wao_20260816", "cluster_reps_wao_ext.npy")


def _build():
    src = open(CORE, encoding="utf-8").read()
    v8 = open(V08E, encoding="utf-8").read()
    lab = os.path.splitext(REPS)[0] + "_labels.json"
    g = {"__name__": "zemi_core"}
    L = json.load(open(lab)); g["__WAO_LABELS__"] = {int(k): v for k, v in L.items()}
    src = src.replace("LABELS = {", "LABELS = __WAO_LABELS__ or {", 1)
    src = src.replace("THRESHOLDS = (215, 200, 185, 228, 170)", "THRESHOLDS = (215, 200, 185, 228, 170, 150, 130)")
    # v0.8e の 2桁ガード付き glyph_boxes を v0.8e のファイルから取り出して差し込む（文字列は v0.8e と同一）
    a = v8.index("new_fn = '''") + len("new_fn = '''"); b = v8.index("''' % MIN_GAP2")
    new_fn = v8[a:b] % 5
    i = src.index("def glyph_boxes(roi_gray):"); j = src.index("def norm_glyph(roi_gray, box):")
    src = src[:i] + new_fn + src[j:]
    # 動画を開く前で止める
    src = src[:src.index("cap = cv2.VideoCapture(VIDEO)")]
    argv = sys.argv
    sys.argv = ["x", "x", "x", REPS]
    try:
        exec(compile(src, "read_counters.py", "exec"), g)
    finally:
        sys.argv = argv
    return g

G = _build()
NCC_MIN = G["NCC_MIN"]


def classify_score(gl):
    v = gl.ravel() - gl.mean(); n = np.linalg.norm(v)
    if n < 1e-6:
        return None, 0.0
    s = G["reps_flat"] @ (v / n); bi = int(np.argmax(s))
    d = G["LABELS"].get(bi) if s[bi] >= NCC_MIN else None
    return d, float(s[bi])


def read_frame(img, roi):
    """v0.8e と同じ判定で1欄を読む。img＝BGR の帯1フレーム。roi＝(x1,y1,x2,y2)。"""
    gimg = img.max(axis=2)
    x1, y1, x2, y2 = roi
    rg = gimg[y1:y2, x1:x2]
    boxes = G["glyph_boxes"](rg)
    if boxes is None:
        return None, []
    det = []; digits = []
    for bx in boxes:
        gl = G["norm_glyph"](rg, bx)
        d, sc = classify_score(gl)
        det.append((bx, d, sc))
        if d is None:
            return None, det
        digits.append(d)
    val = digits[0] if len(digits) == 1 else digits[0] * 10 + digits[1]
    return val, det
