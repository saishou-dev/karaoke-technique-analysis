# 読取器 WAO版 v0.8d の 9/15 再走査用ラッパ（2026-09-15）。read_counters_wao.py と同じ差し替えを read_counters.py に施した上で、
#  (4) 環境変数 KV_MAXFRAMES（先頭 N フレームだけ読む・可読率の短時間テスト用）
#  (5) 環境変数 KV_PREP（ROI 前処理：空＝なし／clahe＝V値に CLAHE(2.0,4x4)／sharp＝アンシャープ／clahe+sharp）
#  を追加。argv は同じ（video outdir reps roi.json）。KV_MAXFRAMES・KV_PREP 未設定なら v0.8d と同一の読取。
import json, os, sys
reps=sys.argv[3]; lab=os.path.splitext(reps)[0]+"_labels.json"
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"read_counters.py"),encoding="utf-8").read()
g={"__name__":"__main__"}
if os.path.exists(lab):
    L=json.load(open(lab)); g["__WAO_LABELS__"]={int(k):v for k,v in L.items()}
    src=src.replace("LABELS = {", "LABELS = __WAO_LABELS__ or {",1)
src=src.replace("THRESHOLDS = (215, 200, 185, 228, 170)","THRESHOLDS = (215, 200, 185, 228, 170, 150, 130)")
src=src.replace("g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)","g = img.max(axis=2)")
i=src.index("def glyph_boxes(roi_gray):"); j=src.index("def norm_glyph(roi_gray, box):")
new_fn='''def glyph_boxes(roi_gray):
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
            if gap > 14 or gap < -4 or hdiff > 8 or ydiff > 6:
                continue  # 2桁として不自然(キラキラ等の混入)＝この閾値は不採用
            if best2 is None: best2 = boxes
        elif best1 is None:
            best1 = boxes
    return best2 if best2 is not None else best1

'''
src=src[:i]+new_fn+src[j:]
# (4) 先頭 N フレーム
MAXF=int(os.environ.get("KV_MAXFRAMES","0") or 0)
if MAXF>0:
    src=src.replace("    fidx += 1\n","    fidx += 1\n    if fidx >= __MAXF__: break\n",1); g["__MAXF__"]=MAXF
    src=src.replace("nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))","nframes = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), __MAXF__)")
# (5) ROI 前処理
PREP=os.environ.get("KV_PREP","")
if PREP:
    g["__PREP__"]=PREP
    src=src.replace("        rg = g[y1:y2, x1:x2]\n","        rg = __prep__(g[y1:y2, x1:x2])\n",1)
    src=src.replace("THRESHOLDS = (","""_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
def __prep__(rg):
    rg = np.ascontiguousarray(rg)
    if "clahe" in __PREP__: rg = _clahe.apply(rg)
    if "sharp" in __PREP__:
        bl = cv2.GaussianBlur(rg, (0, 0), 1.5); rg = cv2.addWeighted(rg, 1.8, bl, -0.8, 0)
    return rg
THRESHOLDS = (""",1)
    assert "__prep__(g[" in src
assert "img.max(axis=2)" in src and "150, 130" in src and "best2" in src
exec(compile(src,"read_counters.py","exec"),g)
