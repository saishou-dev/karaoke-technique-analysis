# 読取器 v0.8d＋ROI の明るさ補正（2026-11-10 Opus・新規・read_counters_wao.py の複製に1点だけ追加）
#  11/10 はしゃくりの数字だけが暗い（帯動画 30曲の ROI 99.5% 点の中央値＝しゃくり 121・こぶし 170・フォール 196・ビブラート 173・アクセント 204・ハンマリング 232）。
#  v0.8d の最低閾値 130 を数字が超えず、しゃくりの可読率 0〜3%＝verify 0/30。
#  追加＝環境変数 KV_ROI_GAIN（例 "shakuri=1.45"）のカウンターだけ、切り出した ROI に係数を掛けてから（255で頭打ち）v0.8d と同じ処理をする。
#  KV_ROI_GAIN が空なら v0.8d と完全に同じ。
# 読取器 WAO版 v0.8d（2026-09-12）：read_counters.py を元に
#  (1) LABELS を <reps>_labels.json で置換  (2) 切り出しを V=max(B,G,R) で行い低閾値150/130を追加
#  (3) glyph_boxes＝最初に通った閾値で即返さず、全閾値の候補から「2桁（2箱）」を優先して採用（十の位の取りこぼし対策・2026-09-12 verify 1/30 の原因）
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
assert "img.max(axis=2)" in src and "150, 130" in src and "best2" in src
_gs = os.environ.get("KV_ROI_GAIN", "").strip()
g["__ROI_GAIN__"] = {k: float(v) for k, v in (x.split("=") for x in _gs.split(",") if x)}
_old = "        rg = g[y1:y2, x1:x2]" + chr(10)
assert src.count(_old) == 1
src = src.replace(_old, _old + "        if name in __ROI_GAIN__: rg = np.clip(rg.astype(np.float32) * __ROI_GAIN__[name], 0, 255).astype(np.uint8)" + chr(10))
print("ROI_GAIN", g["__ROI_GAIN__"], flush=True)
exec(compile(src,"read_counters.py","exec"),g)
