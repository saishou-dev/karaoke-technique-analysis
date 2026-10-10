# -*- coding: utf-8 -*-
"""加点マークのテンプレート（色ファミリのマスク）を、目視で位置を確認済みのフレームから切り出して保存する。
出力: scripts/templates/<kind>_gray.png（照合用グレー）, <kind>_color.png（確認用）, evidence/templates_sheet.png"""
import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import detect_marks as d

TPL_DIR = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(TPL_DIR, exist_ok=True)
# (kind, video, t, 4K座標の矩形 x0,y0,x1,y1, 色ファミリ)  ※目視根拠: evidence/px_*.png
SPECS = [
    ("shakuri", "Brand New_20260924_121235", 32.60, (1158, 688, 1242, 758)),
    ("accent",  "Brand New_20260924_121235", 47.90, (826, 742, 912, 808)),
    ("vibrato", "Brand New_20260924_121235", 47.90, (1826, 702, 1924, 762)),
    ("kobushi", "CHEERS_20260924_110618",    29.75, (2732, 596, 2828, 688)),
    ("hammering", "結_20260924_104818", 73.93, (1925, 672, 2030, 736)),   # 目視根拠: evidence/px_hammering_yui_T73.5.png
]

def grab(video, t, box):
    x0, y0, x1, y1 = box
    frames = list(d.decode_segment(os.path.join(d.VID_DIR, video + ".mp4"), t, 1, (x0, y0, x1 - x0, y1 - y0)))
    return frames[0][1]

tiles = []
for kind, video, t, box in SPECS:
    bgr = grab(video, t, box)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    d.imwrite_u(os.path.join(TPL_DIR, f"{kind}_gray.png"), gray)
    d.imwrite_u(os.path.join(TPL_DIR, f"{kind}_color.png"), bgr)
    print(kind, "size", gray.shape[1], gray.shape[0])
    big = cv2.resize(bgr, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.putText(big, kind, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    tiles.append(big)
W = max(t.shape[1] for t in tiles)
sheet = np.vstack([np.pad(t, ((0, 0), (0, W - t.shape[1]), (0, 0))) for t in tiles])
d.imwrite_u(os.path.join(d.OUT_DIR, "evidence", "templates_sheet.png"), sheet)
print("saved templates_sheet.png")
