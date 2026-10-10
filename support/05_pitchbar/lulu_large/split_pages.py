# -*- coding: utf-8 -*-
"""render_pitchbar.py の縦長PNGを 4段ごとのページに分割し、見出し帯＋凡例を付ける（#14 lulu. 大判）"""
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None

SRC = "G:/研究アーカイブ/成果物/動画解析/pitchbar/lulu_20260924_large/lulu._20260924_101433_pitchbar.png"
OUT = "G:/研究アーカイブ/成果物/動画解析/pitchbar/lulu_20260924_large/pages"
FONT = "C:/Windows/Fonts/YuGothM.ttc"
ROW = 10.0; DUR = 265.0; PER = 4
TITLE = "lulu.（2026-09-24）"
# render_pitchbar.py の COL / MK / JA と同一
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
JA = {"shakuri": "しゃくり", "kobushi": "こぶし", "vibrato": "ビブラート", "fall": "フォール", "accent": "アクセント", "hammering": "ハンマリング"}
COL = {"shakuri": "#2a78d6", "kobushi": "#eb6834", "vibrato": "#1baf7a", "fall": "#eda100", "accent": "#e87ba4", "hammering": "#008300"}
MK = {"shakuri": "^", "kobushi": "D", "vibrato": "o", "fall": "v", "accent": "s", "hammering": "P"}
C_FP = "#e34948"; C_FN = "#4a3aa7"; C_BAR = "#44506b"; C_OUT = "#9a9a9a"
os.makedirs(OUT, exist_ok=True)

im = Image.open(SRC).convert("RGB"); W, H = im.size
a = np.asarray(im); mask = (a != 255).any(2).mean(1) > 0.5
blocks = []; s = None
for y, m in enumerate(mask):
    if m and s is None: s = y
    if not m and s is not None:
        if y - s > 50: blocks.append([s, y])
        s = None
if s is not None: blocks.append([s, H])
merged = []
for b in blocks:
    if merged and b[0] - merged[-1][1] < 100: merged[-1][1] = b[1]
    else: merged.append(b)
blocks = merged; nrow = len(blocks)
assert nrow == int(np.ceil(DUR / ROW)), (nrow, DUR)
print("段数", nrow, "段の高さ", blocks[0][1] - blocks[0][0], "段間", blocks[1][0] - blocks[0][1])

# --- 元ヘッダの残りを消す（図全体の注記が1段目に重なっているため）---
d0 = ImageDraw.Draw(im)
top0 = blocks[0][0]
d0.rectangle([0, 0, W, top0 - 2], fill="white")                      # 元ヘッダ（タイトル・凡例・F1表）＝README と見出し帯に移す
f_lab = ImageFont.truetype(FONT, 23)
d0.text((180, top0 - 6), "0:00 〜 0:10", font=f_lab, fill="#777777", anchor="ls")  # 1段目の区間ラベルを描き直す
# 1段目右上の「一致／誤検知／見逃し」総計テキスト（図全体の注記）を、同じ段の左側の帯をコピーして消す
y0, y1 = top0 + 55, top0 + 125
strip = im.crop((2200, y0, 2600, y1))
for x in range(2590, 2990, 400): im.paste(strip.crop((0, 0, min(400, 2990 - x), y1 - y0)), (x, y0))
d0.rectangle([2990, y0, W, y1], fill="white")

# --- ページ境界 ---
cuts = [top0 - 70]
for i in range(PER, nrow, PER): cuts.append((blocks[i - 1][1] + blocks[i][0]) // 2)
last_nonwhite = np.where((a != 255).any(2).any(1))[0].max()
cuts.append(min(H, last_nonwhite + 40))
N = len(cuts) - 1

BAND = 120; LEG = 150
f1 = ImageFont.truetype(FONT, 46); f2 = ImageFont.truetype(FONT, 31); f3 = ImageFont.truetype(FONT, 30)
def mmss(t): t = int(round(t)); return f"{t // 60}:{t % 60:02d}"
def marker(d, kind, cx, cy, r, col):
    if kind == "^": d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=col)
    elif kind == "v": d.polygon([(cx, cy + r), (cx - r, cy - r), (cx + r, cy - r)], fill=col)
    elif kind == "D": d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=col)
    elif kind == "o": d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    elif kind == "s": d.rectangle([cx - r + 1, cy - r + 1, cx + r - 1, cy + r - 1], fill=col)
    elif kind == "P":
        w = r // 2; d.rectangle([cx - w, cy - r, cx + w, cy + r], fill=col); d.rectangle([cx - r, cy - w, cx + r, cy + w], fill=col)
def blend(hexcol, alpha):
    c = tuple(int(hexcol[i:i + 2], 16) for i in (1, 3, 5)); return tuple(int(255 * (1 - alpha) + v * alpha) for v in c)

pages = []
for p in range(N):
    sl = im.crop((0, cuts[p], W, cuts[p + 1]))
    extra = LEG if p == 0 else 0
    pg = Image.new("RGB", (W, BAND + extra + sl.height), "white"); d = ImageDraw.Draw(pg)
    r0, r1 = p * PER, min(nrow, (p + 1) * PER) - 1
    t0, t1 = r0 * ROW, min(DUR, (r1 + 1) * ROW)
    d.text((60, 52), f"{TITLE}　ページ {p + 1}/{N}　区間 {mmss(t0)}〜{mmss(t1)}（{r0 + 1}〜{r1 + 1}段目）", font=f1, fill="black", anchor="ls")
    d.text((60, 100), "上段＝予測／下段＝正解（G1）・実線＝一致／赤帯＝誤検知／紫帯＝見逃し　（横軸＝秒・縦軸＝音名・1段＝10秒）", font=f2, fill="black", anchor="ls")
    d.line([0, BAND - 1, W, BAND - 1], fill="#bbbbbb", width=2)
    if p == 0:
        # 凡例（技法6色＋記号・render_pitchbar.py の COL/MK/JA と同一）＋帯と線の意味
        x = 60; cy = BAND + 42
        for c in TYPES:
            marker(d, MK[c], x + 16, cy, 15, COL[c]); d.text((x + 42, cy), JA[c], font=f3, fill="#222222", anchor="lm"); x += 42 + d.textlength(JA[c], font=f3) + 46
        x = 60; cy = BAND + 108
        d.ellipse([x, cy - 14, x + 28, cy + 14], outline="#333333", width=3); d.text((x + 40, cy), "正解（中抜き・下側）", font=f3, fill="#222222", anchor="lm"); x += 40 + d.textlength("正解（中抜き・下側）", font=f3) + 40
        d.ellipse([x, cy - 14, x + 28, cy + 14], fill="#333333"); d.text((x + 40, cy), "予測（塗り・上側・濃さ＝確率）", font=f3, fill="#222222", anchor="lm"); x += 40 + d.textlength("予測（塗り・上側・濃さ＝確率）", font=f3) + 40
        d.line([x, cy, x + 40, cy], fill="#333333", width=4); d.text((x + 52, cy), "一致（±1s）", font=f3, fill="#222222", anchor="lm"); x += 52 + d.textlength("一致（±1s）", font=f3) + 40
        for col, al, lab in ((C_FP, 0.3, "誤検知"), (C_FN, 0.3, "見逃し"), (C_OUT, 0.3, "採点対象外"), (C_BAR, 1.0, "歌った音程")):
            d.rectangle([x, cy - 14, x + 44, cy + 14], fill=blend(col, al)); d.text((x + 56, cy), lab, font=f3, fill="#222222", anchor="lm"); x += 56 + d.textlength(lab, font=f3) + 36
        d.line([0, BAND + LEG - 1, W, BAND + LEG - 1], fill="#dddddd", width=1)
    pg.paste(sl, (0, BAND + extra))
    fn = os.path.join(OUT, f"page_{p + 1:02d}.png"); pg.save(fn, optimize=True)
    pages.append(dict(page=p + 1, file=fn, rows=f"{r0 + 1}-{r1 + 1}", span=f"{mmss(t0)}〜{mmss(t1)}", size=pg.size, bytes=os.path.getsize(fn)))
    print(f"page {p + 1}/{N}: 段{r0 + 1}〜{r1 + 1} {mmss(t0)}〜{mmss(t1)} {pg.size} {os.path.getsize(fn) // 1024}KB")
json.dump(pages, open(os.path.join(OUT, "pages.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
