# -*- coding: utf-8 -*-
"""render_pitchbar_icons.py の縦長PNGを 4段ごとのページに分割し、見出し帯＋凡例（1ページ目・技法マーク図形）を付ける
（#14 lulu. 大判・アイコン版 2026-10-08）。分割方法は lulu_20260924_large/split_pages.py と同じ（段の検出＝非白画素が5割超の行の連続）。"""
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "lulu._20260924_101433_pitchbar_icons.png")
OUT = os.path.join(HERE, "pages"); ICONS = os.path.join(HERE, "icons_rgba")     # render_pitchbar_icons.py --icon_out の出力
FONT = "C:/Windows/Fonts/YuGothM.ttc"
ROW = 10.0; DUR = 265.0; PER = 4
TITLE = "lulu.（2026-09-24）アイコン版（技法マーク図形）"
# render_pitchbar_icons.py の COL / JA と同一
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
JA = {"shakuri": "しゃくり", "kobushi": "こぶし", "vibrato": "ビブラート", "fall": "フォール", "accent": "アクセント", "hammering": "ハンマリング"}
COL = {"shakuri": "#d9262c", "kobushi": "#2a5bd7", "vibrato": "#1fb83a", "fall": "#c13ab6", "accent": "#ef8a1c", "hammering": "#15b1c9"}
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

# --- 元ヘッダ（タイトル・凡例・F1表・総計）を消して、1段目の区間ラベルを描き直す（README と見出し帯に移す）---
d0 = ImageDraw.Draw(im)
top0 = blocks[0][0]
d0.rectangle([0, 0, W, top0 - 2], fill="white")
f_lab = ImageFont.truetype(FONT, 23)
d0.text((180, top0 - 6), "0:00 〜 0:10", font=f_lab, fill="#777777", anchor="ls")

# --- ページ境界 ---
cuts = [top0 - 70]
for i in range(PER, nrow, PER): cuts.append((blocks[i - 1][1] + blocks[i][0]) // 2)
last_nonwhite = np.where((a != 255).any(2).any(1))[0].max()
cuts.append(min(H, last_nonwhite + 40))
N = len(cuts) - 1

BAND = 120; LEG = 190
f1 = ImageFont.truetype(FONT, 46); f2 = ImageFont.truetype(FONT, 31); f3 = ImageFont.truetype(FONT, 30)
def mmss(t): t = int(round(t)); return f"{t // 60}:{t % 60:02d}"
def blend(hexcol, alpha):
    c = tuple(int(hexcol[i:i + 2], 16) for i in (1, 3, 5)); return tuple(int(255 * (1 - alpha) + v * alpha) for v in c)
ICON_H = 64
icons = {}
for c in TYPES:
    ic = Image.open(os.path.join(ICONS, f"{c}.png")).convert("RGBA"); icons[c] = ic.resize((max(1, int(round(ic.width * ICON_H / ic.height))), ICON_H), Image.LANCZOS)

pages = []
for p in range(N):
    sl = im.crop((0, cuts[p], W, cuts[p + 1]))
    extra = LEG if p == 0 else 0
    pg = Image.new("RGB", (W, BAND + extra + sl.height), "white"); d = ImageDraw.Draw(pg)
    r0, r1 = p * PER, min(nrow, (p + 1) * PER) - 1
    t0, t1 = r0 * ROW, min(DUR, (r1 + 1) * ROW)
    d.text((60, 52), f"{TITLE}　ページ {p + 1}/{N}　区間 {mmss(t0)}〜{mmss(t1)}（{r0 + 1}〜{r1 + 1}段目）", font=f1, fill="black", anchor="ls")
    d.text((60, 100), "上段＝予測／下段＝正解（G1）・実線＝一致／赤帯＝誤検知／紫帯＝見逃し　（横軸＝秒・縦軸＝音名・1段＝10秒・マーク＝DAM画面の技法マークを模した図形）", font=f2, fill="black", anchor="ls")
    d.line([0, BAND - 1, W, BAND - 1], fill="#bbbbbb", width=2)
    if p == 0:
        # 凡例（技法6種＝render_pitchbar_icons.py が書き出した図形の透過PNG＋技法名）＋帯と線の意味
        x = 60; cy = BAND + 50
        for c in TYPES:
            ic = icons[c]; pg.paste(ic, (int(x), cy - ICON_H // 2), ic); x += ic.width + 14
            d.text((x, cy), JA[c], font=f3, fill=COL[c], anchor="lm"); x += d.textlength(JA[c], font=f3) + 56
        x = 60; cy = BAND + 136
        d.rectangle([x, cy - 16, x + 44, cy + 16], outline="#333333", width=3); d.text((x + 56, cy), "正解（同じ形の縁取りのみ・下側）", font=f3, fill="#222222", anchor="lm"); x += 56 + d.textlength("正解（同じ形の縁取りのみ・下側）", font=f3) + 40
        d.rectangle([x, cy - 16, x + 44, cy + 16], fill="#999999"); d.text((x + 56, cy), "予測（塗り・上側・濃さ＝確率）", font=f3, fill="#222222", anchor="lm"); x += 56 + d.textlength("予測（塗り・上側・濃さ＝確率）", font=f3) + 40
        d.line([x, cy, x + 40, cy], fill="#333333", width=4); d.text((x + 52, cy), "一致（±1s）", font=f3, fill="#222222", anchor="lm"); x += 52 + d.textlength("一致（±1s）", font=f3) + 40
        for col, al, lab in ((C_FP, 0.3, "誤検知"), (C_FN, 0.3, "見逃し"), (C_OUT, 0.3, "採点対象外"), (C_BAR, 1.0, "歌った音程")):
            d.rectangle([x, cy - 14, x + 44, cy + 14], fill=blend(col, al)); d.text((x + 56, cy), lab, font=f3, fill="#222222", anchor="lm"); x += 56 + d.textlength(lab, font=f3) + 36
        d.line([0, BAND + LEG - 1, W, BAND + LEG - 1], fill="#dddddd", width=1)
    pg.paste(sl, (0, BAND + extra))
    fn = os.path.join(OUT, f"page_{p + 1:02d}.png"); pg.save(fn, optimize=True)
    pages.append(dict(page=p + 1, file=fn, rows=f"{r0 + 1}-{r1 + 1}", span=f"{mmss(t0)}〜{mmss(t1)}", size=pg.size, bytes=os.path.getsize(fn)))
    print(f"page {p + 1}/{N}: 段{r0 + 1}〜{r1 + 1} {mmss(t0)}〜{mmss(t1)} {pg.size} {os.path.getsize(fn) // 1024}KB")
json.dump(pages, open(os.path.join(OUT, "pages.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
