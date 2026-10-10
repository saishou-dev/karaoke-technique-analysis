# -*- coding: utf-8 -*-
"""フォールの技法マーク テンプレート fall_color.png を作る（2026-10-08）。
marks_20260928/scripts/templates/ には fall が無い（make_templates.py の SPECS に未登録）ため、
marks_20260924.csv の fall 行（かくれんぼ_20260924_102001・v4出現 122.793s・mark_x=868, mark_y=590）に基づき
録画の 122.928s（v4+8f・sheet_refined_fall.png 1行目右端と同じフレーム）を ffmpeg で 1 枚復号し、
4K座標 (826,560)-(920,644) を目視で決めて切り出す。確認画像 #14_fall_template_確認.png も出力する。
注: dataset/evidence_20260902/m_answer_fall_*.png は「フォール」カウンタの数字周辺で、マーク本体が写っていないため使わない。"""
import os, sys, subprocess, numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
VID = r"G:\研究アーカイブ\ビデオ素材\20260924\かくれんぼ_20260924_102001.mp4"
T = 122.928; BOX = (826, 560, 920, 644)          # 4K座標 x0,y0,x1,y1（目視）
TPL = os.path.join(HERE, "templates"); os.makedirs(TPL, exist_ok=True)
frame_png = os.path.join(HERE, "templates", "_fall_src_frame_crop.png")
tmp = os.path.join(os.environ.get("TEMP", HERE), "fall_frame_4k.png")
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{T:.3f}", "-i", VID, "-frames:v", "1", "-y", tmp], check=True)
im = Image.open(tmp).convert("RGB"); assert im.size == (3840, 2160), im.size
ctx = im.crop((BOX[0] - 120, BOX[1] - 100, BOX[2] + 120, BOX[3] + 100)); ctx.save(frame_png)   # 周辺（根拠として保存）
tpl = im.crop(BOX); tpl.save(os.path.join(TPL, "fall_color.png")); print("fall_color.png", tpl.size)
# 確認画像：周辺＋ROI枠／切り出し（×4）／他5技法のテンプレ（×4）
MK = r"G:\研究アーカイブ\成果物\動画解析\marks_20260928\scripts\templates"
FONT = ImageFont.truetype("C:/Windows/Fonts/YuGothM.ttc", 26)
big = ctx.resize((ctx.width * 3, ctx.height * 3), Image.NEAREST); d = ImageDraw.Draw(big)
d.rectangle([(120 * 3, 100 * 3), ((BOX[2] - BOX[0] + 120) * 3, (BOX[3] - BOX[1] + 100) * 3)], outline="white", width=3)
tiles = [("fall（新規・かくれんぼ 122.928s）", tpl)] + [(k, Image.open(os.path.join(MK, f"{k}_color.png")).convert("RGB")) for k in ("shakuri", "kobushi", "vibrato", "accent", "hammering")]
row_h = max(t.height for _, t in tiles) * 4 + 50; W = max(big.width, sum(t.width * 4 + 30 for _, t in tiles) + 30)
out = Image.new("RGB", (W, 70 + big.height + 40 + row_h), "white"); d = ImageDraw.Draw(out)
d.text((20, 20), f"フォールのテンプレ確認：{os.path.basename(VID)} t={T}s 4K座標 {BOX}（白枠＝切り出し範囲・×3）", font=FONT, fill="black")
out.paste(big, (20, 70)); y = 70 + big.height + 40; x = 20
d.text((20, y - 32), "下段：fall_color.png（×4）と既存5テンプレ（marks_20260928/scripts/templates/*_color.png・×4）", font=FONT, fill="black")
for name, t in tiles:
    out.paste(t.resize((t.width * 4, t.height * 4), Image.NEAREST), (x, y + 10)); d.text((x, y + 10 + t.height * 4 + 4), name, font=FONT, fill="#333"); x += t.width * 4 + 30
out.save(os.path.join(HERE, "#14_fall_template_確認.png")); print("確認画像", out.size)
