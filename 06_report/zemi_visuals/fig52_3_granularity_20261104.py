# #52-3 切り出し粒度の比較（2026-11-04）＝既存の 4指標×全手法 の図から、3条件×2手法×3指標に絞った1枚。
# 数値の出典＝G:\hisho\research\比較_切り出し粒度_2026-10-29.md 2-2節（主たる表）・3節（桁の水増し）・4節（処理時間）。
# 元の CSV＝result\compare_gran_sub4.csv（10,794件・正解確定 10,442件）。ここでは表の値を書き写しただけで、計算はしない。
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_20261104 import *
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else r"G:\本人確認待ち\#52_ゼミ補助資料_2026-11-05\#52-3_切り出し粒度の比較.png"

CONDS = ["技法ごとの枠", "桁ごとの窓", "桁の位置を固定"]   # B / C2 / C
# (自作の見本照合, 既製のOCR＝PaddleOCR 認識のみ・確信度0.90で足切り)
ACC = [(98.21, 99.71), (89.40, 98.06), (39.85, 39.67)]       # 値を出した時の正しさ %
DUP = [(1, 9), (0, 158), (0, 3082)]                          # 桁の水増し＝1が11と読まれる型の誤り 件
SEC = [(0.740, 426.730), (0.306, 503.894), (0.229, 866.284)]  # 1,000件あたり秒

C_OWN = NAVY; C_OCR = "#9aa5b1"

fig = plt.figure(figsize=(20, 9.2), dpi=100, facecolor="white")
fig.text(0.5, 0.955, "切り出しの細かさを3通りにして比べた結果", ha="center", va="center", fontsize=26, fontweight="bold", color=INK)

panels = [("値を出した時の正しさ", "%"), ("1が11と読まれる型の誤り", "件"), ("1,000件あたりの処理時間", "秒・対数")]
lefts = [0.06, 0.385, 0.71]
x = np.arange(3); bw = 0.36
for k, (title, unit) in enumerate(panels):
    step_title(fig, lefts[k] + 0.012, 0.86, k + 1, title, size=19)
    ax = fig.add_axes([lefts[k] + 0.01, 0.20, 0.25, 0.58])
    data = [ACC, DUP, SEC][k]
    own = [d[0] for d in data]; ocr = [d[1] for d in data]
    if k == 2:
        ax.set_yscale("log"); ax.set_ylim(0.1, 3000)
        ax.set_yticks([0.1, 1, 10, 100, 1000]); ax.set_yticklabels(["0.1", "1", "10", "100", "1,000"])
    elif k == 1:
        ax.set_ylim(0, 3500)
    else:
        ax.set_ylim(0, 110)
    b1 = ax.bar(x - bw / 2, own, bw, color=C_OWN, label="自作の見本照合")
    b2 = ax.bar(x + bw / 2, ocr, bw, color=C_OCR, label="既製のOCR")
    for i in range(3):
        for j, (val, xx) in enumerate([(own[i], x[i] - bw / 2), (ocr[i], x[i] + bw / 2)]):
            if k == 0:
                s = f"{val:.1f}"
            elif k == 1:
                s = f"{val:,}"
            else:
                s = f"{val:.2f}" if val < 10 else f"{val:,.0f}"
            col = INK
            if k == 1 and j == 1 and i > 0:
                col = RED
            if k == 0 and i == 0:
                col = GREEN
            yy = val if val > 0 else (0.1 if k == 2 else 0)
            off = yy * 1.15 if k == 2 else yy + (2 if k == 0 else 60)
            ax.text(xx, off, s, ha="center", va="bottom", fontsize=15, fontweight="bold", color=col)
    ax.set_xticks(x); ax.set_xticklabels(CONDS, fontsize=15)
    ax.tick_params(axis="y", labelsize=12, colors=GRAY)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(EDGE); ax.spines["bottom"].set_color(GRAY)
    ax.set_ylabel(unit, fontsize=13, color=GRAY)
    ax.grid(axis="y", color="#eeeeee", lw=1); ax.set_axisbelow(True)
    if k == 0:
        # 技法ごとの枠が一番正しい＝緑の枠
        ax.add_patch(matplotlib.patches.Rectangle((-0.5, 0), 1.0, 108, fill=False, ec=GREEN, lw=2.5, ls="-"))

# 凡例
lg = fig.add_axes([0, 0, 1, 1], facecolor="none"); lg.set_axis_off(); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
lg.add_patch(matplotlib.patches.Rectangle((0.30, 0.075), 0.018, 0.03, color=C_OWN))
lg.text(0.323, 0.09, "自作の見本照合", va="center", fontsize=16, color=INK)
lg.add_patch(matplotlib.patches.Rectangle((0.48, 0.075), 0.018, 0.03, color=C_OCR))
lg.text(0.503, 0.09, "既製のOCR　PaddleOCR・確信度0.90以上だけ採る", va="center", fontsize=16, color=INK)
fig.text(0.5, 0.03, "同じ4曲・同じ10,794件・同じ後処理で比べた", ha="center", fontsize=14, color=GRAY)

fig.savefig(OUT, facecolor="white")
print("saved", OUT)
