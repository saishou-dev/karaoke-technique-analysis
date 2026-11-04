# ゼミ補助資料 #52 の共通の見た目（2026-11-04）＝#45-1 の型：白地・紺の丸番号＋太字見出し・下矢印・赤と緑の強調・灰色の角丸枠。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

for p in [r"C:\Windows\Fonts\BIZ-UDGothicR.ttc", r"C:\Windows\Fonts\BIZ-UDGothicB.ttc"]:
    try:
        font_manager.fontManager.addfont(p)
    except Exception:
        pass
plt.rcParams["font.family"] = ["BIZ UDPGothic", "Meiryo"]
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#1f4e9c"
RED = "#d62728"
GREEN = "#2a9d4b"
GRAY = "#888888"
LIGHT = "#f2f2f2"
EDGE = "#c8c8c8"
INK = "#222222"


def step_title(fig, x, y, num, text, size=19):
    """紺の丸番号＋太字見出し（座標は figure の 0〜1）。"""
    ax = fig.add_axes([0, 0, 1, 1], facecolor="none"); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    w, h = fig.get_size_inches()
    r = 0.012
    ax.add_patch(matplotlib.patches.Ellipse((x, y), 2 * r, 2 * r * w / h, color=NAVY))
    ax.text(x, y, str(num), color="white", ha="center", va="center", fontsize=size * 0.8, fontweight="bold")
    ax.text(x + 0.03, y, text, color=INK, ha="left", va="center", fontsize=size, fontweight="bold")
    return ax


def down_arrow(fig, x, y0, y1):
    ax = fig.add_axes([0, 0, 1, 1], facecolor="none"); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=28, color=NAVY, lw=3))


def rounded_box(ax, x, y, w, h, fc=LIGHT, ec=EDGE, lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, transform=ax.transAxes))
