#!/usr/bin/env python3
"""make_figure_gran.py = 切り出し粒度の比較表から図1枚を作る（2026-10-29）

4枚組
  左上 = 値を出したコマのうち正しい割合（条件A/B/C/C2 × 手法）
  右上 = 被覆率（値を出せたコマの割合）
  左下 = 1,000件あたりの処理時間（対数目盛・専用4コア）
  右下 = 「1が11に見える」型の誤りの件数（正解1・予測11／桁の重複／桁数過多）

日本語の見出しを入れるため、日本語フォントがある側（Windows の python）で実行する。
使い方: python make_figure_gran.py --result <result dir> --out <png>
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

SHORT = {
    "ncc": "現行 見本照合NCC",
    "nn_l2": "対照 生ピクセル最近傍",
    "tess_psm7": "Tesseract psm7",
    "tess_psm8": "Tesseract psm8",
    "easyocr_rec": "EasyOCR 認識のみ",
    "easyocr_readtext": "EasyOCR 検出＋認識",
    "paddle_rec": "PaddleOCR 認識のみ",
    "paddle_full": "PaddleOCR 検出＋認識",
    "easyocr_rec_conf90": "EasyOCR 認識+確信度0.90",
    "paddle_rec_conf90": "PaddleOCR 認識+確信度0.90",
}
CONDS = ["A", "B", "C", "C2"]
CONDLABEL = {"A": "A 帯", "B": "B 技法別の窓", "C": "C 固定ピッチの桁窓", "C2": "C2 桁ごとの窓"}
COLORS = {"A": "#b0413e", "B": "#2f6f9f", "C": "#d9943a", "C2": "#4f9d69"}


def pick_font():
    for name in ("Meiryo", "Yu Gothic", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return None


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="sub4")
    a = ap.parse_args()
    f = pick_font()
    if f:
        plt.rcParams["font.family"] = f
    plt.rcParams["axes.unicode_minus"] = False

    rows = list(csv.DictReader(open(
        os.path.join(a.result, "compare_gran_%s.csv" % a.tag), encoding="utf-8")))
    cands = []
    for r in rows:
        if r["cand"] not in cands and not r["cand"].endswith("_conf90"):
            cands.append(r["cand"])
    d = {(r["cond"], r["cand"]): r for r in rows}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    y = np.arange(len(cands))
    bw = 0.2

    def grouped(ax, key, title, xlabel, log=False, pct=True):
        for j, cond in enumerate(CONDS):
            vals = [fnum(d.get((cond, c), {}).get(key, "")) for c in cands]
            ax.barh(y + (1.5 - j) * bw, vals, bw, label=CONDLABEL[cond], color=COLORS[cond])
            for k, v in enumerate(vals):
                if v == v and v > 0:
                    ax.text(v, y[k] + (1.5 - j) * bw, (" %.1f" % v) if pct else (" %.0f" % v),
                            va="center", fontsize=6.5)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT.get(c, c) for c in cands], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        if log:
            ax.set_xscale("log")
        ax.grid(axis="x", alpha=0.3)

    grouped(axes[0][0], "acc_answered", "値を出したコマのうち正しい割合", "%")
    axes[0][0].set_xlim(0, 108)
    grouped(axes[0][1], "coverage", "被覆率（値を出せたコマの割合）", "%")
    axes[0][1].set_xlim(0, 108)
    grouped(axes[1][0], "sec_per_1000_cells", "1,000件あたりの処理時間（専用4コア）",
            "秒（対数）", log=True, pct=False)
    grouped(axes[1][1], "e_more_digit", "桁数が多すぎる誤り（1が11に見える型）",
            "件", pct=False)
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle("切り出し粒度を変えた数字読み取りの比較（2026-10-29・10/14 の4曲・%s）"
                 % ("10,794件" if a.tag == "sub4" else "43,170件"), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
