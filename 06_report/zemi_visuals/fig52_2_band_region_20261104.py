# #52-2 表示がある区間の判定 before/after（2026-11-04）。
# 実フレーム＝10/14 収録の1曲（加点表示が消えた後に偽の加点を読んでいた3曲のうち、件数が最も多い曲）の曲末付近。
# 線＝1フレームあたり読めたカウンターの数（6欄のうち値を出せた数）の移動平均。窓＝前後150フレーム（v0.8e と同じ _BAND_W）。
# 0.5 の線＝KV_BAND_MIN。移動平均が 0.5 以上の区間だけを「表示がある区間」とし、その外の読みを捨てる。
# before の偽の加点＝v0.8d の結果（results_wao_20261014.pre_v08e_20261015）の counter_events.csv から、区間の外の増分を拾う。
# after＝v0.8e の結果（results_wao_20261014）で、同じ区間の外に増分が無いことを確かめる。
# 30曲の合計（区間の外の偽の加点）は count_outband() で数える。
import os, sys, json, csv, re, pickle, tempfile
os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2
cv2.setNumThreads(1)
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from style_20261104 import *
import reader_core_20261104 as rc

ROOT = rc.ROOT
NEW = os.path.join(ROOT, "results_wao_20261014")
OLD = os.path.join(ROOT, "results_wao_20261014.pre_v08e_20261015")
SONG = "怪獣の花唄_20261014_104344"
START = 12300
BAND_W, BAND_MIN = 150, 0.5
OUT = sys.argv[1] if len(sys.argv) > 1 else r"G:\本人確認待ち\#52_ゼミ補助資料_2026-11-05\#52-2_表示がある区間の判定.png"
CACHE = os.path.join(tempfile.gettempdir(), f"zemi52_reads_{SONG}_{START}.pkl")


def band_of(song):
    m = re.search(r"band: present (\d+)\.\.(\d+) / (\d+)", open(os.path.join(NEW, song, "read_log.txt"), encoding="utf-8", errors="replace").read())
    return tuple(map(int, m.groups()))


def events(base, song):
    return list(csv.DictReader(open(os.path.join(base, song, "counter_events.csv"), encoding="utf-8")))


def count_outband():
    """30曲で、v0.8d が表示区間の外で出した増分（偽の加点）と、v0.8e の同じ数。"""
    old_n = new_n = 0; per = []
    for d in sorted(os.listdir(NEW)):
        if not os.path.exists(os.path.join(NEW, d, "read_log.txt")):
            continue
        bs, be, n = band_of(d)
        o = sum(int(r["delta"]) for r in events(OLD, d) if int(r["delta"]) > 0 and not (bs <= int(r["frame"]) <= be))
        nw = sum(int(r["delta"]) for r in events(NEW, d) if int(r["delta"]) > 0 and not (bs <= int(r["frame"]) <= be))
        old_n += o; new_n += nw
        if o:
            per.append((d, o))
    return old_n, new_n, per


# ---- 曲末付近を先頭から順に読む（結果は一時フォルダに置いて再利用）----
if os.path.exists(CACHE):
    D = pickle.load(open(CACHE, "rb"))
else:
    roi = json.load(open(os.path.join(NEW, SONG, "roi.json"), encoding="utf-8"))
    cap = cv2.VideoCapture(os.path.join(ROOT, "bands_wao", SONG + ".mp4")); fps = cap.get(cv2.CAP_PROP_FPS)
    for _ in range(START):
        cap.grab()
    vals = {k: [] for k in roi}; f = START - 1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        f += 1
        for k, r in roi.items():
            v = rc.read_frame(img, r)[0]
            if v is not None:
                vals[k].append((f, v))
    D = {"n": f + 1, "start": START, "fps": fps, "vals": vals}
    pickle.dump(D, open(CACHE, "wb"))

n, fps, vals = D["n"], D["fps"], D["vals"]
cnt = np.zeros(n)
for k, seq in vals.items():
    for fr, v in seq:
        cnt[fr] += 1
cum = np.concatenate([[0], np.cumsum(cnt)])
fr_axis = np.arange(START + BAND_W, n)
sm = np.array([(cum[min(n, i + BAND_W + 1)] - cum[max(0, i - BAND_W)]) / (min(n, i + BAND_W + 1) - max(0, i - BAND_W)) for i in fr_axis])
bs, be, n_log = band_of(SONG)
my_be = int(fr_axis[np.where(sm >= BAND_MIN)[0].max()])
assert my_be == be and n_log == n, f"区間の終わりが本番と合わない {my_be} {be}"
old_out = [r for r in events(OLD, SONG) if int(r["delta"]) > 0 and int(r["frame"]) > be]
new_out = [r for r in events(NEW, SONG) if int(r["delta"]) > 0 and int(r["frame"]) > be]
tot_old, tot_new, per = count_outband()
print("band end", be, "of", n, "old out", [(r["counter"], r["frame"], r["value"], r["delta"]) for r in old_out], "new out", len(new_out))
print("30 songs out-of-band false increments", tot_old, "->", tot_new, per)

# ---- 描画 ----
t = fr_axis / fps
t_be = (be + 0.5) / fps
fig = plt.figure(figsize=(20, 13), dpi=100, facecolor="white")
fig.text(0.5, 0.965, "加点表示が出ている区間だけを読む", ha="center", va="center", fontsize=27, fontweight="bold", color=INK)


def panel(y, num, title, after):
    step_title(fig, 0.06, y + 0.355, num, title)
    ax = fig.add_axes([0.08, y + 0.035, 0.86, 0.27])
    if after:
        ax.axvspan(t[0], t_be, color="#dff3e5", lw=0)
        ax.axvspan(t_be, t[-1], color="#eeeeee", lw=0)
        ax.text((t[0] + t_be) / 2, 5.5, "表示がある区間", ha="center", fontsize=17, fontweight="bold", color=GREEN)
        ax.text((t_be + t[-1]) / 2, 5.5, "区間の外の読みは捨てる", ha="center", fontsize=17, fontweight="bold", color=GRAY)
    ax.plot(t, sm, color=NAVY, lw=2.5)
    ax.axhline(BAND_MIN, color=RED, ls="--", lw=2)
    ax.text(t[-1] + 0.6, BAND_MIN, "0.5", color=RED, fontsize=16, fontweight="bold", va="center")
    for r in old_out:
        tx = int(r["frame"]) / fps
        if after:
            ax.plot([tx], [1.6], marker="x", ms=18, mew=3, color=GRAY)
            ax.text(tx, 2.0, "加点なし", ha="center", fontsize=17, fontweight="bold", color=GRAY)
        else:
            ax.plot([tx, tx], [0, 1.6], color=RED, lw=3)
            ax.plot([tx], [1.6], marker="v", ms=20, color=RED)
            ax.text(tx, 2.0, "ありもしない加点", ha="center", fontsize=17, fontweight="bold", color=RED)
    ax.annotate("加点表示が消えた", xy=(t_be, 0.05), xytext=(t_be + 4, 3.2), fontsize=15, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, 6.2)
    ax.set_ylabel("読めたカウンターの数\n移動平均", fontsize=14, color=GRAY)
    ax.tick_params(labelsize=12, colors=GRAY)
    ax.set_xlabel("曲の始まりからの秒", fontsize=13, color=GRAY)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(EDGE); ax.spines["bottom"].set_color(GRAY)


panel(0.51, 1, "区間を判定しない時　曲が終わった後の背景を数字と読む", after=False)
down_arrow(fig, 0.5, 0.505, 0.475)
panel(0.06, 2, "移動平均が0.5以上の区間だけを読む", after=True)
fig.text(0.5, 0.025, f"30曲の合計で　区間の外で読んだ偽の加点　{tot_old}件 → {tot_new}件", ha="center", fontsize=19, fontweight="bold", color=RED)
fig.savefig(OUT, facecolor="white")
print("saved", OUT)
