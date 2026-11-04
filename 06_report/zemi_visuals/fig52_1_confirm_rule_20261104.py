# #52-1 加点の時刻の確定の規則の工程図（2026-11-04）。
# 実フレーム＝10/14 収録の帯動画（#45-1 と同じ曲・同じ帯）。読みは v0.8e と同じ判定（reader_core_20261104.py）。
# 例1＝アクセント欄 9→10（フレーム 10544〜10580）。例2＝しゃくり欄 17→18（フレーム 13440〜13529）。
# 本番の結果 results_wao_20261014/<曲>/counter_events.csv の推定時刻と一致することを assert で確かめる。
import os, sys, json, csv
os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2
cv2.setNumThreads(1)
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from style_20261104 import *
import reader_core_20261104 as rc

ROOT = rc.ROOT
SONG = "共犯_20261014_111506"
VIDEO = os.path.join(ROOT, "bands_wao", SONG + ".mp4")
RES = os.path.join(ROOT, "results_wao_20261014", SONG)
OUT = sys.argv[1] if len(sys.argv) > 1 else r"G:\本人確認待ち\#52_ゼミ補助資料_2026-11-05\#52-1_確定の規則_工程図.png"
roi = json.load(open(os.path.join(RES, "roi.json"), encoding="utf-8"))

A_NAME, A0, A1 = "accent", 10544, 10581
S_NAME, S0, S1 = "shakuri", 13440, 13530
CROPS = {10551: None, 10560: None, 10566: None, 10570: None}
SCROPS = {13454: None, 13457: None, 13479: None}

# ---- 実フレームを先頭から順に読む（シークはフレームがずれるため使わない）----
cap = cv2.VideoCapture(VIDEO)
reads = {A_NAME: {}, S_NAME: {}}
f = -1
while True:
    if f + 1 < A0:
        cap.grab(); f += 1; continue
    ok, img = cap.read()
    if not ok:
        break
    f += 1
    if A0 <= f < A1:
        reads[A_NAME][f] = rc.read_frame(img, roi[A_NAME])[0]
        if f in CROPS:
            x1, y1, x2, y2 = roi[A_NAME]
            CROPS[f] = cv2.cvtColor(img[max(0, y1 - 8):y2 + 8, x1 - 10:x2 + 10], cv2.COLOR_BGR2RGB)
    if S0 <= f < S1:
        reads[S_NAME][f] = rc.read_frame(img, roi[S_NAME])[0]
        if f in SCROPS:
            x1, y1, x2, y2 = roi[S_NAME]
            SCROPS[f] = cv2.cvtColor(img[max(0, y1 - 8):y2 + 8, x1 - 10:x2 + 10], cv2.COLOR_BGR2RGB)
    if f >= S1:
        break


def runs_of(d):
    runs = []
    for fr in sorted(d):
        v = d[fr]
        if v is None:
            continue
        if runs and runs[-1][1] == v:
            runs[-1][2].append(fr)
        else:
            runs.append([fr, v, [fr]])
    return runs


ra = runs_of(reads[A_NAME])
old_last = max(fr for fr, v in reads[A_NAME].items() if v == 9)
new_run = [r for r in ra if r[1] == 10][0]
stray = [r for r in ra if r[1] not in (9, 10)]
sixth = new_run[2][5]
event = old_last + 1
ev = {(r["counter"], int(r["frame"])): r for r in csv.DictReader(open(os.path.join(RES, "counter_events.csv"), encoding="utf-8"))}
assert int(ev[("accent", new_run[0])]["event_frame_est"]) == event, "accent の時刻が本番の結果と合わない"
rs = runs_of(reads[S_NAME])
s_old = [r for r in rs if r[1] == 17]; s_bad = [r for r in rs if r[1] == 7]; s_new = [r for r in rs if r[1] == 18]
s_last17 = max(fr for fr, v in reads[S_NAME].items() if v == 17)
assert int(ev[("shakuri", s_new[0][0])]["event_frame_est"]) == s_last17 + 1, "shakuri の時刻が本番の結果と合わない"
n17 = sum(len(r[2]) for r in s_old); n7 = sum(len(r[2]) for r in s_bad); n18 = sum(len(r[2]) for r in s_new)
print("accent: last9", old_last, "stray", [(r[0], r[1], len(r[2])) for r in stray], "first10", new_run[0], "6th", sixth, "event", event)
print("shakuri: last17", s_last17, "reads 17/7/18 in window", n17, n7, n18, "first18", s_new[0][0])

# ---- 描画 ----
fig = plt.figure(figsize=(20, 16), dpi=100, facecolor="white")
fig.text(0.5, 0.968, "読み取った数字の列から加点の時刻を1つに決める", ha="center", va="center", fontsize=27, fontweight="bold", color=INK)
base = fig.add_axes([0, 0, 1, 1], facecolor="none"); base.set_axis_off(); base.set_xlim(0, 1); base.set_ylim(0, 1)


def cell_strip(y, h, frames, vals, colors, x0=0.06, x1=0.94, fs=15):
    n = len(frames); cw = (x1 - x0) / n
    for i, (fr, v, c) in enumerate(zip(frames, vals, colors)):
        fc, tc = c
        base.add_patch(matplotlib.patches.Rectangle((x0 + i * cw + 0.0012, y), cw - 0.0024, h, fc=fc, ec="none"))
        base.text(x0 + (i + 0.5) * cw, y + h / 2, "・" if v is None else str(v), ha="center", va="center", fontsize=fs, fontweight="bold", color=tc)
    return x0, cw


C_OLD = ("#dfe6f1", NAVY); C_UN = ("#f4f4f4", "#b0b0b0"); C_RED = ("#fde3e3", RED); C_GRN = ("#d4efdc", GREEN); C_NEW = ("#eef7f0", "#6aa57b")

# ① 実フレームの読み
step_title(fig, 0.06, 0.905, 1, "読み取った値は揺れる　アクセント欄が9から10に上がる所")
xs = [0.10, 0.31, 0.52, 0.73]; labs = ["9と読めた", "光って読めない", "1と読んだ", "10と読めた"]
cols = [NAVY, GRAY, RED, GREEN]
for (fr, im), x, lb, c in zip(sorted(CROPS.items()), xs, labs, cols):
    ax = fig.add_axes([x, 0.745, 0.15, 0.115]); ax.imshow(im, interpolation="nearest", aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(c); sp.set_linewidth(3)
    base.text(x + 0.075, 0.728, lb, ha="center", va="top", fontsize=17, fontweight="bold", color=c)
    base.text(x + 0.075, 0.700, f"フレーム {fr}", ha="center", va="top", fontsize=12, color=GRAY)
for x in xs[:-1]:
    base.annotate("", xy=(x + 0.205, 0.80), xytext=(x + 0.155, 0.80), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))

down_arrow(fig, 0.5, 0.672, 0.645)

# ② 同じ値が6フレーム続いた所で確定
step_title(fig, 0.06, 0.615, 2, "同じ値が6フレーム続いた所で確定")
frs = list(range(A0, A1)); vs = [reads[A_NAME][f] for f in frs]
colors = []
for fr, v in zip(frs, vs):
    if v is None: colors.append(C_UN)
    elif v == 9: colors.append(C_OLD)
    elif v == 10: colors.append(C_GRN if fr <= sixth else C_NEW)
    else: colors.append(C_RED)
x0, cw = cell_strip(0.53, 0.045, frs, vs, colors)
i_st = frs.index(stray[0][0]); i_n0 = frs.index(new_run[0]); i_6 = frs.index(sixth)
base.text(x0 + (i_st + 0.5) * cw, 0.52, "1回だけ\n捨てる", ha="center", va="top", fontsize=15, fontweight="bold", color=RED, linespacing=1.4)
base.add_patch(matplotlib.patches.Rectangle((x0 + i_n0 * cw, 0.526), (i_6 - i_n0 + 1) * cw, 0.053, fill=False, ec=GREEN, lw=3))
base.text(x0 + (i_n0 + i_6 + 1) / 2 * cw, 0.52, "6回続いた\n10で確定", ha="center", va="top", fontsize=15, fontweight="bold", color=GREEN, linespacing=1.4)
base.text(0.06, 0.585, "1マス＝1フレーム　・＝読めない", ha="left", va="bottom", fontsize=12, color=GRAY)

down_arrow(fig, 0.5, 0.462, 0.435)

# ③ 増える向きだけ
step_title(fig, 0.06, 0.405, 3, "値は増える向きだけ採る　しゃくり欄が17から18に上がる所")
blocks = [("17 と読めた", C_OLD, 13454), (f"7 と読んだ\n{n7}回続いた", C_RED, 13457), ("18 と読めた", C_GRN, 13479)]
bx = [0.08, 0.385, 0.69]
for (txt, (fc, tc), sf), x in zip(blocks, bx):
    base.add_patch(FancyBboxPatch((x, 0.292), 0.23, 0.083, boxstyle="round,pad=0.004,rounding_size=0.01", fc=fc, ec=tc, lw=2.5))
    ax = fig.add_axes([x + 0.008, 0.298, 0.085, 0.071]); ax.imshow(SCROPS[sf], interpolation="nearest", aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(tc); sp.set_linewidth(2)
    base.text(x + 0.163, 0.3335, txt, ha="center", va="center", fontsize=20, fontweight="bold", color=tc, linespacing=1.3)
for x in bx[:-1]:
    base.annotate("", xy=(x + 0.302, 0.3335), xytext=(x + 0.235, 0.3335), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2.5))
base.text(0.50, 0.282, "前の17より小さいので捨てる", ha="center", va="top", fontsize=16, fontweight="bold", color=RED)
base.text(0.805, 0.282, "増えたので採る", ha="center", va="top", fontsize=16, fontweight="bold", color=GREEN)

down_arrow(fig, 0.5, 0.232, 0.205)

# ④ 時刻
step_title(fig, 0.06, 0.175, 4, "時刻＝前の値が最後に読めたフレームの次のフレーム")
frs4 = list(range(10548, 10578)); vs4 = [reads[A_NAME][f] for f in frs4]
c4 = []
for fr, v in zip(frs4, vs4):
    if fr == event: c4.append(C_GRN)
    elif v == 9: c4.append(C_OLD)
    elif v == 10: c4.append(C_NEW)
    else: c4.append(C_UN)
vs4 = [None if (v is not None and v not in (9, 10)) else v for v in vs4]
x0, cw = cell_strip(0.095, 0.045, frs4, vs4, c4, fs=14)
i_ev = frs4.index(event); i_ol = frs4.index(old_last); i_6b = frs4.index(sixth)
base.add_patch(matplotlib.patches.Rectangle((x0 + i_ev * cw, 0.091), cw, 0.053, fill=False, ec=GREEN, lw=3.5))
base.annotate("加点の時刻", xy=(x0 + (i_ev + 0.5) * cw, 0.09), xytext=(x0 + (i_ev + 0.5) * cw + 0.04, 0.035), ha="center", va="center",
              fontsize=17, fontweight="bold", color=GREEN, arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.5))
base.annotate("最後の9", xy=(x0 + (i_ol + 0.5) * cw, 0.09), xytext=(x0 + (i_ol + 0.5) * cw - 0.06, 0.035), ha="center", va="center",
              fontsize=15, fontweight="bold", color=NAVY, arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2))
base.annotate("確定したのはここ", xy=(x0 + (i_6b + 0.5) * cw, 0.09), xytext=(x0 + (i_6b + 0.5) * cw, 0.035), ha="center", va="center",
              fontsize=15, color=GRAY, arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))

fig.savefig(OUT, facecolor="white")
print("saved", OUT)
