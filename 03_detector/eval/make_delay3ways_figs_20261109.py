# -*- coding: utf-8 -*-
r"""delay-3ways の図4枚（2026-11-09 Opus・本人向け）。Yu Gothic UI・日本語・白地・1800×1000。
  図A＝1曲の30秒で「点の正解」と「区間の正解」を並べ、学習後の確率の線（今の種0 と 案A 種0）と選ばれた時刻（区間の中の最大）＋右に学習が選んだ遅れの分布。
  図B＝上に正解の列（カウンターの時刻）・下に候補の列（規則）・対応の線・未対応の印＋右に遅れの分布（規則の候補・今のモデルの山）。
  図D＝3回の置き直しで正解の時刻が動く様子（1曲の抜粋）と各回の 8a。
  図まとめ＝4本（今・A・B・D）の 8a と 8b の棒＋種の幅の帯。
入力＝out/delay_3ways_20261109 の表（遅れの3案_2026-11-09.json）・B/align_*.csv・D/relabel_r*.csv・各本の probs_train.npz・learned_delay_raw_20261109.json。
出力＝G:\本人確認待ち\#63_遅れの3案_2026-11-09\図A_弱いラベル.png・図B_順番の対応付け.png・図D_反復の置き直し.png・図まとめ_4本の比べ.png
"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common_delay3ways_20261109 import ROOT, TYPES, JP, DEV, TEST, FR, SHIFT, IXG, OUT, load_index

plt.rcParams["font.family"] = "Yu Gothic UI"; plt.rcParams["axes.unicode_minus"] = False
DST = "G:/本人確認待ち/#63_遅れの3案_2026-11-09"
COL = {"shakuri": "#1f77b4", "kobushi": "#d62728", "fall": "#9467bd", "accent": "#2ca02c", "hammering": "#ff7f0e"}
W, H = 18, 10
J = json.load(open(os.path.join(OUT, "遅れの3案_2026-11-09.json"), encoding="utf-8"))
R, DL = J["R"], J["DL"]
glo = load_index(IXG); G = {d["video"]: d for d in glo}


def pick_song(split="train", lo=40.0):
    """学習の曲で、30秒に技法が多い曲（図の見本）。"""
    best = None
    for d in glo:
        if (split == "train" and d["session"] in (DEV, TEST)):
            continue
        ev = [t for c, t in d["events"] if c in TYPES and lo <= t <= lo + 30]
        n = len(set(c for c, t in d["events"] if c in TYPES and lo <= t <= lo + 30))
        key = (n, len(ev))
        if 12 <= len(ev) <= 22 and (best is None or key > best[0]):
            best = (key, d["video"])
    return best[1]


def save(fig, name):
    fig.savefig(os.path.join(DST, name), dpi=100, facecolor="white"); plt.close(fig); print("書いた", name)


def fig_A():
    v = pick_song(); d = G[v]; t0 = 40.0; t1 = t0 + 30
    pa = os.path.join(OUT, "dA_weak_8ep_s0", "probs_train.npz"); pb = os.path.join(OUT, "probs", "probs_base_all.npz")
    PA = np.load(pa)[v] if os.path.exists(pa) else None; PB = np.load(pb)[v]
    fig = plt.figure(figsize=(W, H)); gs = fig.add_gridspec(4, 3, width_ratios=[2.2, 2.2, 1.3], hspace=0.45, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, :2]); ax2 = fig.add_subplot(gs[1, :2], sharex=ax1); ax3 = fig.add_subplot(gs[2, :2], sharex=ax1); ax4 = fig.add_subplot(gs[3, :2], sharex=ax1)
    ev = [(c, t) for c, t in d["events"] if c in TYPES and t0 - 3 <= t <= t1 + 3]
    tt = np.arange(int(t0 * FR), int(t1 * FR)) / FR
    for c, t in ev:
        k = TYPES.index(c)
        ax1.plot(tt, k + 0.8 * np.exp(-0.5 * ((tt - t) / 0.1) ** 2), color=COL[c], lw=1.5)
        tc = t + SHIFT
        ax2.add_patch(plt.Rectangle((tc - 3.0, k + 0.1), 3.0, 0.8, color=COL[c], alpha=0.25, lw=0))
        ax2.plot([tc, tc], [k + 0.05, k + 0.95], color=COL[c], lw=2)
    for ax, ttl in ((ax1, "点の正解（今）＝カウンターの時刻から 1.6722秒 引いた所に山"), (ax2, "区間の正解（案A）＝カウンターの時刻 tc（縦線）の 3秒前〜tc の箱")):
        ax.set_yticks([k + 0.5 for k in range(5)]); ax.set_yticklabels([JP[c] for c in TYPES]); ax.set_ylim(0, 5); ax.set_title(ttl, loc="left", fontsize=13)
    for ax, P, ttl in ((ax3, PB, "学習後の確率（今の種0・点の正解で学習）"), (ax4, PA, "学習後の確率（案A 種0・区間の正解で学習）＝▼＝区間の中の最大＝学習が選んだ時刻")):
        ax.set_title(ttl, loc="left", fontsize=13); ax.set_ylim(0, 1.05); ax.set_ylabel("確率")
        if P is None:
            ax.text(0.5, 0.5, "案A 種0 の学習がまだ", transform=ax.transAxes, ha="center"); continue
        f0, f1 = int(t0 * FR), int(t1 * FR)
        for k, c in enumerate(TYPES):
            ax.plot(tt, P[k, f0:f1], color=COL[c], lw=1.2, label=JP[c])
        if ax is ax4:
            for c, t in ev:
                k = TYPES.index(c); tc = t + SHIFT; a0, a1 = int(round((tc - 3) * FR)), int(round(tc * FR))
                f = a0 + int(np.argmax(P[k, a0:a1 + 1]))
                if t0 <= f / FR <= t1:
                    ax.plot(f / FR, P[k, f] + 0.03, marker="v", color=COL[c], ms=9)
    ax3.legend(ncol=5, loc="upper right", fontsize=10)
    ax4.set_xlim(t0, t1); ax4.set_xlabel(f"時刻（秒）＝{v}（学習の曲）")
    axr = fig.add_subplot(gs[:, 2])
    raw = json.load(open(os.path.join(OUT, "learned_delay_raw_20261109.json"), encoding="utf-8")) if os.path.exists(os.path.join(OUT, "learned_delay_raw_20261109.json")) else {}
    bins = np.arange(0, 3.05, 0.1)
    for key, nm, col in (("now0", "今 種0（参考）", "#888888"), ("A0", "案A 種0", "#d62728"), ("A1", "案A 種1", "#ff9896"), ("A2", "案A 種2", "#e377c2"), ("Ax", "案A 変種（重さ 30）", "#2ca02c")):
        if key in raw:
            al = sum(raw[key].values(), [])
            axr.hist(al, bins=bins, histtype="step", lw=2, color=col, label=f"{nm}（中央 {np.median(al):.2f}秒）", density=True)
    axr.axvline(SHIFT, color="k", ls="--", lw=1); axr.text(SHIFT + 0.03, axr.get_ylim()[1] * 0.55, "今の固定値 1.6722", fontsize=10)
    axr.set_title("学習が選んだ遅れ（学習150曲・全技法）\n＝tc − 区間の中の確率の最大の時刻", loc="left", fontsize=13); axr.set_xlabel("遅れ（秒）"); axr.legend(fontsize=10, loc="lower center")
    fig.suptitle("図A　弱いラベル学習＝「この3秒のどこかに技法がある」で学ぶ（Opus 参考値・本人未検証）", fontsize=17, x=0.01, ha="left")
    save(fig, "図A_弱いラベル.png")


def fig_B():
    rows = list(csv.DictReader(open(os.path.join(OUT, "B", "align_rule_20261109.csv"), encoding="utf-8-sig")))
    import delay3ways_B_align_20261109 as BA
    v = pick_song("all", 40.0) if False else pick_song(); d = G[v]; t0, t1 = 40.0, 70.0
    cand = BA.cand_rule(v)
    rr = [r for r in rows if r["video"] == v]
    fig = plt.figure(figsize=(W, H)); gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 2.2, 1.4], wspace=0.25, bottom=0.16)
    ax = fig.add_subplot(gs[0, :2])
    yT, yC = 1.0, 0.0
    for r in rr:
        tc = float(r["tc"])
        if not (t0 <= tc <= t1 + 3.5):
            continue
        c = r["type"]
        if r["matched"] == "1":
            tcand = float(r["t_cand"])
            ax.plot([tc, tcand], [yT, yC], color=COL[c], lw=1.4, alpha=0.8)
            ax.plot(tc, yT, "o", color=COL[c], ms=10)
        else:
            ax.plot(tc, yT, "x", color=COL[c], ms=12, mew=3)
    used = set(round(float(r["t_cand"]), 2) for r in rr if r["matched"] == "1")
    for t, c in cand:
        if t0 <= t <= t1:
            ax.plot(t, yC, "|", color=COL[c], ms=18 if round(t, 2) in used else 10, mew=2.5 if round(t, 2) in used else 1, alpha=1 if round(t, 2) in used else 0.35)
    ax.set_xlim(t0, t1); ax.set_ylim(-0.4, 1.4); ax.set_yticks([yC, yT]); ax.set_yticklabels(["候補の列\n（規則・4技法）", "正解の列\n（カウンターの時刻 tc）"], fontsize=13)
    ax.set_xlabel(f"時刻（秒）＝{v}"); ax.grid(axis="x", alpha=0.3)
    for c in TYPES:
        ax.plot([], [], "o", color=COL[c], label=JP[c])
    ax.plot([], [], "x", color="k", mew=3, label="未対応"); ax.legend(ncol=6, loc="upper center", fontsize=11)
    ax.set_title("順番を崩さず・技法が一致・差 0.5〜3.5秒 で対応付け（太い縦棒＝対応に使った候補・薄い縦棒＝使わなかった候補）", loc="left", fontsize=13)
    axr = fig.add_subplot(gs[0, 2]); bins = np.arange(0.5, 3.55, 0.1)
    for fn, nm, col in (("align_rule_20261109.csv", "候補＝規則", "#1f77b4"), ("align_model_20261109.csv", "候補＝今のモデル種0 の山", "#d62728")):
        p = os.path.join(OUT, "B", fn)
        if not os.path.exists(p):
            continue
        al = [float(r["delay"]) for r in csv.DictReader(open(p, encoding="utf-8-sig")) if r["matched"] == "1"]
        tot = sum(1 for _ in csv.DictReader(open(p, encoding="utf-8-sig")))
        axr.hist(al, bins=bins, histtype="step", lw=2, color=col, density=True, label=f"{nm}（対応 {len(al) / tot * 100:.0f}%・中央 {np.median(al):.2f}秒）")
    axr.axvline(SHIFT, color="k", ls="--", lw=1); axr.text(SHIFT + 0.03, axr.get_ylim()[1] * 0.95, "今の固定値 1.6722", fontsize=10)
    axr.set_title("件ごとの遅れ（208曲・対応した件）\n＝tc − 候補の時刻", loc="left", fontsize=13); axr.set_xlabel("遅れ（秒）"); axr.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("図B　順番の対応付け＝カウンターの列と候補の列を順番で合わせる（Opus 参考値・本人未検証）", fontsize=17, x=0.01, ha="left")
    save(fig, "図B_順番の対応付け.png")


def fig_D():
    rs = [r for r in (1, 2, 3) if os.path.exists(os.path.join(OUT, "D", f"relabel_r{r}_20261109.csv"))]
    rows = {r: list(csv.DictReader(open(os.path.join(OUT, "D", f"relabel_r{r}_20261109.csv"), encoding="utf-8-sig"))) for r in rs}
    v = pick_song(); t0, t1 = 40.0, 70.0
    fig = plt.figure(figsize=(W, H)); gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 2.2, 1.4], wspace=0.25, bottom=0.15)
    ax = fig.add_subplot(gs[0, :2])
    base = [(c, t) for c, t in G[v]["events"] if c in TYPES and t0 <= t <= t1]
    ys = {0: 0}
    for c, t in base:
        ax.plot(t, 0, "o", color=COL[c], ms=9); ax.plot(t + SHIFT, -0.6, "|", color=COL[c], ms=14, mew=2)
    prev = {(r_["type"], r_["t_glo"]): float(r_["t_glo"]) for r_ in rows[rs[0]] if r_["video"] == v} if rs else {}
    for r in rs:
        for r_ in rows[r]:
            if r_["video"] != v:
                continue
            tg = float(r_["t_glo"])
            if not (t0 <= tg <= t1):
                continue
            c = r_["type"]; tn = float(r_["t_new"]); tp = prev.get((c, r_["t_glo"]), tg)
            ax.plot([tp, tn], [r - 1, r], color=COL[c], lw=1.2, alpha=0.7)
            ax.plot(tn, r, "o" if r_["moved_to_peak"] == "1" else "s", color=COL[c], ms=9, mfc=COL[c] if r_["moved_to_peak"] == "1" else "white")
            prev[(c, r_["t_glo"])] = tn
    ax.set_yticks([-0.6, 0] + rs); ax.set_yticklabels(["カウンターの時刻 tc", "出発＝1.6722秒 引いた時刻"] + [f"{r}回目の置き直し" for r in rs], fontsize=12)
    ax.set_ylim(-1.0, max(rs + [0]) + 0.6); ax.set_xlim(t0, t1); ax.set_xlabel(f"時刻（秒）＝{v}（学習の曲）"); ax.grid(axis="x", alpha=0.3)
    for c in TYPES:
        ax.plot([], [], "o", color=COL[c], label=JP[c])
    ax.plot([], [], "s", color="k", mfc="white", label="山に届かず前の時刻のまま"); ax.legend(ncol=6, loc="upper center", fontsize=11, bbox_to_anchor=(0.5, -0.07))
    ax.set_title("正解の時刻が回ごとに前のモデルの確率の山へ動く（[tc−3, tc] の中・8a の境目以上の山）", loc="left", fontsize=13)
    axr = fig.add_subplot(gs[0, 2])
    keys = [("now0", "出発\n（今 種0）")] + [(f"D{r}", f"{r}回目") for r in (1, 2, 3)]
    xs = [nm for k, nm in keys if k in R]; a8 = [R[k]["res"]["glo"]["0.5"]["a"]["ALL"]["hit"] for k, _ in keys if k in R]
    b8 = [R[k]["res"]["glo"]["0.5"]["b"]["ALL"]["hit"] for k, _ in keys if k in R]
    x = np.arange(len(xs))
    axr.bar(x - 0.2, a8, 0.4, color="#1f77b4", label="8a 当てた数"); axr.bar(x + 0.2, b8, 0.4, color="#aec7e8", label="8b 当てた数")
    for i in range(len(xs)):
        axr.text(x[i] - 0.2, a8[i] + 3, str(a8[i]), ha="center", fontsize=11); axr.text(x[i] + 0.2, b8[i] + 3, str(b8[i]), ha="center", fontsize=11)
    if all(k in R for k in ("now0", "now1", "now2")):
        hs = [R[k]["res"]["glo"]["0.5"]["a"]["ALL"]["hit"] for k in ("now0", "now1", "now2")]
        axr.axhspan(min(hs), max(hs), color="#1f77b4", alpha=0.12, label=f"今の種3本の幅（8a {min(hs)}〜{max(hs)}）")
    axr.set_xticks(x); axr.set_xticklabels(xs, fontsize=12); axr.set_ylabel("試験28曲で当てた数（±0.5秒・正解 982）"); axr.legend(fontsize=10, loc="upper left")
    axr.set_title("各回の当てた数", loc="left", fontsize=13); axr.set_ylim(0, max(a8 + b8 + [260]) * 1.4)
    fig.suptitle("図D　反復の置き直し＝学ぶ→山へ正解を寄せる→学び直す を3回（Opus 参考値・本人未検証）", fontsize=17, x=0.01, ha="left")
    save(fig, "図D_反復の置き直し.png")


def fig_sum():
    groups = [("今", ["now0", "now1", "now2"], "#7f7f7f"), ("案A\n弱いラベル", ["A0", "A1", "A2"], "#d62728"), ("案B\n順番の対応付け", ["B"], "#1f77b4"), ("案D\n反復の置き直し", ["D1", "D2", "D3"], "#2ca02c")]
    fig, axs = plt.subplots(1, 2, figsize=(W, H)); fig.subplots_adjust(wspace=0.3, left=0.06, right=0.97)
    for ax, m, ttl in ((axs[0], "a", "8a＝調整の30曲で出す数を正解の数に合わせた境目"), (axs[1], "b", "8b＝確率の高い順に正解の数だけ出した時")):
        for i, (nm, ks, col) in enumerate(groups):
            vals = [(k, R[k]["res"]["glo"]["0.5"][m]["ALL"]["hit"]) for k in ks if k in R]
            if not vals:
                ax.text(i, 10, "未完", ha="center"); continue
            main = vals[-1][1] if nm.startswith("案D") else vals[0][1]
            ax.bar(i, main, 0.6, color=col, alpha=0.85)
            ax.text(i, main + 4, f"{main}", ha="center", fontsize=14, fontweight="bold")
            if len(vals) > 1:
                lo, hi = min(v for _, v in vals), max(v for _, v in vals)
                ax.add_patch(plt.Rectangle((i - 0.38, lo), 0.76, max(hi - lo, 1), color=col, alpha=0.25, lw=0))
                for k, v_ in vals:
                    lab = R[k]["name"].replace("今 ", "").replace("案A ", "").replace("案D ", "")
                    if m == "a":
                        lab += f"（出した {R[k]['res']['glo']['0.5']['a']['ALL']['pred']}）"
                    ax.plot(i + 0.33, v_, "o", color="k", ms=6); ax.text(i + 0.37, v_, lab, va="center", fontsize=10)
            elif m == "a":
                ax.text(i + 0.33, main, f"（出した {R[vals[0][0]]['res']['glo']['0.5']['a']['ALL']['pred']}）", va="center", fontsize=10)
        hs = [R[k]["res"]["glo"]["0.5"][m]["ALL"]["hit"] for k in ("now0", "now1", "now2") if k in R]
        if hs:
            ax.axhspan(min(hs), max(hs), color="#7f7f7f", alpha=0.15)
            ax.text(len(groups) - 0.6, max(hs) + 3, f"灰色の帯＝今の種3本の幅 {min(hs)}〜{max(hs)}", ha="right", va="bottom", fontsize=11, color="#555555")
        ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=13)
        ax.set_ylabel("試験28曲で当てた数（±0.5秒・正解 982・1.6722秒を引いた正解で数える）", fontsize=12); ax.set_title(ttl, loc="left", fontsize=14)
        ax.set_ylim(0, 320); ax.grid(axis="y", alpha=0.3)
    fig.text(0.01, 0.02, "棒＝今と案A は種0・案B は種0（1本）・案D は3回目。帯と黒点＝種（案D は回）ごとの値の幅。8a の（出した）＝試験で出した数（正解 982）。案A は境目が上限 0.99 でも出す数が合わない本がある＝8b で比べる。", fontsize=12)
    fig.suptitle("図まとめ　遅れの扱い4通りの当てた数（Opus 参考値・本人未検証）", fontsize=17, x=0.01, ha="left")
    save(fig, "図まとめ_4本の比べ.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["A", "B", "D", "sum"]
    for w in which:
        {"A": fig_A, "B": fig_B, "D": fig_D, "sum": fig_sum}[w]()
