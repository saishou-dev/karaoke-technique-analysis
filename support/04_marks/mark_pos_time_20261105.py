# -*- coding: utf-8 -*-
"""案D（ページ表示版）手順2〜6＝カーソル直線から歌った時刻 D を戻し、カウンター時刻・マーク出現時刻・C と比べる・2026-11-05 Opus
入力＝cursor_track_20261105.py の tracks/*.csv と対応件（9/24＝G1_events_v2_20260924.csv・10/14＝mark_lead_events＋marks_20261014.csv）。
ページ＝カーソル x が t に対し一次で増える区間。マーク前後 [mt-6, mt+0.6] の追跡点に RANSAC で直線（x＝a＋v·t）を順に当て、
1本ごとに当たった点を除いて次の直線を探す（v はページごとに当てる＝一定を仮定しない）。
D＝カーソルが mark_x を通った時刻＝(mark_x-a)/v。出現時刻 mt の時点の直線（mt 以前に始まった最後のページ）で解き、
D>mt＋TOL（まだ通っていない＝ページ替わり直後）なら一つ前のページの直線で解く。どちらでも範囲に入らなければ「不明」。
比較＝(a) カウンター-D (b) マーク出現-D (c) C-D（C＝カウンター-技法別中央値〈日別・カウンター-マーク出現〉）。
判定＝技法別・全体で IQR(カウンター-D) が IQR(カウンター-マーク出現)＝C の残差のばらつき より20%以上小さいか。
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cursor_track_20261105 import load_pairs, OUT, PRE, POST

TOL = 0.10            # 出現時刻より後の D を許す幅［秒］（出現の検出は持続判定で数フレーム遅れうる）
INL = 70              # 直線への当たり幅［px・4K］（カーソルは段で進み直線から±40px ほど振れる）
VMIN, VMAX = 150, 2500
MIN_IN, MIN_SPAN = 12, 0.30
MAD_MAX = 35          # 直線への当たり点の残差（中央絶対値）の上限［px］＝誤検出をつないだ直線を落とす（カーソルは約0.13秒ごとに約70px の段で進むので段の分は許す）
EXT = 0.6             # 直線を観測点の外へ延ばしてよい幅［秒］
rng = np.random.default_rng(0)
TECH_JA = dict(vibrato="ビブラート", shakuri="しゃくり", kobushi="こぶし", accent="アクセント", hammering="ハンマリング", fall="フォール")
TECHS = ["vibrato", "shakuri", "kobushi", "accent", "hammering", "fall"]


def fit_lines(t, x):
    """RANSAC で直線を順に取る。返り値＝[(a, v, t_first, t_last, n_in, resid_mad)] を t_first 順"""
    lines = []; idx = np.arange(len(t))
    while len(idx) >= MIN_IN:
        tt, xx = t[idx], x[idx]; best = None
        for _ in range(300):
            i, j = rng.choice(len(idx), 2, replace=False)
            if abs(tt[j] - tt[i]) < 0.08: continue
            v = (xx[j] - xx[i]) / (tt[j] - tt[i])
            if not (VMIN <= v <= VMAX): continue
            a = xx[i] - v * tt[i]
            inl = np.abs(xx - (a + v * tt)) <= INL
            if best is None or inl.sum() > best.sum(): best = inl
        if best is None or best.sum() < MIN_IN: break
        for _ in range(2):   # 当たり点で最小二乗し直し→当たりを取り直す
            v, a = np.polyfit(tt[best], xx[best], 1)
            best = np.abs(xx - (a + v * tt)) <= INL
        if best.sum() < 2: break
        # 時間的に連続な塊だけ（0.8秒以上の空白で切り、点の多い塊）
        ts = np.sort(tt[best]); gaps = np.where(np.diff(ts) > 0.8)[0]
        seg = max(np.split(ts, gaps + 1), key=len)
        keep = best & (tt >= seg[0]) & (tt <= seg[-1])
        if keep.sum() >= MIN_IN and seg[-1] - seg[0] >= MIN_SPAN:
            v, a = np.polyfit(tt[keep], xx[keep], 1)
            r = xx[keep] - (a + v * tt[keep])
            if VMIN <= v <= VMAX and np.median(np.abs(r)) <= MAD_MAX:
                lines.append((a, v, float(seg[0]), float(seg[-1]), int(keep.sum()), float(np.median(np.abs(r)))))
            idx = idx[~keep]
        else:
            idx = idx[~best]
    # 同じページの二重線（カーソルの縦線と光の筋が並んで取れる）＝時間が半分以上重なる直線は点の多い方だけ残す
    lines = sorted(lines, key=lambda l: -l[4]); kept = []
    for l in lines:
        if any(min(l[3], m[3]) - max(l[2], m[2]) > 0.5 * min(l[3] - l[2], m[3] - m[2]) for m in kept): continue
        kept.append(l)
    return sorted(kept, key=lambda l: l[2])


def drop_static(T, X, run=20, tol=2.0, gap=0.05):
    """静止した白い物（長い白バー・文字）の誤検出を除く＝連続フレームで x が tol px 以内のまま run 点以上続く塊を捨てる
    （本物のカーソルは約0.13秒＝8フレームごとに段で進む＝20フレーム以上動かない塊は静止物とみなす）"""
    keep = np.ones(len(T), bool); i = 0
    while i < len(T):
        j = i
        while j + 1 < len(T) and abs(X[j + 1] - X[i]) <= tol and T[j + 1] - T[j] <= gap: j += 1
        if j - i + 1 >= run: keep[i:j + 1] = False
        i = j + 1
    return T[keep], X[keep]


def solve(mt, mx, lines, xend, xstart):
    """返り値＝(D, 使ったページ, v, 不明の理由)。xend／xstart＝その曲のページ右端／左端（カーソル x の98%点／2%点）。
    ページ＝cur（出現時刻のページ・観測点の前後 EXT 秒以内）／cur_ext（同じページの直線を観測点の外へ延ばした＝ページの左右端の内側・3秒以内）／prev（前ページ）"""
    if not lines: return np.nan, "", np.nan, "直線なし"
    started = [k for k, l in enumerate(lines) if l[2] <= mt + TOL]
    if not started: return np.nan, "", np.nan, "出現時刻までに始まったページなし"
    k = started[-1]; a, v, t0, t1, n, _ = lines[k]
    if a + v * mt > xend + 60:   # この直線のページは出現時刻までに終わっている＝出現時のページが取れていない
        return np.nan, "", np.nan, "出現時刻のページのカーソル未検出"
    D = (mx - a) / v
    prev_end = lines[k - 1][3] if k > 0 else -np.inf
    if t0 - EXT <= D <= min(mt, t1 + EXT) + TOL:
        return D, "cur", v, ""
    if D > mt + TOL and k > 0:   # 出現時刻にまだ通っていない＝前ページに属する
        a2, v2, s0, s1, _, _ = lines[k - 1]
        D2 = (mx - a2) / v2
        if s0 - EXT <= D2 <= t0 + TOL:
            return D2, "prev", v2, ""
        return np.nan, "", np.nan, "前ページの直線の範囲外"
    if D > mt + TOL: return np.nan, "", np.nan, "前ページの直線なし"
    # 同じページの中で直線を延ばす（カーソルの検出がページの頭／出現時刻の手前で途切れた件）
    if mx >= xstart - 60 and D > prev_end and D >= t0 - 3.0 and D <= mt + TOL:
        return D, "cur_ext", v, ""
    if D > t1 + EXT + TOL: return np.nan, "", np.nan, "今ページの観測より後（カーソル途切れ）"
    return np.nan, "", np.nan, "今ページの始まりより前"


def q(s, p): return float(np.nanpercentile(s, p)) if len(s) else np.nan


def stats(s):
    s = pd.Series(s).dropna()
    return dict(n=len(s), med=q(s, 50), q1=q(s, 25), q3=q(s, 75), iqr=q(s, 75) - q(s, 25), p10=q(s, 10), p90=q(s, 90),
                over2=float((s.abs() > 2).mean()) if len(s) else np.nan)


def main():
    rows = []; vtab = []
    for ses in (20260924, 20261014):
        P = load_pairs(ses)
        P["lead"] = P.video_time_sec - P.mark_time_sec
        P["C"] = P.video_time_sec - P.groupby("counter").lead.transform("median")
        for v, g in P.groupby("video"):
            fn = os.path.join(OUT, "tracks", f"{ses}__{v}.csv")
            if not os.path.exists(fn):
                for _, r in g.iterrows(): rows.append(dict(r, D=np.nan, page="", v_pxs=np.nan, why="追跡なし", n_lines=0))
                continue
            tr = pd.read_csv(fn).dropna().sort_values("pts_sec")
            T, X = tr.pts_sec.to_numpy(float), tr.cursor_x_4k.to_numpy(float)
            T, X = drop_static(T, X)
            xend = float(np.percentile(X, 98)) if len(X) else np.inf
            xstart = float(np.percentile(X, 2)) if len(X) else -np.inf
            for _, r in g.iterrows():
                mt, mx = float(r.mark_time_sec), float(r.mark_x)
                sel = (T >= mt - PRE) & (T <= mt + POST)
                lines = fit_lines(T[sel], X[sel])
                D, page, vv, why = solve(mt, mx, lines, xend, xstart)
                rows.append(dict(r, D=D, page=page, v_pxs=vv, why=why, n_lines=len(lines)))
                for l in lines:
                    vtab.append(dict(session=ses, video=v, t_first=round(l[2], 2), t_last=round(l[3], 2), v_pxs=round(l[1], 1), n_pts=l[4], resid_mad_px=round(l[5], 1)))
    E = pd.DataFrame(rows)
    E["a_counter_minus_D"] = E.video_time_sec - E.D
    E["b_mark_minus_D"] = E.mark_time_sec - E.D
    E["c_C_minus_D"] = E.C - E.D
    cols = ["session", "video", "counter", "count_value", "video_time_sec", "mark_time_sec", "mark_x", "mark_y", "lead", "C", "D", "page", "v_pxs", "n_lines", "why",
            "a_counter_minus_D", "b_mark_minus_D", "c_C_minus_D"]
    E[cols].to_csv(os.path.join(OUT, "D_times_20261105.csv"), index=False, encoding="utf-8-sig")
    V = pd.DataFrame(vtab).drop_duplicates(["session", "video", "t_first", "t_last"])
    # 窓の重なりで同じページが複数回取れる＝曲ごとに時間が半分以上重なるものは点の多い方だけ残す
    keepV = []
    for (s_, v_), g in V.sort_values("n_pts", ascending=False).groupby(["session", "video"], sort=False):
        kept = []
        for _, r in g.iterrows():
            if any(min(r.t_last, m.t_last) - max(r.t_first, m.t_first) > 0.5 * min(r.t_last - r.t_first, m.t_last - m.t_first) for m in kept): continue
            kept.append(r)
        keepV += kept
    V = pd.DataFrame(keepV).sort_values(["session", "video", "t_first"])
    V.to_csv(os.path.join(OUT, "cursor_pages_v_20261105.csv"), index=False, encoding="utf-8-sig")
    # 手順2の表＝曲ごとに曲頭・中・末（ページを時刻順に3等分）の v の中央値と差
    vt = []
    for (s_, v_), g in V.groupby(["session", "video"]):
        parts = np.array_split(g.sort_values("t_first").v_pxs.to_numpy(), 3) if len(g) >= 3 else [g.v_pxs.to_numpy()]
        meds = [float(np.median(p)) for p in parts if len(p)]
        vt.append(dict(session=s_, video=v_, n_pages=len(g), v_head=meds[0], v_mid=meds[1] if len(meds) == 3 else np.nan,
                       v_tail=meds[-1] if len(meds) == 3 else np.nan, v_min=g.v_pxs.min(), v_med=g.v_pxs.median(), v_max=g.v_pxs.max(),
                       spread_3part=(max(meds) / min(meds) - 1) if len(meds) == 3 else np.nan,
                       spread_pages=(g.v_pxs.quantile(.9) / g.v_pxs.quantile(.1) - 1)))
    VT = pd.DataFrame(vt); VT.to_csv(os.path.join(OUT, "v_by_song_20261105.csv"), index=False, encoding="utf-8-sig")
    print(VT.round(2).to_string())
    R_all = []
    for variant, mask in [("全件（延長あり）", E.D.notna()), ("厳密（延長なし）", E.page.isin(["cur", "prev"]))]:
        E2 = E.copy(); E2.loc[~mask, "D"] = np.nan
        for col in ["a_counter_minus_D", "b_mark_minus_D", "c_C_minus_D"]: E2.loc[~mask, col] = np.nan
        okm = E2.D.notna()
        E2["lead_res"] = np.nan; E2["a_res"] = np.nan; E2["c_res"] = np.nan
        for (s_, c_), idx in E2[okm].groupby(["session", "counter"]).groups.items():
            E2.loc[idx, "lead_res"] = E2.loc[idx, "lead"] - E2.loc[idx, "lead"].median()
            E2.loc[idx, "a_res"] = E2.loc[idx, "a_counter_minus_D"] - E2.loc[idx, "a_counter_minus_D"].median()
            E2.loc[idx, "c_res"] = E2.loc[idx, "c_C_minus_D"] - E2.loc[idx, "c_C_minus_D"].median()
        R_all.append(table(E2, variant))
    R = pd.concat(R_all, ignore_index=True)
    R.to_csv(os.path.join(OUT, "compare_20261105.csv"), index=False, encoding="utf-8-sig")
    finish(E, R, V, VT)


def table(E, variant):
    out = []
    groups = [("全体", E)] + [(TECH_JA[t], E[E.counter == t]) for t in TECHS]
    for sesname, S in [("両日", None), ("9/24", 20260924), ("10/14", 20261014)]:
        for name, G in groups:
            if S is not None: G = G[G.session == S]
            ok = G[G.D.notna()]
            base = stats(ok.lead); a = stats(ok.a_counter_minus_D); b = stats(ok.b_mark_minus_D); c = stats(ok.c_C_minus_D)
            # ばらつき＝日×技法の中央値を引いた残差の IQR（1日・1技法なら素の IQR と同じ）
            base["iqr"] = stats(ok.lead_res)["iqr"]; a["iqr"] = stats(ok.a_res)["iqr"]; c["iqr"] = stats(ok.c_res)["iqr"]
            red = 1 - a["iqr"] / base["iqr"] if base["iqr"] else np.nan
            out.append(dict(版=variant, 日=sesname, 技法=name, 対応=len(G), D取得=len(ok), 不明=len(G) - len(ok), 前ページ=int((ok.page == "prev").sum()),
                            延長=int((ok.page == "cur_ext").sum()),
                            C基準_中央=base["med"], C基準_Q1=base["q1"], C基準_Q3=base["q3"], C基準_IQR=base["iqr"], C基準_p90=base["p90"], C基準_2秒超=base["over2"],
                            a_中央=a["med"], a_Q1=a["q1"], a_Q3=a["q3"], a_IQR=a["iqr"], a_p90=a["p90"], a_2秒超=a["over2"],
                            b_中央=b["med"], b_IQR=b["iqr"], b_p10=b["p10"], b_p90=b["p90"],
                            b_負=float((ok.b_mark_minus_D < 0).mean()) if len(ok) else np.nan,
                            c_中央=c["med"], c_IQR=c["iqr"], c_p90=c["p90"], c_2秒超=c["over2"],
                            IQR縮み=red, 判定=(("D 有効" if red >= 0.2 else "D 無効") if pd.notna(red) else ""),
                            C基準_IQR_全対応=stats(G.lead)["iqr"]))
    return pd.DataFrame(out)


def finish(E, R, V, VT):
    why = E[E.D.isna()].groupby(["session", "why"]).size().reset_index(name="件数")
    why.to_csv(os.path.join(OUT, "unknown_reasons_20261105.csv"), index=False, encoding="utf-8-sig")
    f = lambda x: "" if (not isinstance(x, str) and pd.isna(x)) else (f"{x:.3f}" if isinstance(x, (float, np.floating)) else str(x))
    with open(os.path.join(OUT, "compare_20261105.md"), "w", encoding="utf-8") as fh:
        fh.write("### 案D（ページ表示版）比較表・2026-11-05 Opus 参考値・本人未検証\n\n")
        fh.write("C基準＝カウンター-マーク出現（D の取れた件だけ・C はこの中央値を引く補正なので残差の IQR はこれと同じ）。IQR はすべて日×技法の中央値を引いた残差で測る（両日・全体の行でも C と同じ補正をした後のばらつき）。版＝全件（同じページの直線を観測点の外へ延ばした件を含む）／厳密（観測点の前後0.6秒以内と前ページだけ）。a＝カウンター-D。b＝マーク出現-D。c＝C-D（C＝カウンター-技法別中央値・日別）。IQR縮み＝1-IQR(a)/IQR(C基準)。判定＝縮み20%以上で「D 有効」。単位＝秒（割合は0〜1）。\n\n")
        hd = ["版", "日", "技法", "対応", "D取得", "不明", "前ページ", "延長", "C基準_中央", "C基準_IQR", "C基準_p90", "C基準_2秒超", "a_中央", "a_Q1", "a_Q3", "a_IQR", "a_p90", "a_2秒超",
              "b_中央", "b_IQR", "b_負", "c_IQR", "c_2秒超", "IQR縮み", "判定"]
        fh.write("| " + " | ".join(hd) + " |\n|" + "---|" * len(hd) + "\n")
        for _, r in R.iterrows(): fh.write("| " + " | ".join(f(r[h]) for h in hd) + " |\n")
        fh.write("\n### 不明の理由\n\n| 日 | 理由 | 件数 |\n|---|---|---|\n")
        for _, r in why.iterrows(): fh.write(f"| {r.session} | {r.why} | {r['件数']} |\n")
        d = V.groupby("session").v_pxs.describe(percentiles=[.1, .5, .9]).round(1)
        fh.write("\n### ページごとのカーソルの速さ v［px/秒・4K］\n\n| 日 | ページ数 | 平均 | 標準偏差 | p10 | 中央 | p90 |\n|---|---|---|---|---|---|---|\n")
        for s, r in d.iterrows(): fh.write(f"| {s} | {int(r['count'])} | {r['mean']} | {r['std']} | {r['10%']} | {r['50%']} | {r['90%']} |\n")
        fh.write("\n### 曲ごとの v（曲頭・中・末＝ページを時刻順に3等分した中央値・差＝最大/最小−1）\n\n| 日 | 曲 | ページ数 | 曲頭 | 中 | 末 | 3区間の差 | ページ間 p90/p10−1 |\n|---|---|---|---|---|---|---|---|\n")
        for _, r in VT.iterrows():
            fh.write(f"| {r.session} | {r.video} | {r.n_pages} | {f(r.v_head)} | {f(r.v_mid)} | {f(r.v_tail)} | {f(r.spread_3part)} | {f(r.spread_pages)} |\n")
        fh.write(f"\n3区間の差が10%超の曲＝{int((VT.spread_3part > 0.10).sum())}／{int(VT.spread_3part.notna().sum())}曲。\n")
    print(why.to_string()); print(E.page.value_counts(dropna=False))
    print(R[["版", "日", "技法", "対応", "D取得", "不明", "前ページ", "延長", "C基準_IQR", "a_IQR", "a_中央", "b_中央", "b_IQR", "b_負", "c_IQR", "IQR縮み", "判定"]].round(3).to_string())
    print(V.groupby("session").v_pxs.describe().round(1).to_string())
    plot(E)


def plot(E):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for fn in ["Yu Gothic", "Meiryo", "MS Gothic"]:
        if fn in names: plt.rcParams["font.family"] = fn; break
    ok = E[E.D.notna()]
    techs = [t for t in TECHS if (ok.counter == t).any()]
    fig, ax = plt.subplots(figsize=(10.5, 5.4), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
    pos = np.arange(len(techs) + 1) * 3.0
    data_c = [ok.lead] + [ok[ok.counter == t].lead for t in techs]
    data_d = [ok.a_counter_minus_D] + [ok[ok.counter == t].a_counter_minus_D for t in techs]
    kw = dict(widths=0.9, patch_artist=True, showfliers=True, whis=(10, 90),
              flierprops=dict(marker="o", markersize=2.5, alpha=0.35, markeredgewidth=0), medianprops=dict(color="#0b0b0b", linewidth=1.5))
    b1 = ax.boxplot(data_c, positions=pos - 0.55, **kw); b2 = ax.boxplot(data_d, positions=pos + 0.55, **kw)
    for b, col in [(b1, "#2a78d6"), (b2, "#eb6834")]:
        for p in b["boxes"]: p.set_facecolor(col); p.set_alpha(0.85); p.set_edgecolor(col)
        for k in ("whiskers", "caps"):
            for l in b[k]: l.set_color("#52514e")
        for l in b["fliers"]: l.set_markerfacecolor(col)
    ax.set_xticks(pos); ax.set_xticklabels([f"全体\n(n={len(ok)})"] + [f"{TECH_JA[t]}\n(n={(ok.counter == t).sum()})" for t in techs], color="#0b0b0b")
    ax.set_ylabel("カウンター時刻との差［秒］", color="#52514e"); ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    lo = min(np.nanpercentile(ok.a_counter_minus_D, 1), np.nanpercentile(ok.lead, 1)); hi = max(np.nanpercentile(ok.a_counter_minus_D, 99), np.nanpercentile(ok.lead, 99))
    ax.set_ylim(min(-0.5, lo - 0.2), hi + 0.6)
    ax.legend([b1["boxes"][0], b2["boxes"][0]], ["カウンター − マーク出現（C の基準）", "カウンター − D（カーソルが mark_x を通った時刻）"], loc="upper right", frameon=False)
    ax.set_title("遅れの分布・技法別（9/24＋10/14・D の取れた件・箱＝四分位・ひげ＝p10〜p90）", color="#0b0b0b", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "box_C_vs_D_20261105.png"), dpi=150, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
