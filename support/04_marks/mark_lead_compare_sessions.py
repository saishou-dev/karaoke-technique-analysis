# -*- coding: utf-8 -*-
"""カウンター増分の時刻と技法マーク出現の時刻の差（mark_dt）を、2収録で同じ型に出して比べる（2026-10-30）。
定義は build_v2_and_eval.py の build_v2() と同一：
  mark_dt = T(カウンター増分＝G1 video_time_sec) − T(マーク出現＝marks_<S>.csv の mark_time_sec)［秒・正＝マークが先］
  対応＝G1 と marks を (video, counter, count_value) で突き合わせ、mark_time_sec が入っている行＝「対応」
  対象＝multi_increment=0 の行のみ・期待範囲 0.2〜1.5 秒・「ゲート無効化」＝mark_note に「無効化」
  対応窓＝detect_marks.py（ビブラート 3.0 秒・他 2.2 秒）＋refine_onsets.py（さらに 2.6 秒さかのぼる）＝両収録とも同じ版の論理
入力:
  比較元  marks_20260928/G1_events_v2_<BASE>.csv（build_v2 が 10/08 に書いた行単位の正本＝mark_lead_stats_<BASE>.md と同じ 1182/484 件）
  比較先  marks_20260928/marks_<S>.csv（refine 済みの現行）＋ dataset/G1_events_<S>_wao.csv（正解＝カウンター増分）
  観測区間 dataset/song_valid_range.csv（感度確認だけに使う＝主の表は 9/24 と同じく区間で絞らない）
出力（marks_20260928/・すべて新規）:
  mark_lead_events_<S>.csv              … 行単位（G1 の列＋mark_time_sec・mark_dt_sec・mark_status・in_valid_range）
  mark_lead_stats_<S>.csv/.md           … 9/24 と同じ列の技法別・曲別の表
  mark_lead_compare_<BASE>_<S>.csv/.md  … 技法別中央値の並び・差・Mann-Whitney U（両側）・Holm 補正・観測区間の感度
  mark_lead_box_<BASE>_<S>.png          … 技法別の箱ひげ（2収録を並べる）
使い方: python mark_lead_compare_sessions.py --session 20261014 --base 20260924"""
import os, sys, argparse
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_v2_and_eval import BASE, OUT_DIR, TYPES, JA, EXPECT, q   # 定義は 9/24 と同じものを読み込む

ORDER = ["vibrato", "shakuri", "kobushi", "accent", "hammering", "fall"]

def stats_rows(ev):
    rows = []
    for c, gg in [(c, ev[ev.counter == c]) for c in sorted(ev.counter.unique())] + [("ALL", ev)]:
        ok = gg[gg.mark_status == "mark"]; d = ok.mark_dt_sec
        rows.append(dict(technique=c, technique_ja=JA[c], n_events=len(gg), n_mark=len(ok), rate=round(len(ok) / len(gg), 3) if len(gg) else np.nan,
                         rate_none=round(1 - len(ok) / len(gg), 3) if len(gg) else np.nan,
                         dt_median=q(d, .5), dt_q1=q(d, .25), dt_q3=q(d, .75), dt_p10=q(d, .1), dt_p90=q(d, .9), dt_mean=round(float(d.mean()), 3) if len(d) else np.nan,
                         in_expect=round(float(((d >= EXPECT[0]) & (d <= EXPECT[1])).mean()), 3) if len(d) else np.nan,
                         n_negative=int((d < 0).sum()), n_invalidated=int(gg.mark_note.fillna("").str.contains("無効化").sum()),
                         dt_max=round(float(d.max()), 3) if len(d) else np.nan, over2s=round(float((d > 2.0).mean()), 3) if len(d) else np.nan))
    return rows

def build_session(s):
    marks = pd.read_csv(os.path.join(OUT_DIR, f"marks_{s}.csv"))
    g = pd.read_csv(os.path.join(BASE, "dataset", f"G1_events_{s}_wao.csv"))
    # 突き合わせの前提確認：G1_events_all の同収録と一致するか・marks の video_time_sec と G1 が一致するか
    ga = pd.read_csv(os.path.join(BASE, "dataset", "G1_events_all.csv")); ga = ga[ga.session == int(s)]
    k = ["video", "counter", "count_value"]
    chk = g.merge(ga[k + ["video_time_sec", "multi_increment"]], on=k, how="outer", suffixes=("", "_all"), indicator=True)
    n_all_mismatch = int((chk._merge != "both").sum() + (chk.video_time_sec.round(3) != chk.video_time_sec_all.round(3)).sum())
    m = marks[["video", "counter", "count_value", "video_time_sec", "mark_time_sec", "conf", "method", "note"]].rename(
        columns={"video_time_sec": "marks_video_time_sec", "conf": "mark_conf", "method": "mark_method", "note": "mark_note"})
    v2 = g.merge(m, on=k, how="left")
    n_marks_unmatched = int(len(marks) - v2.marks_video_time_sec.notna().sum())
    n_time_mismatch = int(((v2.marks_video_time_sec - v2.video_time_sec).abs() > 1e-3).sum())
    v2["mark_dt_sec"] = (v2.video_time_sec - v2.mark_time_sec).round(3)
    v2["mark_status"] = np.where(v2.mark_time_sec.notna(), "mark", np.where(v2.multi_increment == 1, "skipped_multi", "none"))
    v2["mark_note"] = v2.mark_note.fillna("")
    chk_info = dict(g1_rows=len(g), g1_all_mismatch=n_all_mismatch, marks_rows=len(marks), marks_unmatched=n_marks_unmatched, time_mismatch=n_time_mismatch)
    return v2, chk_info

def add_valid_range(v, s):
    vr = pd.read_csv(os.path.join(BASE, "dataset", "song_valid_range.csv")); vr = vr[vr.session == int(s)]
    end = vr.band_last_pts.where(vr.band_last_pts.notna(), vr.last_frame_pts)
    r = pd.DataFrame(dict(video=vr.video, vr_start=vr.first_frame_pts, vr_end=end))
    v = v.merge(r, on="video", how="left")
    v["in_valid_range"] = ((v.video_time_sec >= v.vr_start) & (v.video_time_sec <= v.vr_end)).astype(int)
    return v

def md_stats(rows, ev, s):
    L = [f"### 差の分布（{s}・mark_dt = T(カウンター増分) − mark 出現、秒。multi_increment=0 の {len(ev)} 件）", "",
         "| 技法 | 件数 | 対応 | 対応率 | 未対応率 | 中央値 | Q1 | Q3 | p10 | p90 | 平均 | 0.2〜1.5s 内 | 負(マークが後) | ゲート無効化 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['technique_ja']} | {r['n_events']} | {r['n_mark']} | {r['rate']:.1%} | {r['rate_none']:.1%} | {r['dt_median']} | {r['dt_q1']} | {r['dt_q3']} | {r['dt_p10']} | {r['dt_p90']} | {r['dt_mean']} | {r['in_expect']:.1%} | {r['n_negative']} | {r['n_invalidated']} |")
    bv = ev.groupby("video").apply(lambda x: pd.Series(dict(n=len(x), n_mark=int((x.mark_status == "mark").sum()), dt_median=q(x.mark_dt_sec[x.mark_status == "mark"], .5)))).reset_index()
    bv["rate"] = (bv.n_mark / bv.n).round(3)
    L += ["", "### 曲別", "", "| 曲 | 件数 | 対応 | 対応率 | 差の中央値 |", "|---|---|---|---|---|"]
    for _, r in bv.iterrows(): L.append(f"| {r.video} | {int(r.n)} | {int(r.n_mark)} | {r.rate:.1%} | {r.dt_median} |")
    return L, bv

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(o):
        run = max(run, min(1.0, (m - i) * ps[idx])); adj[idx] = run
    return adj

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--session", default="20261014"); ap.add_argument("--base", default="20260924")
    a = ap.parse_args(); s, b = a.session, a.base
    v2, chk = build_session(s)
    v2 = add_valid_range(v2, s)
    v2.to_csv(os.path.join(OUT_DIR, f"mark_lead_events_{s}.csv"), index=False, encoding="utf-8-sig")
    ev = v2[v2.multi_increment == 0]
    rows = stats_rows(ev); st = pd.DataFrame(rows)
    st.drop(columns=["dt_max", "over2s"]).to_csv(os.path.join(OUT_DIR, f"mark_lead_stats_{s}.csv"), index=False, encoding="utf-8-sig")
    L, bv = md_stats(rows, ev, s)
    open(os.path.join(OUT_DIR, f"mark_lead_stats_{s}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    # 比較元（9/24）＝10/08 に build_v2 が書いた行単位の正本
    vb = pd.read_csv(os.path.join(OUT_DIR, f"G1_events_v2_{b}.csv")); vb["mark_note"] = vb.mark_note.fillna("")
    vb = add_valid_range(vb, b)
    eb = vb[vb.multi_increment == 0]
    rb = {r["technique"]: r for r in stats_rows(eb)}; rs = {r["technique"]: r for r in rows}
    cmp = []
    for c in ORDER + ["ALL"]:
        db = eb[(eb.mark_status == "mark") & ((eb.counter == c) | (c == "ALL"))].mark_dt_sec.dropna()
        ds = ev[(ev.mark_status == "mark") & ((ev.counter == c) | (c == "ALL"))].mark_dt_sec.dropna()
        U, p = mannwhitneyu(ds, db, alternative="two-sided") if len(db) and len(ds) else (np.nan, np.nan)
        KS, pks = ks_2samp(ds, db) if len(db) and len(ds) else (np.nan, np.nan)   # 補助＝形（裾）の違いも見る
        in_b = float(((db >= EXPECT[0]) & (db <= EXPECT[1])).mean()) if len(db) else np.nan
        in_s = float(((ds >= EXPECT[0]) & (ds <= EXPECT[1])).mean()) if len(ds) else np.nan
        rbis = 2 * U / (len(ds) * len(db)) - 1 if len(db) and len(ds) else np.nan   # 順位双列相関（正＝比較先が大きい側）
        # 観測区間の内側だけで同じ検定（感度）
        dbv = eb[(eb.mark_status == "mark") & (eb.in_valid_range == 1) & ((eb.counter == c) | (c == "ALL"))].mark_dt_sec.dropna()
        dsv = ev[(ev.mark_status == "mark") & (ev.in_valid_range == 1) & ((ev.counter == c) | (c == "ALL"))].mark_dt_sec.dropna()
        Uv, pv = mannwhitneyu(dsv, dbv, alternative="two-sided") if len(dbv) and len(dsv) else (np.nan, np.nan)
        cmp.append(dict(technique=c, technique_ja=JA[c], n_mark_base=len(db), n_mark_sess=len(ds),
                        rate_base=rb[c]["rate"], rate_sess=rs[c]["rate"],
                        med_base=rb[c]["dt_median"], med_sess=rs[c]["dt_median"], med_diff=round(rs[c]["dt_median"] - rb[c]["dt_median"], 3),
                        iqr_base=f"{rb[c]['dt_q1']}–{rb[c]['dt_q3']}", iqr_sess=f"{rs[c]['dt_q1']}–{rs[c]['dt_q3']}",
                        p90_base=rb[c]["dt_p90"], p90_sess=rs[c]["dt_p90"], over2s_base=rb[c]["over2s"], over2s_sess=rs[c]["over2s"],
                        max_base=rb[c]["dt_max"], max_sess=rs[c]["dt_max"],
                        U=float(U), p=float(p), r_rb=round(float(rbis), 3), KS=round(float(KS), 3), p_ks=float(pks),
                        p10_base=rb[c]["dt_p10"], p10_sess=rs[c]["dt_p10"], in_expect_base=round(in_b, 3), in_expect_sess=round(in_s, 3),
                        n_vr_base=len(dbv), n_vr_sess=len(dsv), med_vr_base=q(dbv, .5), med_vr_sess=q(dsv, .5), p_vr=float(pv)))
    ct = pd.DataFrame(cmp)
    tech = ct.technique != "ALL"
    ct.loc[tech, "p_holm"] = holm(ct.loc[tech, "p"]); ct.loc[~tech, "p_holm"] = ct.loc[~tech, "p"]
    ct.to_csv(os.path.join(OUT_DIR, f"mark_lead_compare_{b}_{s}.csv"), index=False, encoding="utf-8-sig")
    fmtp = lambda x: "nan" if pd.isna(x) else (f"{x:.3f}" if x >= 0.001 else f"{x:.1e}")
    M = [f"### 技法別の比較（{b} 対 {s}・対応した件だけ・mark_dt［秒］・Mann-Whitney U 両側）", "",
         "| 技法 | 対応 9/24 | 対応 10/14 | 中央値 9/24 | 中央値 10/14 | 差（10/14−9/24） | Q1–Q3 9/24 | Q1–Q3 10/14 | p90 9/24 | p90 10/14 | 2秒超 9/24 | 2秒超 10/14 | U | p | p（Holm） | 順位双列相関 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in ct.iterrows():
        M.append(f"| {r.technique_ja} | {r.n_mark_base} | {r.n_mark_sess} | {r.med_base} | {r.med_sess} | {r.med_diff:+.3f} | {r.iqr_base} | {r.iqr_sess} | {r.p90_base} | {r.p90_sess} | {r.over2s_base:.1%} | {r.over2s_sess:.1%} | {r.U:.0f} | {fmtp(r.p)} | {fmtp(r.p_holm)} | {r.r_rb:+.3f} |")
    M += ["", "### 補助＝分布の形（Kolmogorov-Smirnov 2標本・両側）と裾", "", "| 技法 | p10 9/24 | p10 10/14 | 0.2〜1.5s 内 9/24 | 0.2〜1.5s 内 10/14 | KS 統計量 | p（KS） |", "|---|---|---|---|---|---|---|"]
    for _, r in ct.iterrows():
        M.append(f"| {r.technique_ja} | {r.p10_base} | {r.p10_sess} | {r.in_expect_base:.1%} | {r.in_expect_sess:.1%} | {r.KS} | {fmtp(r.p_ks)} |")
    M += ["", "### 感度＝観測区間（song_valid_range）の内側だけ", "", "| 技法 | 対応 9/24 | 対応 10/14 | 中央値 9/24 | 中央値 10/14 | p |", "|---|---|---|---|---|---|"]
    for _, r in ct.iterrows():
        M.append(f"| {r.technique_ja} | {r.n_vr_base} | {r.n_vr_sess} | {r.med_vr_base} | {r.med_vr_sess} | {fmtp(r.p_vr)} |")
    M += ["", "### 前提の確認（10/14）", "", f"- G1 行数 {chk['g1_rows']}・G1_events_all との不一致 {chk['g1_all_mismatch']} 行",
          f"- marks 行数 {chk['marks_rows']}・G1 に対応しない marks 行 {chk['marks_unmatched']}・video_time_sec の不一致 {chk['time_mismatch']} 行",
          f"- 観測区間の外の G1 行（multi_increment=0）＝10/14 {int((ev.in_valid_range == 0).sum())}・9/24 {int((eb.in_valid_range == 0).sum())}"]
    open(os.path.join(OUT_DIR, f"mark_lead_compare_{b}_{s}.md"), "w", encoding="utf-8").write("\n".join(M) + "\n")
    # 図
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in [r"C:\Windows\Fonts\YuGothM.ttc", r"C:\Windows\Fonts\meiryo.ttc"]:
        if os.path.exists(fp): font_manager.fontManager.addfont(fp); plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name(); break
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    cols = {"b": "#4C72B0", "s": "#DD8452"}; pos = []
    for i, c in enumerate(ORDER + ["ALL"]):
        for j, (df, key) in enumerate([(eb, "b"), (ev, "s")]):
            d = df[(df.mark_status == "mark") & ((df.counter == c) | (c == "ALL"))].mark_dt_sec.dropna()
            bp = ax.boxplot([d], positions=[i * 3 + j], widths=0.7, patch_artist=True, showfliers=True,
                            flierprops=dict(marker=".", markersize=3, alpha=.5), medianprops=dict(color="black"))
            bp["boxes"][0].set_facecolor(cols[key]); bp["boxes"][0].set_alpha(.75)
            ax.text(i * 3 + j, -1.15, f"n={len(d)}", ha="center", fontsize=7)
    ax.set_xticks([i * 3 + .5 for i in range(len(ORDER) + 1)]); ax.set_xticklabels([JA[c] for c in ORDER + ["ALL"]])
    ax.axhspan(EXPECT[0], EXPECT[1], color="gray", alpha=.12, label="期待範囲 0.2〜1.5 秒")
    ax.axhline(0, color="gray", lw=.8)
    ax.set_ylim(-1.3, 6.0); ax.set_ylabel("カウンター増分 − 技法マーク出現［秒］（正＝マークが先）")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=cols["b"], alpha=.75, label=f"{b[4:6]}/{b[6:]} 収録"), Patch(facecolor=cols["s"], alpha=.75, label=f"{s[4:6]}/{s[6:]} 収録"),
                       Patch(facecolor="gray", alpha=.12, label="期待範囲 0.2〜1.5 秒")], loc="upper right", fontsize=8)
    ax.set_title("技法マークが出てからカウンターが増えるまでの時間（2収録の比較）")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"mark_lead_box_{b}_{s}.png")); plt.close(fig)
    print("\n".join(L[:12])); print(); print("\n".join(M))

if __name__ == "__main__":
    main()
