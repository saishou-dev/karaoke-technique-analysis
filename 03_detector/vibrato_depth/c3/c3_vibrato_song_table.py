# -*- coding: utf-8 -*-
"""C3 整合表：加点単位で測ったビブラート（9/24・正解208件）を曲単位に戻し、DAM の集計
（vibratoCount・vibratoTotalSecond・vibratoType・vibratoSkill）と比べる。
入力: vibrato_depth_events_win-2.5_-1.0.csv（主）・vibrato_depth_events.csv（窓[-1.5,+0.3]・副）
      dataset/G1_events_all.csv・dam_api/dam_ai-hearts_full_2026-10-06T04-05-48.csv
出力: c3_vibrato_song_table.csv / c3_vibrato_type_summary.csv / c3_vibrato_散布図.png / c3_stats.json
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
VD = HERE.parent
ROOT = Path("G:/研究アーカイブ/成果物")
DAM_CSV = ROOT/"dam_api/dam_ai-hearts_full_2026-10-06T04-05-48.csv"
G1_CSV = ROOT/"動画解析/dataset/G1_events_all.csv"
SESSION = "20260924"

# vibratoType コード→表示名。出典＝第三者実装 LlechiKaito/karaoke-api examples/detail-data.ts
# (https://github.com/LlechiKaito/karaoke-api, commit 0805fb1, 2026-02-04)。公式資料での確証は取れていない。
# コード 0 ↔ vibratoCount 0・vibratoTotalSecond 0 はこの CSV 全 200 行で一致（ノンビブ=合計1秒未満）。
TYPE_NAME = {0:"ノンビブ形(N)",1:"ボックス形(A-1)",2:"ボックス形(B-1)",3:"ボックス形(C-1)",
             4:"ボックス形(A-2)",5:"ボックス形(B-2)",6:"ボックス形(C-2)",7:"ボックス形(A-3)",
             8:"ボックス形(B-3)",9:"ボックス形(C-3)",10:"上昇形(D)",11:"下降形(E)",
             12:"縮小形(F)",13:"拡張形(G)",14:"ひし形(H)"}
# ボックス形の速さ階級（A=速い…C=遅い→1,2,3）と深さ階級（1=浅い…3=深い）
TYPE_SPEED = {1:1,2:2,3:3,4:1,5:2,6:3,7:1,8:2,9:3}
TYPE_DEPTH = {1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3}

# 較正 3-2 深さ比中央値（測定/真）：速さ依存の減衰。深さの補正に使う（線形補間・範囲外は端で固定）
CAL_RATE = np.array([4,5,6,7,8.]); CAL_RATIO = np.array([1.03,1.00,0.70,0.44,0.25])
def depth_ratio(rate):
    return np.interp(rate, CAL_RATE, CAL_RATIO)

def norm_name(s):
    s = str(s)
    s = s.replace("[生音]","").replace("(生音)","").replace("（生音）","")
    s = re.sub(r"\s+","",s)
    return s

# ---------- 読み込み ----------
dam = pd.read_csv(DAM_CSV, encoding="utf-8-sig")
dam = dam[dam.playDate.astype(str).str.startswith("2026-09-24")].copy()
dam["play_dt"] = pd.to_datetime(dam.playDate)
dam = dam.sort_values("play_dt").reset_index(drop=True)
assert len(dam)==28, len(dam)

g1 = pd.read_csv(G1_CSV)
g1 = g1[(g1.session.astype(str)==SESSION)]
g1v = g1[g1.counter=="vibrato"].groupby("video").size().rename("g1_vibrato_count")
g1multi = g1[(g1.counter=="vibrato")&(g1.multi_increment>0)].groupby("video").size().rename("g1_multi_increment")

def load_events(path):
    e = pd.read_csv(path, encoding="utf-8-sig")
    e = e[e.kind=="truth"].copy()
    e["rate_use"] = e.rate_chain_hz.where(e.rate_chain_hz.notna(), e.rate_hz)
    e["has_chain"] = e.duration_s>0
    e["dur_corr"] = np.where(e.has_chain, e.duration_s + 0.5/e.rate_use, 0.0)
    e["depth_corr"] = e.depth_cents/depth_ratio(e.rate_hz)
    return e
ev_main = load_events(VD/"vibrato_depth_events_win-2.5_-1.0.csv")   # 主：窓 [-2.5,-1.0]
ev_alt  = load_events(VD/"vibrato_depth_events.csv")                 # 副：窓 [-1.5,+0.3]
assert len(ev_main)==208 and len(ev_alt)==208

videos = sorted(ev_main.video.unique(), key=lambda v: v.rsplit("_",1)[1])
assert len(videos)==28
def _vs(v):
    t = v.rsplit("_",1)[1]
    return pd.to_datetime("2026-09-24 "+t[:2]+":"+t[2:4]+":"+t[4:6])
vstart = {v: _vs(v) for v in videos}

# ---------- DAM 行 ↔ 動画 の対応（時刻順・曲名一致で検算） ----------
rows=[]
for i,(dr,v) in enumerate(zip(dam.itertuples(), videos)):
    sn, vn = norm_name(dr.songName), norm_name(v.rsplit("_",2)[0])
    assert sn==vn, (sn, vn)
    assert vstart[v] < dr.play_dt, (v, dr.play_dt)
    rows.append((v, dr.songName, dr.playDate, (dr.play_dt-vstart[v]).total_seconds()))
match = pd.DataFrame(rows, columns=["video","dam_songName","dam_playDate","dam_end_minus_video_start_s"])

def song_agg(e, suffix):
    def agg(g):
        le6 = g[g.rate_hz<=6.0]
        ch = g[g.has_chain]
        return pd.Series({
            f"n_meas{suffix}": len(g),
            f"n_chain{suffix}": int(g.has_chain.sum()),
            f"dur_sum{suffix}": g.duration_s.sum(),
            f"dur_sum_corr{suffix}": g.dur_corr.sum(),
            f"dur_mean{suffix}": g.duration_s.mean(),
            f"dur_mean_corr{suffix}": g.dur_corr.mean(),
            f"depth_med{suffix}": g.depth_cents.median(),
            f"depth_corr_med{suffix}": g.depth_corr.median(),
            f"rate_med{suffix}": g.rate_hz.median(),
            f"rate_chain_med{suffix}": ch.rate_chain_hz.median() if len(ch) else np.nan,
            f"periodicity_med{suffix}": g.periodicity.median(),
            f"n_le6{suffix}": len(le6),
            f"depth_med_le6{suffix}": le6.depth_cents.median() if len(le6) else np.nan,
            f"rate_med_le6{suffix}": le6.rate_hz.median() if len(le6) else np.nan,
            f"dur_sum_le6{suffix}": le6.duration_s.sum(),
            f"periodicity_med_le6{suffix}": le6.periodicity.median() if len(le6) else np.nan,
        })
    return e.groupby("video").apply(agg)

A = song_agg(ev_main, "")
B = song_agg(ev_alt, "_w15")[["n_meas_w15","dur_sum_w15","dur_sum_corr_w15","depth_med_w15","rate_med_w15","periodicity_med_w15"]]

T = match.set_index("video").join(A).join(B).join(g1v).join(g1multi)
T["g1_multi_increment"] = T["g1_multi_increment"].fillna(0).astype(int)
damc = dam.set_index(pd.Index(match.video.tolist()))[["vibratoCount","vibratoTotalSecond","vibratoType","vibratoSkill",
        "longtoneSkill","longtoneCount","radarChartVibratoLongtone","radarChartPitch","totalScore"]]
damc.columns = ["dam_"+c for c in damc.columns]
T = T.join(damc)
T["dam_vibratoTypeName"] = T.dam_vibratoType.map(TYPE_NAME)
T["dam_type_speed_class"] = T.dam_vibratoType.map(TYPE_SPEED)
T["dam_type_depth_class"] = T.dam_vibratoType.map(TYPE_DEPTH)
T["dam_sec_per_count"] = T.dam_vibratoTotalSecond/T.dam_vibratoCount.replace(0,np.nan)
T["count_match_dam_g1"] = (T.dam_vibratoCount==T.g1_vibrato_count)
T["count_match_dam_meas"] = (T.dam_vibratoCount==T.n_meas)
T["sec_diff_dam_minus_meas"] = T.dam_vibratoTotalSecond - T.dur_sum
T["sec_diff_dam_minus_meas_corr"] = T.dam_vibratoTotalSecond - T.dur_sum_corr
T["sec_ratio_meas_over_dam"] = T.dur_sum/T.dam_vibratoTotalSecond
T["sec_ratio_meas_corr_over_dam"] = T.dur_sum_corr/T.dam_vibratoTotalSecond
T = T.reset_index().rename(columns={"index":"video"})
# 窓なしの持続（c3_fullspan.py の出力）：加点ごとの連鎖全長と、曲全体の連鎖合計
FS = pd.read_csv(HERE/"c3_fullspan_songs.csv", encoding="utf-8-sig").set_index("video")[
    ["full_sum","full_sum_corr","full_mean_corr","n_none","song_chain_sec_all","song_chain_sec_ge0.6","song_chain_sec_ge1.0","n_chains_all","n_chains_ge0.6","n_chains_ge1.0"]]
T = T.set_index("video").join(FS).reset_index()
T["sec_diff_dam_minus_full_corr"] = T.dam_vibratoTotalSecond - T.full_sum_corr
T["sec_ratio_full_corr_over_dam"] = T.full_sum_corr/T.dam_vibratoTotalSecond
T.insert(1,"song",T.video.str.rsplit("_",n=2).str[0])
T = T.sort_values("video", key=lambda s: s.str.rsplit("_",n=1).str[1]).reset_index(drop=True)
T.round(4).to_csv(HERE/"c3_vibrato_song_table.csv", index=False, encoding="utf-8-sig")

# ---------- 統計 ----------
S = {}
S["count"] = {
  "n_songs": int(len(T)),
  "dam_eq_g1": int(T.count_match_dam_g1.sum()),
  "dam_eq_meas": int(T.count_match_dam_meas.sum()),
  "sum_dam": int(T.dam_vibratoCount.sum()), "sum_g1": int(T.g1_vibrato_count.sum()), "sum_meas": int(T.n_meas.sum()),
  "mismatch_dam_meas": T.loc[~T.count_match_dam_meas, ["video","dam_vibratoCount","g1_vibrato_count","n_meas"]].to_dict("records"),
}
def corr(x,y):
    m = np.isfinite(x)&np.isfinite(y)
    x,y = np.asarray(x,dtype=float)[m], np.asarray(y,dtype=float)[m]
    if len(x)<3 or np.std(x)==0 or np.std(y)==0: return {"n":int(len(x)),"pearson_r":None,"spearman_rho":None,"spearman_p":None}
    pr = stats.pearsonr(x,y); sp = stats.spearmanr(x,y)
    return {"n":int(len(x)),"pearson_r":round(float(pr[0]),3),"pearson_p":round(float(pr[1]),4),
            "spearman_rho":round(float(sp[0]),3),"spearman_p":round(float(sp[1]),4)}
sec = {}
for lab,col in [("raw_win-2.5_-1.0","dur_sum"),("corr_win-2.5_-1.0","dur_sum_corr"),
                ("raw_win-1.5_+0.3","dur_sum_w15"),("corr_win-1.5_+0.3","dur_sum_corr_w15"),
                ("raw_le6Hz_win-2.5_-1.0","dur_sum_le6"),
                ("fullspan_raw","full_sum"),("fullspan_corr","full_sum_corr"),
                ("songwide_all_chains","song_chain_sec_all"),("songwide_chains_ge0.6s","song_chain_sec_ge0.6"),("songwide_chains_ge1.0s","song_chain_sec_ge1.0")]:
    d = T.dam_vibratoTotalSecond - T[col]
    sec[lab] = dict(corr(T.dam_vibratoTotalSecond, T[col]),
                    diff_median=round(float(d.median()),2), diff_q1=round(float(d.quantile(.25)),2), diff_q3=round(float(d.quantile(.75)),2),
                    ratio_median=round(float((T[col]/T.dam_vibratoTotalSecond).median()),3),
                    sum_dam=round(float(T.dam_vibratoTotalSecond.sum()),1), sum_meas=round(float(T[col].sum()),1))
# 1 回あたり秒
sec["per_event_raw"] = dict(corr(T.dam_sec_per_count, T.dur_mean), dam_median=round(float(T.dam_sec_per_count.median()),2), meas_median=round(float(T.dur_mean.median()),2))
sec["per_event_corr"] = dict(corr(T.dam_sec_per_count, T.dur_mean_corr), meas_median=round(float(T.dur_mean_corr.median()),2))
sec["per_event_fullspan_corr"] = dict(corr(T.dam_sec_per_count, T.full_mean_corr), meas_median=round(float(T.full_mean_corr.median()),2))
# 連鎖ありだけの件数と DAM 回数
sec["chain_count_vs_dam_count"] = corr(T.dam_vibratoCount, T.n_chain)
# 回数の影響を除いた偏相関（DAM 合計秒・測定合計の両方から vibratoCount への回帰残差を取り Spearman）
def resid(v, c):
    A = np.c_[np.ones(len(c)), c]; b = np.linalg.lstsq(A, v, rcond=None)[0]; return v - A@b
cnt = T.dam_vibratoCount.values.astype(float); r_dam = resid(T.dam_vibratoTotalSecond.values.astype(float), cnt)
sec["partial_given_count"] = {}
for col in ["dur_sum_corr","full_sum_corr","song_chain_sec_all","song_chain_sec_ge0.6","song_chain_sec_ge1.0"]:
    sec["partial_given_count"][col] = corr(r_dam, resid(T[col].values.astype(float), cnt))
sec["dam_total_vs_dam_count"] = corr(T.dam_vibratoCount, T.dam_vibratoTotalSecond)
sec["songwide_count_vs_dam_count"] = {c: corr(T.dam_vibratoCount, T[c]) for c in ["n_chains_all","n_chains_ge0.6","n_chains_ge1.0"]}
S["total_second"] = sec

# タイプ別
ev_main_t = ev_main.merge(T[["video","dam_vibratoType","dam_vibratoTypeName","dam_vibratoSkill"]], on="video")
def type_summary(df_song, df_ev):
    rows=[]
    for code, g in df_song.groupby("dam_vibratoType"):
        ge = df_ev[df_ev.dam_vibratoType==code]; ge6 = ge[ge.rate_hz<=6]
        rows.append({"vibratoType":code,"name":TYPE_NAME.get(code),"speed_class":TYPE_SPEED.get(code),"depth_class":TYPE_DEPTH.get(code),
            "n_songs":len(g),"n_events":len(ge),"n_events_le6":len(ge6),
            "song_depth_med_median":g.depth_med.median(),"song_rate_med_median":g.rate_med.median(),
            "song_dur_sum_median":g.dur_sum.median(),"song_periodicity_med_median":g.periodicity_med.median(),
            "event_depth_median":ge.depth_cents.median(),"event_depth_q1":ge.depth_cents.quantile(.25),"event_depth_q3":ge.depth_cents.quantile(.75),
            "event_rate_median":ge.rate_hz.median(),"event_rate_q1":ge.rate_hz.quantile(.25),"event_rate_q3":ge.rate_hz.quantile(.75),
            "event_duration_median":ge.duration_s.median(),"event_periodicity_median":ge.periodicity.median(),
            "event_depth_median_le6":ge6.depth_cents.median() if len(ge6) else np.nan,
            "event_rate_median_le6":ge6.rate_hz.median() if len(ge6) else np.nan,
            "event_depth_corr_median":ge.depth_corr.median(),
            "dam_skill_median":g.dam_vibratoSkill.median(),"dam_sec_per_count_median":g.dam_sec_per_count.median(),
            "songs":"; ".join(g.song.tolist())})
    return pd.DataFrame(rows)
TS = type_summary(T, ev_main_t)
TS.round(3).to_csv(HERE/"c3_vibrato_type_summary.csv", index=False, encoding="utf-8-sig")

typ = {}
box = T[T.dam_type_depth_class.notna()]
typ["n_box_songs"] = int(len(box)); typ["box_songs"] = box[["song","dam_vibratoTypeName","depth_med","rate_med","n_meas"]].round(2).to_dict("records")
typ["spearman_depthclass_vs_depth_med"] = corr(box.dam_type_depth_class, box.depth_med) if len(box)>=3 else {"n":int(len(box)),"note":"検定不能"}
typ["spearman_speedclass_vs_rate_med"] = corr(box.dam_type_speed_class, box.rate_med) if len(box)>=3 else {"n":int(len(box)),"note":"検定不能"}
# 上昇形(10) vs それ以外（曲単位・Mann-Whitney）
a = T[T.dam_vibratoType==10]; b = T[T.dam_vibratoType!=10]
mw = {}
for col in ["depth_med","rate_med","dur_sum","dur_mean","periodicity_med","dam_vibratoSkill"]:
    u = stats.mannwhitneyu(a[col].dropna(), b[col].dropna(), alternative="two-sided")
    mw[col] = {"n10":int(a[col].notna().sum()),"n_other":int(b[col].notna().sum()),"median10":round(float(a[col].median()),3),"median_other":round(float(b[col].median()),3),"U":float(u.statistic),"p":round(float(u.pvalue),4)}
typ["type10_vs_other_mannwhitney_songlevel"] = mw
# イベント単位 Kruskal（タイプ間で深さ・速さに差があるか）
grp = [g.depth_cents.values for _,g in ev_main_t.groupby("dam_vibratoType") if len(g)>=1]
kr = stats.kruskal(*grp)
typ["kruskal_depth_by_type_eventlevel"] = {"k":len(grp),"H":round(float(kr.statistic),3),"p":round(float(kr.pvalue),4)}
grp = [g.rate_hz.dropna().values for _,g in ev_main_t.groupby("dam_vibratoType")]
kr = stats.kruskal(*grp)
typ["kruskal_rate_by_type_eventlevel"] = {"k":len(grp),"H":round(float(kr.statistic),3),"p":round(float(kr.pvalue),4)}
S["type"] = typ

# Skill との相関（曲単位）
sk = {}
for col in ["depth_med","depth_corr_med","depth_med_le6","rate_med","rate_med_le6","dur_sum","dur_sum_corr","dur_mean","dur_mean_corr","periodicity_med","periodicity_med_le6","n_meas","n_chain",
            "dam_vibratoCount","dam_vibratoTotalSecond","dam_sec_per_count","depth_med_w15","dur_sum_w15","periodicity_med_w15",
            "full_sum_corr","full_mean_corr","song_chain_sec_all","song_chain_sec_ge0.6"]:
    sk[col] = corr(T.dam_vibratoSkill, T[col])
S["skill_spearman"] = sk
rd = {}
for col in ["depth_med","dur_sum","periodicity_med","dam_vibratoSkill","dam_vibratoTotalSecond","dam_longtoneSkill"]:
    rd[col] = corr(T.dam_radarChartVibratoLongtone, T[col])
S["radar_vibratolongtone_spearman"] = rd
S["skill_distribution"] = {int(k):int(v) for k,v in T.dam_vibratoSkill.value_counts().sort_index().items()}
S["event_overall"] = {"n":208,"depth_med":float(ev_main.depth_cents.median()),"rate_med":float(ev_main.rate_hz.median()),"dur_med":float(ev_main.duration_s.median()),
                      "chain_frac":round(float(ev_main.has_chain.mean()),3),"le6_frac":round(float((ev_main.rate_hz<=6).mean()),3)}
json.dump(S, open(HERE/"c3_stats.json","w",encoding="utf-8"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o,"item") else str(o))

# ---------- 図 ----------
for f in ["Yu Gothic","Meiryo","MS Gothic","Noto Sans CJK JP"]:
    if any(x.name==f for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"]=f; break
plt.rcParams["axes.unicode_minus"]=False
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; AXIS="#c3c2b7"
C1="#2a78d6"; C2="#eb6834"
fig, axes = plt.subplots(1,2, figsize=(13,5.6), facecolor=SURF, gridspec_kw={"width_ratios":[1.05,1]})
ax=axes[0]; ax.set_facecolor(SURF)
for _,r in T.iterrows():
    ax.plot([r.dam_vibratoTotalSecond]*2,[r.dur_sum_corr,r.full_sum_corr], color=MUTED, lw=0.8, zorder=2)
ax.scatter(T.dam_vibratoTotalSecond, T.dur_sum_corr, s=46, color=C1, edgecolor=SURF, lw=1.5, zorder=3, label="窓 [-2.5,-1.0] s 内の連鎖（＋半周期）")
ax.scatter(T.dam_vibratoTotalSecond, T.full_sum_corr, s=46, color=C2, marker="D", edgecolor=SURF, lw=1.5, zorder=3, label="窓なし＝連鎖の全長（＋半周期）")
r_cor=S["total_second"]["corr_win-2.5_-1.0"]; r_full=S["total_second"]["fullspan_corr"]
p_full=S["total_second"]["partial_given_count"]["full_sum_corr"]
ax.set_xlabel("DAM vibratoTotalSecond [s]（曲の合計）", color=INK2); ax.set_ylabel("測定の持続の合計 [s]（正解イベントの和）", color=INK2)
ax.set_title(f"A. 合計秒：DAM と測定（28曲）  ρ={r_cor['spearman_rho']}（窓内）/{r_full['spearman_rho']}（全長）\n   回数を除いた偏相関 ρ={p_full['spearman_rho']}（p={p_full['spearman_p']}）", color=INK, fontsize=10.5, loc="left")
ax.set_xlim(0, T.dam_vibratoTotalSecond.max()*1.08); ax.set_ylim(0, T.full_sum_corr.max()*1.12); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED); ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.text(0.98, 0.03, f"測定/DAM の比 中央値 {r_full['ratio_median']}（全長）", transform=ax.transAxes, ha="right", color=MUTED, fontsize=9)

ax=axes[1]; ax.set_facecolor(SURF)
order = sorted(ev_main_t.dam_vibratoType.unique())
data = [ev_main_t[ev_main_t.dam_vibratoType==c].depth_cents.values for c in order]
SHORT = {0:"N",1:"A-1",2:"B-1",3:"C-1",4:"A-2",5:"B-2",6:"C-2",7:"A-3",8:"B-3",9:"C-3",10:"上昇形 D",11:"下降形 E",12:"縮小形 F",13:"拡張形 G",14:"ひし形 H"}
labels = [f"{SHORT[c]}\nコード{c}\n{len(d)}件/{int((T.dam_vibratoType==c).sum())}曲" for c,d in zip(order,data)]
bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.6),
                boxprops=dict(facecolor="#cde2fb", edgecolor=C1, lw=1.2), whiskerprops=dict(color=C1), capprops=dict(color=C1))
rng = np.random.default_rng(0)
for i,d in enumerate(data):
    ax.scatter(i+1+rng.uniform(-0.16,0.16,len(d)), d, s=14, color=C1, alpha=0.55, edgecolor="none", zorder=3)
ax.set_xticks(range(1,len(order)+1)); ax.set_xticklabels(labels, fontsize=8.5, color=INK2)
ax.set_ylabel("深さ [cent]（正解イベント・1件1点）", color=INK2)
kr=S["type"]["kruskal_depth_by_type_eventlevel"]
ax.set_title(f"B. タイプ別の深さ（正解イベント・1件1点）  Kruskal H={kr['H']}  p={kr['p']}\n   タイプ名は第三者実装の対応表（公式未確証）", color=INK, fontsize=10.5, loc="left")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED)
fig.suptitle("C3 整合表：加点単位で測ったビブラートを曲単位に戻して DAM の集計と比べる（9/24・28曲・正解208件）", color=INK, fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(HERE/"c3_vibrato_散布図.png", dpi=150, facecolor=SURF)
print(json.dumps(S, ensure_ascii=False, indent=1, default=str))
print(T[["song","dam_vibratoCount","g1_vibrato_count","n_meas","dam_vibratoTotalSecond","dur_sum","dur_sum_corr","dam_vibratoType","dam_vibratoSkill","depth_med","rate_med","periodicity_med","n_le6","depth_med_le6"]].round(2).to_string())
