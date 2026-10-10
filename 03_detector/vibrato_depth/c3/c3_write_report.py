# -*- coding: utf-8 -*-
"""報告_C3整合_ビブラート_2026-10-08.md を c3_stats.json / CSV から組み立てる（数値の転記ミスを避ける）"""
import json
from pathlib import Path
import pandas as pd, numpy as np
HERE = Path(__file__).resolve().parent
S = json.load(open(HERE/"c3_stats.json", encoding="utf-8"))
T = pd.read_csv(HERE/"c3_vibrato_song_table.csv", encoding="utf-8-sig")
TS = pd.read_csv(HERE/"c3_vibrato_type_summary.csv", encoding="utf-8-sig")
E = pd.read_csv(HERE/"c3_fullspan_events.csv", encoding="utf-8-sig")

def f(x, d=2):
    if x is None or (isinstance(x,float) and not np.isfinite(x)): return "—"
    return f"{x:.{d}f}"
def c(k, d=None):
    v = d if d is not None else S["skill_spearman"][k]
    return f"ρ={f(v['spearman_rho'],3)}（p={f(v['spearman_p'],3)}）・r={f(v['pearson_r'],3)}"

ts = S["total_second"]; cnt = S["count"]
L = []
L.append("# C3 整合表：加点単位で測ったビブラートを曲単位に戻して DAM の集計と比べる（2026-10-08）")
L.append("")
L.append("主張 C3「加点単位で測った中身を曲単位に戻すと DAM の集計と一致する」の最初の証拠表。対象＝9/24（28 曲・正解ビブラート 208 件）。")
L.append("DAM 側＝精密採点 Ai Hearts の API 出力（`dam_api/dam_ai-hearts_full_2026-10-06T04-05-48.csv`・列 vibratoCount／vibratoTotalSecond／vibratoType／vibratoSkill／radarChartVibratoLongtone）。")
L.append("測定側＝`vibrato_depth_20261008/vibrato_depth_events_win-2.5_-1.0.csv`（正解 208 件・深さ cent・速さ Hz・持続 s・周期性）を主に使い、窓 [−1.5,+0.3] 版と、本件で追加した窓なし版（`c3_fullspan.py`）を併記。")
L.append("")
L.append("| ファイル | 中身 |")
L.append("|---|---|")
L.append("| c3_vibrato_song_table.csv | 曲別の全列（DAM の集計・G1 の回数・測定の曲単位集計・差と比・≤6 Hz 限定・窓なし版） |")
L.append("| c3_vibrato_type_summary.csv | vibratoType 別の中央値（曲単位・イベント単位） |")
L.append("| c3_vibrato_散布図.png | A＝合計秒の散布図（DAM vs 測定）・B＝タイプ別の深さ箱ひげ |")
L.append("| c3_fullspan_events.csv / c3_fullspan_songs.csv | 窓なしの持続（加点ごとの連鎖全長・曲全体の連鎖合計） |")
L.append("| c3_stats.json | 相関・検定の全数値 |")
L.append("| c3_vibrato_song_table.py / c3_fullspan.py / c3_write_report.py | 再現用（`python c3_fullspan.py` → `python c3_vibrato_song_table.py` → `python c3_write_report.py`） |")
L.append("")
L.append("## 1. 方法")
L.append("")
L.append("**曲の対応**：DAM の playDate（採点時刻＝曲の終わり）と動画名の開始時刻を時刻順に 1 対 1 で対応させ、曲名の一致（[生音] の表記差を除く）と「動画開始 < playDate」を全 28 曲で検算した（playDate − 動画開始＝" + f"{T.dam_end_minus_video_start_s.min():.0f}〜{T.dam_end_minus_video_start_s.max():.0f} s" + "）。")
L.append("")
L.append("**曲単位への戻し方**：正解イベント（G1 台帳の vibrato・採点区間内）ごとの測定値を曲で集計。回数＝件数、合計秒＝持続の和、深さ・速さ・周期性＝曲内の中央値。持続は定義上（極値から極値）半周期短いので「＋0.5／連鎖の速さ」を足した補正後も併記。深さは較正 3-2 の速さ別の比（4 Hz 1.03・5 Hz 1.00・6 Hz 0.70・7 Hz 0.44・8 Hz 0.25・線形補間）で割った補正値も併記。")
L.append("")
L.append("**速さ帯の統制**：6 Hz 超は深さが浅く出る系統誤差があるので、速さ ≤ 6 Hz のイベントだけで集計した列（`*_le6`）を併記（208 件中 " + f"{S['event_overall']['le6_frac']*100:.1f}% が ≤ 6 Hz" + "）。")
L.append("")
L.append("**窓なしの持続（追加）**：窓 [−2.5,−1.0] s は持続を最長 1.5 s で切ってしまうため、(a) 加点ごとに窓 [−6.0,+0.5] s で連鎖を全部拾い主窓と重なる連鎖の全長、(b) 曲全体（採点区間）の全有声区間から連鎖を全部拾った合計秒（加点に紐づけない・長さ閾値 0.3／0.6／1.0 s）も測った。連鎖の定義は vibrato_depth.py と同一。")
L.append("")
L.append("**検定**：Spearman ρ（曲単位・n=28）。タイプ間はイベント単位の Kruskal-Wallis。上昇形(D) vs その他は曲単位の Mann-Whitney U。合計秒は回数の影響を除くため、DAM 合計秒と測定合計の両方から vibratoCount への回帰残差を取った偏相関も出した。")
L.append("")
L.append("## 2. vibratoType のコード対応（出典と確証の程度）")
L.append("")
L.append("API の vibratoType は 0〜13 の整数。公式ページ（clubdam.com）にはコード表が無く、確証が取れた公式資料は無い。採用した対応表は第三者実装 **LlechiKaito/karaoke-api**（examples/detail-data.ts・`getVibratoTypeName`・commit 0805fb1・2026-02-04）＝ https://github.com/LlechiKaito/karaoke-api ：")
L.append("")
L.append("| コード | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
L.append("| 名称 | ノンビブ形(N) | A-1 | B-1 | C-1 | A-2 | B-2 | C-2 | A-3 | B-3 | C-3 | 上昇形(D) | 下降形(E) | 縮小形(F) | 拡張形(G) | ひし形(H) |")
L.append("")
L.append("タイプの意味（複数の解説サイトで一致）：ボックス形の英字＝波の速さ（A 速い・B 標準・C 遅い）、数字＝波の深さ（1 浅い・2 中・3 深い）。D〜H は音程や振幅が時間変化する形。N＝ビブラート合計 1 秒未満。出典：stepupkaraoke.com（https://stepupkaraoke.com/vibratolongtone/what-vibratone-type ）・meltan5.hateblo.jp（精密採点Ai攻略③VL編）・clubdam.com の DX 向け説明（https://www.clubdam.com/membership/import/marking/vibrato_description.html ・N/A/B/C/D/E/F/G/H の 9 区分の説明）。")
L.append("")
L.append("**確証の程度**：(1) コード表そのものは第三者実装の 1 件のみ＝**未確証**。(2) 内部整合として、CSV 全 200 行でコード 0 の 2 行だけが vibratoCount=0・vibratoTotalSecond=0（N＝ノンビブの定義と一致）。(3) 9/24 の録画は曲の終わりで止めており結果画面（タイプ名の表示）が映っていないため、画面での突き合わせは不可（Brand New／CHEERS／うっせぇわ の末尾フレームで確認）。(4) DAM★とも Web はログインが要るため本件では当たっていない＝**次回のセッションで結果画面を録画に含めれば確証できる**。以下、タイプ名は「第三者実装準拠・未確証」の但し書き付きで使う。")
L.append("")
L.append("## 3. 曲単位の整合表（9/24・28 曲）")
L.append("")
L.append("| 曲（開始時刻） | DAM 回数 | G1 回数 | 測定件数 | DAM 合計秒 | 測定合計 窓内＋半周期 | 測定合計 全長＋半周期 | 曲全体の連鎖合計 | タイプ | 上手さ | レーダー VL | 深さ中央値 | 深さ ≤6Hz | 速さ中央値 | 周期性 |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in T.itertuples():
    hhmm = r.video.rsplit("_",1)[1]; hhmm = hhmm[:2]+":"+hhmm[2:4]
    L.append(f"| {r.song}（{hhmm}） | {r.dam_vibratoCount} | {r.g1_vibrato_count} | {int(r.n_meas)} | {r.dam_vibratoTotalSecond} | {f(r.dur_sum_corr)} | {f(r.full_sum_corr)} | {f(r.song_chain_sec_all,1)} | {r.dam_vibratoType} {r.dam_vibratoTypeName} | {r.dam_vibratoSkill} | {r.dam_radarChartVibratoLongtone} | {f(r.depth_med,1)} | {f(r.depth_med_le6,1)} | {f(r.rate_med)} | {f(r.periodicity_med)} |")
L.append("")
L.append("### 3-1. 回数（verify の再確認）")
L.append("")
mm = cnt["mismatch_dam_meas"]
L.append(f"DAM vibratoCount と G1 台帳の行数が一致＝**{cnt['dam_eq_g1']}／28 曲**（合計 DAM {cnt['sum_dam']}・G1 {cnt['sum_g1']}・測定に載った正解 {cnt['sum_meas']}）。不一致 4 曲の内訳：")
L.append("")
L.append("| 曲 | DAM | G1 行数 | G1 の画面カウント終値 | 測定件数 | 原因 |")
L.append("|---|---|---|---|---|---|")
L.append("| ホワイトノイズ（11:18） | 14 | 13 | 14 | 12 | 2 増えの読み 1 件（multi_increment）＝台帳の行数は 1 少ないが終値は一致。測定はさらに 1 件が採点区間外 |")
L.append("| サマータイムシンデレラ（12:45） | 14 | 11 | 14 | 8 | 2 増えの読み 3 件＝終値は一致。測定は 3 件が採点区間外 |")
L.append("| 花占い（12:49） | 8 | 7 | 8 | 6 | 2 増えの読み 1 件＝終値は一致。測定は 1 件が採点区間外 |")
L.append("| ケセラセラ（10:43） | 12 | 8 | 8 | 8 | **台帳の欠落**。録画の末尾フレームで画面のビブラート数が 230 s＝8 → 258 s＝9 → 262 s＝10 → 272 s＝11 → 275 s＝12 と増えるのを確認（アクセントも 13→15）。台帳は 221.9 s で止まり、index.json の採点区間も 259.2 s で終わっている＝曲の最後の約 45 s（アウトロ）の加点が G1 に入っていない |")
L.append("")
L.append(f"画面カウントの終値（2 増えを足した値）で見ると DAM と一致＝**27／28 曲**。残る 1 曲（ケセラセラ）は G1 台帳と採点区間の修正対象（本件では触っていない・他 Agent が台帳を使用中のため）。")
L.append("")
L.append("### 3-2. 合計秒")
L.append("")
L.append("| 測定の合計 | 定義 | 測定の総和 [s]（DAM は 749） | 比 測定/DAM 中央値 | 差 DAM−測定 中央値 [Q1, Q3] | Pearson r | Spearman ρ（p） | 回数を除いた偏相関 ρ（p） |")
L.append("|---|---|---|---|---|---|---|---|")
rows = [("窓 [−2.5,−1.0] そのまま","raw_win-2.5_-1.0",None),("窓 [−2.5,−1.0] ＋半周期","corr_win-2.5_-1.0","dur_sum_corr"),
        ("窓 [−1.5,+0.3] ＋半周期","corr_win-1.5_+0.3",None),("窓 [−2.5,−1.0] 速さ≤6Hz のみ","raw_le6Hz_win-2.5_-1.0",None),
        ("窓なし＝連鎖の全長＋半周期（加点ごと）","fullspan_corr","full_sum_corr"),
        ("曲全体の連鎖 全部（加点に紐づけない）","songwide_all_chains","song_chain_sec_all"),("曲全体の連鎖 ≥0.6 s","songwide_chains_ge0.6s","song_chain_sec_ge0.6"),("曲全体の連鎖 ≥1.0 s","songwide_chains_ge1.0s","song_chain_sec_ge1.0")]
for name,k,pk in rows:
    v = ts[k]; p = ts["partial_given_count"].get(pk) if pk else None
    L.append(f"| {name} | {k} | {v['sum_meas']} | {f(v['ratio_median'],3)} | {f(v['diff_median'],1)} [{f(v['diff_q1'],1)}, {f(v['diff_q3'],1)}] | {f(v['pearson_r'],3)} | {f(v['spearman_rho'],3)}（{f(v['spearman_p'],4)}） | " + (f"{f(p['spearman_rho'],3)}（{f(p['spearman_p'],3)}）" if p else "—") + " |")
L.append("")
pe = ts["per_event_corr"]; pf = ts["per_event_fullspan_corr"]; dd = ts["dam_total_vs_dam_count"]
L.append(f"1 回あたりの秒数：DAM（合計秒／回数）の中央値 **{f(ts['per_event_raw']['dam_median'])} s**、測定（窓内＋半周期）{f(pe['meas_median'])} s・全長＋半周期 {f(pf['meas_median'])} s。曲ごとの対応は 窓内 {c(None,pe)}、全長 {c(None,pf)}＝**無相関**。")
L.append(f"DAM 内部では合計秒と回数が {c(None,dd)} と強く連動する。")
L.append("")
L.append("読み：合計秒の相関（ρ≈0.83〜0.86）は**回数の相関が運んでいる**。回数を除くと偏相関は 0 付近（−0.17〜+0.09・全て p>0.38）で、1 回あたりの長さ（DAM 3.3 s vs 測定 0.4〜0.6 s）は曲ごとに対応しない。測定の「持続」は周期的な揺れの連鎖（山谷差 ≥15 cent が続く区間）だけを数えるので、DAM が「ビブラート 1 回」として数えている 3 秒前後の区間のうち連鎖として残るのは 2 割弱＝**DAM の合計秒を再現するには持続の定義を変える必要がある**（7 節）。曲全体の連鎖を全部足すと総和は DAM の 1.8 倍（連鎖の多くはビブラートではない揺れ＝FP 側）、≥0.6 s に絞ると 0.67 倍で、どの閾値でも偏相関は付かない。")
L.append("")
L.append("### 3-3. タイプ（振幅・周期の分類）と測定の中央値")
L.append("")
L.append("| コード | 名称（未確証） | 速さ階級 | 深さ階級 | 曲数 | 件数（≤6Hz） | 深さ 中央値 [Q1,Q3] cent | 深さ ≤6Hz | 深さ 速さ補正後 | 速さ 中央値 [Q1,Q3] Hz | 持続 中央値 s | 周期性 中央値 | DAM 上手さ 中央値 | DAM 秒/回 中央値 | 曲 |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in TS.itertuples():
    L.append(f"| {r.vibratoType} | {r.name} | {f(r.speed_class,0)} | {f(r.depth_class,0)} | {r.n_songs} | {r.n_events}（{r.n_events_le6}） | {f(r.event_depth_median,0)} [{f(r.event_depth_q1,0)}, {f(r.event_depth_q3,0)}] | {f(r.event_depth_median_le6,0)} | {f(r.event_depth_corr_median,0)} | {f(r.event_rate_median)} [{f(r.event_rate_q1)}, {f(r.event_rate_q3)}] | {f(r.event_duration_median)} | {f(r.event_periodicity_median)} | {f(r.dam_skill_median,1)} | {f(r.dam_sec_per_count_median)} | {r.songs} |")
L.append("")
ty = S["type"]; kd = ty["kruskal_depth_by_type_eventlevel"]; kr = ty["kruskal_rate_by_type_eventlevel"]
sd = ty["spearman_depthclass_vs_depth_med"]; ss = ty["spearman_speedclass_vs_rate_med"]
L.append(f"- 9/24 の 28 曲は **24 曲が同じコード 10（上昇形 D）** で、ボックス形は 3 曲（A-1・B-2・C-2 が各 1 曲）、縮小形 F が 1 曲。深さ階級（1/2/3）と速さ階級（A/B/C）の単調性を Spearman で検定するには曲が足りない（ボックス形 n=3：深さ階級 vs 深さ中央値 ρ={f(sd.get('spearman_rho'),3)} p={f(sd.get('spearman_p'),3)}・速さ階級 vs 速さ中央値 ρ={f(ss.get('spearman_rho'),3)} p={f(ss.get('spearman_p'),3)}＝**検定不能・参考値**）。向きだけ見ると A-1（浅い）35 cent < B-2／C-2（中）80〜110 cent で階級と矛盾しない。速さは A（速い）の恋風が 3.0 Hz・C（遅い）のいつかが 4.7 Hz で**逆**だが、恋風は 1 件・連鎖なし（周期性 0.46）で速さの推定自体が信用できない件。")
L.append(f"- タイプ間の差（イベント単位・Kruskal-Wallis・5 群）：深さ H={kd['H']} **p={kd['p']}**（有意）、速さ H={kr['H']} p={kr['p']}（差なし）。深さの差は縮小形 F（CHEERS・130 cent）と B-2（うっせぇわ・110 cent）が上昇形 D（60 cent）より深いことによる。")
mw = ty["type10_vs_other_mannwhitney_songlevel"]
L.append(f"- 上昇形 D（24 曲）vs その他（4 曲）・曲単位 Mann-Whitney：深さ中央値 {f(mw['depth_med']['median10'],0)} vs {f(mw['depth_med']['median_other'],0)} cent（p={f(mw['depth_med']['p'],3)}）・速さ {f(mw['rate_med']['median10'])} vs {f(mw['rate_med']['median_other'])} Hz（p={f(mw['rate_med']['p'],3)}）・持続の和 {f(mw['dur_sum']['median10'])} vs {f(mw['dur_sum']['median_other'])} s（p={f(mw['dur_sum']['p'],3)}・回数の差）・上手さ {f(mw['dam_vibratoSkill']['median10'],1)} vs {f(mw['dam_vibratoSkill']['median_other'],1)}（p={f(mw['dam_vibratoSkill']['p'],3)}）。")
L.append("- 「上昇形 D」が 24／28 曲という偏りは、この CSV 全 200 曲でもコード 10 が 138 曲（69%）で同じ。コード対応が正しければ本人のビブラートは「揺れながら音程が上がる」と分類されている曲が大半＝**次の測定項目＝連鎖区間の土台（移動中央値）の傾き**で検証できる（7 節）。")
L.append("")
L.append("### 3-4. 上手さ（vibratoSkill 0〜9）との相関（曲単位・n=28・Spearman）")
L.append("")
L.append(f"上手さの分布：{S['skill_distribution']}（0 が 6 曲・9 が 1 曲）。")
L.append("")
L.append("| 測定（曲単位） | ρ | p | Pearson r |")
L.append("|---|---|---|---|")
for k,name in [("depth_med","深さ 中央値"),("depth_corr_med","深さ 速さ補正後 中央値"),("depth_med_le6","深さ 中央値（≤6 Hz）"),("rate_med","速さ 中央値"),("rate_med_le6","速さ 中央値（≤6 Hz）"),
               ("dur_sum_corr","持続の和（窓内＋半周期）"),("full_sum_corr","持続の和（全長＋半周期）"),("dur_mean_corr","持続の平均（窓内＋半周期）"),("full_mean_corr","持続の平均（全長＋半周期）"),
               ("periodicity_med","周期性 中央値"),("periodicity_med_le6","周期性 中央値（≤6 Hz）"),("n_meas","件数"),("n_chain","連鎖ありの件数"),("depth_med_w15","深さ 中央値（窓 [−1.5,+0.3]）"),
               ("dam_vibratoCount","（DAM 内）回数"),("dam_vibratoTotalSecond","（DAM 内）合計秒"),("dam_sec_per_count","（DAM 内）秒/回")]:
    v = S["skill_spearman"][k]
    L.append(f"| {name} | {f(v['spearman_rho'],3)} | {f(v['spearman_p'],4)} | {f(v['pearson_r'],3)} |")
L.append("")
rd = S["radar_vibratolongtone_spearman"]
L.append(f"読み：上手さは DAM 内部の合計秒（ρ=0.54・p=0.003）・秒/回（ρ=0.47・p=0.011）と連動する一方、私の測定では持続の和（ρ=0.38・p=0.044）が弱く付くだけで、深さ（ρ=0.06）・速さ（ρ=−0.22）・周期性（ρ=−0.08）・1 回あたりの持続（ρ=0.10）は無相関。窓 [−1.5,+0.3] の深さ中央値だけ ρ=0.48（p=0.010）だが主窓では消える＝窓の取り方で変わる値で、17 項目の多重比較（Bonferroni α≈0.003）を考えると**上手さに対応する測定量は今のところ無い**。レーダー「ビブラート＆ロングトーン」（{c(None,rd['dam_vibratoSkill'])} で上手さと連動）も同様に測定とは付かない（深さ ρ={f(rd['depth_med']['spearman_rho'],3)}・持続の和 ρ={f(rd['dur_sum']['spearman_rho'],3)}）。")
L.append("")
L.append("## 4. 図")
L.append("")
L.append("`c3_vibrato_散布図.png`：A＝横 DAM 合計秒・縦 測定合計（青＝窓内＋半周期・橙＝全長＋半周期・同じ曲を縦線で結ぶ）。点は右上がりだが縦軸は横軸の 1/5〜1/8 の尺度で、回数を除いた偏相関は 0。B＝タイプ別の深さ箱ひげ（イベント単位・1 件 1 点）。")
L.append("")
L.append("## 5. 結論（3 行）")
L.append("")
L.append(f"1. **回数は一致する**：DAM の vibratoCount と録画から読んだ加点数は画面終値で 27／28 曲一致（残る 1 曲は台帳の欠落＝ケセラセラの末尾 45 s・修正対象）。加点単位→曲単位の往復は回数については成り立つ。")
L.append(f"2. **合計秒は回数ぶんしか一致しない**：DAM 合計秒と測定の持続の和は ρ=0.83〜0.86 だが回数を除くと偏相関 ≈0。DAM は 1 回あたり 3.3 s を数え、私の「連鎖」は 0.4〜0.6 s しか捉えない＝持続の定義（連鎖＝山谷差 ≥15 cent の連続）が DAM の数え方と違う。")
L.append(f"3. **タイプ・上手さは今の測定量と対応しない**：28 曲中 24 曲が同じタイプ（コード 10）で単調性の検定は不能、上手さは深さ・速さ・周期性と無相関（|ρ|≤0.22）。コード対応表は第三者実装準拠で未確証。")
L.append("")
L.append("## 6. 限界")
L.append("")
L.append("- vibratoType のコード→名称は公式資料で確証が取れていない（第三者実装 1 件＋コード 0＝ノンビブの内部整合のみ）。結果画面を録画に含めれば確証できる。")
L.append("- 9/24 はタイプの分散がほぼ無い（24／28 曲がコード 10）ため、分類と測定の単調性はこのセッションでは検定できない。全 200 曲でもコード 10 が 69% で、ボックス形は 1（13 曲）・2（12 曲）に偏る。")
L.append("- 測定の「持続」は連鎖の長さであり DAM の秒数の定義と違う（1 回あたり 0.5 s vs 3.3 s）。半周期の補正・窓の撤廃・曲全体の連鎖のどれでも埋まらない＝定義の問題で、測定精度の問題ではない。")
L.append("- 上手さ（0〜9）は 28 曲中 6 曲が 0 で分布が偏り、n=28 の Spearman は |ρ|≥0.38 で p<0.05 になるが 17 項目の多重比較を考えると有意なものは無い。")
L.append("- 深さは 6 Hz 超で浅く出る（較正）。≤6 Hz に限定した列と速さ補正の列を併記したが、結論（上手さ・タイプとの無相関）はどの列でも変わらない。")
L.append("- 曲全体の連鎖はビブラート以外の揺れ（FP 側）を含む。加点に紐づけない合計秒は総和が DAM の 1.8 倍になる。")
L.append("")
L.append("## 7. 次（他技法とセッションへの展開）")
L.append("")
L.append("1. **持続の定義を DAM に寄せる**：加点時刻の前の「同じ音高の持続区間（有声・音程の土台が一定）」の長さを測り、DAM の 秒/回（3.3 s）と曲ごとに比べる。連鎖条件（山谷差 ≥15 cent・3〜9 Hz）を緩め、途切れ（1 山分の欠け）を許す版も候補。合計秒の偏相関が 0 から動くかが判定基準。")
L.append("2. **「上昇形 D」の検証**：連鎖区間の土台（0.3 s 移動中央値）の傾き [cent/s] を 208 件で測り、コード 10 の曲で正の傾きが優勢か、ボックス形 3 曲で 0 付近かを見る。コード対応の確証にもなる。")
L.append("3. **コード対応の確証**：次回セッションで結果画面（ビブラートタイプの表示）を録画に含める。1 曲で足りる。")
L.append("4. **G1 台帳の修正**：ケセラセラ（10:43）の 221.9 s 以降の加点 4 件（ビブラート）と採点区間の終端。他 Agent の作業完了後に `marks_` 系の再抽出で補う。")
L.append("5. **他セッションへの展開**：同じ表を 9/11・9/15・10/6 で作り、タイプの分散がある曲（ボックス形 A/B/C・1/2/3）を集めて単調性を検定する。全 200 曲ではコード 1（13 曲）・2（12 曲）・13（14 曲）があり、録音と feats が揃う曲を数えるのが先。")
L.append("6. **他技法への展開**：回数の一致（27／28）は技法に依らず検算できるので、しゃくり・こぶし・フォール・アクセント・ハンマリングの G1 回数 vs DAM の各 Count（kobushiCount・shakuriCount・fallCount・accentCount・hammeringOnCount）の表を同じ手順で出す。ロングトーンは longtoneCount・longtoneSkill と「同じ音高の持続区間」の長さで 1 と同時にできる。")
L.append("")
L.append("## 8. 所要時間")
L.append("")
L.append("16:36 開始 → 報告・写し・一覧追記 16:56（約 20 分・DESKTOP-TGOL9I2）。内訳＝コード対応の調査（Web 検索 5 件・リポジトリ取得・録画末尾フレームの確認）約 8 分、整合表と図 約 6 分、窓なし持続の追加測定 約 1 分（208 件＋28 曲全体）、報告 約 5 分。")
open(HERE/"報告_C3整合_ビブラート_2026-10-08.md","w",encoding="utf-8").write("\n".join(L)+"\n")
print("written", len(L), "lines")
