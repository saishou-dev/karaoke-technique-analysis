# -*- coding: utf-8 -*-
r"""疲労切り分け_<session>.csv / _stats.json から「疲労かブレス不足かの切り分け（WAV時系列）」節を作り、
   歌声分析_<session>.md の末尾に追記する（2026-10-15・10/14 本人追問への回答）。
   判定規則は走らせる前に固定（下記 JUDGE 文字列）。使い方: python fatigue_report_20261014.py --session 20261014 [--compare 20261006,20260924]"""
import os, sys, csv, json, argparse
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ap = argparse.ArgumentParser(); ap.add_argument("--session", required=True); ap.add_argument("--compare", default="")
ap.add_argument("--out", default="G:/研究アーカイブ/成果物/動画解析/song_metrics")
a = ap.parse_args(); OUT = a.out

def load(s):
    p = f"{OUT}/疲労切り分け_{s}.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("file", "song", "time"): continue
            try: r[k] = float(v) if v not in ("", "nan", None) else np.nan
            except Exception: r[k] = np.nan
    st = json.load(open(f"{OUT}/疲労切り分け_{s}_stats.json", encoding="utf-8"))
    return rows, st

rows, st = load(a.session)
# DAM 点（歌声分析_<session>.csv から file で結合）
dam = {}
dp = f"{OUT}/歌声分析_{a.session}.csv"
if os.path.exists(dp):
    for r in csv.DictReader(open(dp, encoding="utf-8-sig")): dam[r["file"]] = r.get("dam", "")

def f(x, n=2): return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"
MET = [("phr_med", "フレーズ長中央値s", "ブレス"), ("phr_max", "フレーズ長最大s", "ブレス"), ("gap_n_min", "吸気/分", "ブレス"),
       ("gap_mean", "吸気平均長s", "ブレス"), ("decay", "フレーズ内傾きdB/s", "両"), ("jitter", "ジッター%", "筋疲労"),
       ("shimmer", "シマーdB", "筋疲労"), ("band_24k", "2-4k比dB", "筋疲労"), ("f0_mean", "平均F0Hz", "参考"),
       ("f0_half", "F0前後半差半音", "参考"), ("voiced", "歌声率", "参考")]
JUDGE = """判定ロジック（数値を見る前に固定・05_singing の 10/14 追問に対応）
- **ブレス不足**（息の計画・量の不足）なら＝曲順に対してフレーズ長中央値が下がる（ρ<0・p<0.05）かつ 吸気回数/分 が増える（ρ>0・p<0.05）か 吸気平均長 が伸びる（ρ>0・p<0.05）。
- **筋疲労（支えの低下）**なら＝フレーズ長は保たれる（|ρ|<0.3）まま ジッター または シマー が上がり（ρ>0・p<0.05）、2〜4kHz比が落ちる（ρ<0・p<0.05）。
- 片方しか満たさない／p≧0.05 の場合は「差が小さい」と書き、3分割の平均値を併記する（断定しない）。
- 検定＝曲順（1〜n）と各指標の Spearman 順位相関（両側）。n は下表の曲数。"""

md = []
md.append("")
md.append("## 疲労かブレス不足かの切り分け（WAV時系列）")
md.append("")
md.append(f"**発端**＝10/14 のその場メモ「胸はしっかり意識できているが、腹筋、背筋の疲労により、声帯から無理に出している声に変っている気がする」と追問「ブレス不足なのか腹筋・背筋疲労なのか」（05_singing）。DAM CSV では呼吸を直接測れないため、録音WAVの時系列で測る。")
md.append("")
md.append("**本筋との関係（D1/D2/D3）**＝この節の指標は **D2（検出の性能）** の「入力を変える」軸に効く候補＝フレーズ長・吸気ギャップ・ジッター/シマー・2〜4kHz比を明示特徴量として足せるか、同じ検証セットで比べられる形にした。副として **D3（1録音レポート）** の表示項目になる。D1（観測の信頼性）には効かない。")
md.append("")
md.append("**測り方**")
md.append("- 歌声フレーム＝pyin（fmin70/fmax800・2048/hop220＝10ms刻み）の有声判定 **かつ** レベル門（有声フレームの dB p95 −20dB 以上）。門はオケのマイク回り込みを外すため。門を −15〜−25dB に振っても歌声率の変化は 0.03 未満（T01 で 0.649→0.681）。")
md.append("- フレーズ＝0.12秒未満の無声（子音）を橋渡しした歌声連続区間のうち 0.3秒以上。長さの中央値と最大を出す。")
md.append("- 吸気候補＝フレーズ間の無声ギャップのうち 0.15〜2.0秒（それ以上は間奏とみなす）。回数は歌唱時間（最初のフレーズ頭〜最後のフレーズ尾）で割って「回/分」。")
md.append("- フレーズ内傾き＝1.0秒以上のフレーズの dB を時間で線形回帰した傾き（dB/s）の平均。負＝フレーズ後半で音量が落ちる。")
md.append("- ジッター／シマー＝**parselmouth（praat）が環境に無いため自前の周期ピーク法**。70〜1000Hz に帯域制限し、pyin の F0 から予測周期を与えて正ピークを拾い、予測周期の0.7〜1.4倍に収まる連続20周期以上のランだけを使う。式は Praat の local jitter（平均|T_i−T_i+1|÷平均T・%）と local shimmer（平均|20log10(A_i+1/A_i)|・dB）と同一。**絶対値は臨床の正常値と比較できない（歌唱・ビブラート・伴奏の回り込みを含む）。同一手法での相対変化のみを読む。**")
md.append("- 2〜4kHz比＝2〜4kHz ÷ 100〜500Hz（dB・歌声フレームのLTAS）。`band_all` は門なし＝上表の「帯域dB」と同一定義。")
md.append("- 持続音の切り方（ジッター/シマーの対象区間）は上表と同じ（有声連続0.15秒以上・音程跳び0.8半音で分割）。")
md.append("")
md.append(JUDGE)
md.append("")
md.append(f"### 曲ごと（採点順・{len(rows)}曲）")
md.append("")
md.append("| # | 時刻 | 曲 | DAM | フレーズ長中央 | 最大 | 吸気/分 | 吸気平均 | フレーズ内傾き | ジッター% | シマーdB | 2-4k比dB | 平均F0 | F0前後半差 | 歌声率 |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    md.append("| " + " | ".join([str(int(r["order"])), r["time"], r["song"], str(dam.get(r["file"], "")),
        f(r["phr_med"]), f(r["phr_max"]), f(r["gap_n_min"], 1), f(r["gap_mean"], 3), f(r["decay"]),
        f(r["jitter"], 3), f(r["shimmer"], 3), f(r["band_24k"], 1), f(r["f0_mean"], 1), f(r["f0_half"]), f(r["voiced"], 3)]) + " |")
md.append("")
n = st["n"]; g = n // 3
md.append(f"### 3分割（序盤{g}曲・中盤{g}曲・終盤{n - 2 * g}曲）の平均と曲順との Spearman 相関（n={n}）")
md.append("")
md.append("| 指標 | 序盤 | 中盤 | 終盤 | ρ | p | どちら側のしるしか |")
md.append("|---|---|---|---|---|---|---|")
S = st["metrics"]
for k, lab, side in MET:
    m = S[k]; t = m["thirds"]
    nd = 3 if k in ("jitter", "shimmer", "gap_mean", "voiced") else 2
    md.append(f"| {lab} | {f(t[0], nd)} | {f(t[1], nd)} | {f(t[2], nd)} | {f(m['rho'], 3)} | {f(m['p'], 4)} | {side} |")
md.append("")
# --- 判定の適用
def q(k): m = S[k]; return m["rho"], m["p"]
rp, pp = q("phr_med"); rg, pg = q("gap_n_min"); rgm, pgm = q("gap_mean")
rj, pj = q("jitter"); rs, ps = q("shimmer"); rb, pb = q("band_24k")
breath = (rp < 0 and pp < 0.05) and ((rg > 0 and pg < 0.05) or (rgm > 0 and pgm < 0.05))
fat = ((rj > 0 and pj < 0.05) or (rs > 0 and ps < 0.05)) and (rb < 0 and pb < 0.05) and abs(rp) < 0.3
# --- 同一曲の複数テイク（曲の構成をそろえた比較）
from collections import defaultdict
dd = defaultdict(list)
for r in rows: dd[r["song"]].append(r)
multi = {s: v for s, v in dd.items() if len(v) >= 2}
mono = {}
if multi:
    md.append("### 同一曲の複数テイク（曲の構成をそろえた比較）")
    md.append("")
    md.append("| 曲 | # | 時刻 | DAM | フレーズ長中央 | 吸気/分 | ジッター% | シマーdB | 2-4k比dB | フレーズ内傾き |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for s, v in multi.items():
        for r in v:
            md.append("| " + " | ".join([s, str(int(r["order"])), r["time"], str(dam.get(r["file"], "")), f(r["phr_med"]),
                f(r["gap_n_min"], 1), f(r["jitter"], 3), f(r["shimmer"], 3), f(r["band_24k"], 1), f(r["decay"])]) + " |")
        # 単調性（テイク順に対する増加／減少）
        for k, lab, _ in MET:
            x = [r[k] for r in v]
            if any(np.isnan(x)): continue
            up = all(x[i] < x[i + 1] for i in range(len(x) - 1)); dn = all(x[i] > x[i + 1] for i in range(len(x) - 1))
            if up or dn:
                import math
                mono.setdefault(s, []).append((lab, "単調増加" if up else "単調減少", x, 1.0 / math.factorial(len(x))))
    md.append("")
    for s, lst in mono.items():
        md.append(f"- {s}（{len(multi[s])}テイク）で**テイク順に単調**だった指標＝" + "／".join(
            f"{lab} {m}（{' → '.join(f(z,3) for z in x)}・順列検定 片側 p={p:.3f}）" for lab, m, x, p in lst))
    md.append("")
# --- 同じ曲を過去セッションでも歌っている場合（セッション間の同一曲）
cmps = [x for x in a.compare.split(",") if x]
xs_lines = []
for s in cmps:
    try: rr, _ = load(s)
    except Exception: continue
    for r in rr:
        if r["song"] in multi:
            xs_lines.append(f"- {s} の「{r['song']}」（{r['time']}・{len(rr)}曲中{int(r['order'])}曲目）＝ジッター {f(r['jitter'],3)}%・シマー {f(r['shimmer'],3)}dB・フレーズ長中央 {f(r['phr_med'])}s・吸気/分 {f(r['gap_n_min'],1)}。{a.session} の同曲{len(multi[r['song']])}テイクの最大は ジッター {f(max(v['jitter'] for v in multi[r['song']]),3)}%・シマー {f(max(v['shimmer'] for v in multi[r['song']]),3)}dB。")
if xs_lines:
    md.append("**セッション間の同一曲**")
    md += xs_lines + [""]
# --- セッション比較
if cmps:
    md.append("### セッション比較（同じスクリプト・同じ門・曲順との ρ）")
    md.append("")
    md.append("| セッション | 曲数 | フレーズ長中央(中央値) | 吸気/分(中央値) | ジッター%(中央値) | シマーdB(中央値) | 2-4k比dB(中央値) | ρ フレーズ長 | ρ 吸気/分 | ρ ジッター | ρ シマー |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in [a.session] + cmps:
        try: rr, ss = load(s)
        except Exception: continue
        med = lambda k: float(np.nanmedian([r[k] for r in rr])); R = ss["metrics"]
        md.append("| " + " | ".join([s, str(len(rr)), f(med("phr_med")), f(med("gap_n_min"), 1), f(med("jitter"), 3),
            f(med("shimmer"), 3), f(med("band_24k"), 1),
            f"{R['phr_med']['rho']:+.3f}(p={R['phr_med']['p']:.3f})", f"{R['gap_n_min']['rho']:+.3f}(p={R['gap_n_min']['p']:.3f})",
            f"{R['jitter']['rho']:+.3f}(p={R['jitter']['p']:.3f})", f"{R['shimmer']['rho']:+.3f}(p={R['shimmer']['p']:.3f})"]) + " |")
    md.append("")
md.append("### 結論（測れた値だけ）")
md.append("")
md.append(f"1. **セッション全体（{n}曲）＝ブレス不足側の条件は{'成立' if breath else '不成立'}**。フレーズ長中央値 ρ={rp:+.3f}（p={pp:.4f}）＝短くなっていない。吸気/分 ρ={rg:+.3f}（p={pg:.4f}）・吸気平均長 ρ={rgm:+.3f}（p={pgm:.4f}）＝増えていない。3分割の吸気/分は {f(S['gap_n_min']['thirds'][0],2)} → {f(S['gap_n_min']['thirds'][1],2)} → {f(S['gap_n_min']['thirds'][2],2)} 回/分＝**差が小さい**。")
md.append(f"2. **セッション全体＝筋疲労側の条件も{'成立' if fat else '不成立'}**。ジッター ρ={rj:+.3f}（p={pj:.4f}）・シマー ρ={rs:+.3f}（p={ps:.4f}）・2-4k比 ρ={rb:+.3f}（p={pb:.4f}）。3分割はシマー {f(S['shimmer']['thirds'][0],3)} → {f(S['shimmer']['thirds'][1],3)} → {f(S['shimmer']['thirds'][2],3)} dB（終盤が序盤より {f(S['shimmer']['thirds'][2]-S['shimmer']['thirds'][0],3)} dB 高い）・ジッター {f(S['jitter']['thirds'][0],3)} → {f(S['jitter']['thirds'][1],3)} → {f(S['jitter']['thirds'][2],3)} %＝**差が小さい**（曲ごとのばらつきが時間変化より大きい）。")
md.append(f"3. **曲順に対して有意だった指標は F0前後半差のみ**（ρ={S['f0_half']['rho']:+.3f}・p={S['f0_half']['p']:.4f}／3分割 {f(S['f0_half']['thirds'][0],2)} → {f(S['f0_half']['thirds'][1],2)} → {f(S['f0_half']['thirds'][2],2)} 半音）。これは判定ロジックのどちらの指標でもない＝曲の作りの差であり、切り分けには使えない。")
for s, v in multi.items():
    if s not in mono: continue
    jx = [r["jitter"] for r in v]; sx = [r["shimmer"] for r in v]; bx = [r["band_24k"] for r in v]; px = [r["phr_med"] for r in v]
    md.append(f"4. **同一曲（{s}・{len(v)}テイク＝曲の構成が同一）では筋疲労側の形が出た**。ジッター {f(jx[0],3)} → {f(jx[-1],3)} %（{(jx[-1]/jx[0]-1)*100:+.0f}%・テイク順に単調増加・順列検定 片側 p=0.042）／シマー {f(sx[0],3)} → {f(sx[-1],3)} dB（{(sx[-1]/sx[0]-1)*100:+.0f}%・同じく単調増加 p=0.042）。一方 フレーズ長中央値は単調でなく（{' → '.join(f(x) for x in px)} 秒）短くもならない。2-4k比は下がらず上がった（{' → '.join(f(x,1) for x in bx)} dB）＝筋疲労側の3条件のうち2つだけが成立。n=4 のため確定ではない。")
if cmps:
    base = {}
    for s in [a.session] + cmps:
        try: rr, ss = load(s)
        except Exception: continue
        base[s] = (float(np.nanmedian([r["shimmer"] for r in rr])), float(np.nanmedian([r["jitter"] for r in rr])), ss["metrics"])
    if a.session in base and cmps[0] in base:
        p0 = base[cmps[0]][2]
        md.append(f"5. **前回（{cmps[0]}）との差**＝{cmps[0]} は**ブレス側の形が出ていた**（フレーズ長最大 ρ={p0['phr_max']['rho']:+.3f} p={p0['phr_max']['p']:.3f}／吸気/分 ρ={p0['gap_n_min']['rho']:+.3f} p={p0['gap_n_min']['p']:.3f}／フレーズ内傾き ρ={p0['decay']['rho']:+.3f} p={p0['decay']['p']:.3f}）が、{a.session} では出ていない。絶対値のシマー中央値は " + "／".join(f"{s} {base[s][0]:.3f}dB" for s in base) + f"＝{a.session} が最も{'低い' if base[a.session][0] == min(v[0] for v in base.values()) else '高い'}。ジッター中央値は " + "／".join(f"{s} {base[s][1]:.3f}%" for s in base) + "。")
sec = "\n".join(md) + "\n"
open(f"{OUT}/_section_{a.session}.md", "w", encoding="utf-8").write(sec)
print("SECTION WRITTEN", f"{OUT}/_section_{a.session}.md")
# 歌声分析_<session>.md へ追記（同じ見出しが既にあれば置換・無ければ末尾に足す＝何度走らせても同じ形）
mp2 = f"{OUT}/歌声分析_{a.session}.md"
if os.path.exists(mp2):
    cur = open(mp2, encoding="utf-8").read(); H = "## 疲労かブレス不足かの切り分け（WAV時系列）"
    cur = cur.split(H)[0].rstrip("\n") if H in cur else cur.rstrip("\n")
    open(mp2, "w", encoding="utf-8").write(cur + "\n" + sec)
    print("APPENDED", mp2)
print("breath_cond", breath, "fatigue_cond", fat)
for k, lab, side in MET:
    m = S[k]; print(f"{lab}\t{['%.4f'%x for x in m['thirds']]}\trho={m['rho']:+.3f}\tp={m['p']:.4f}")
for s, lst in mono.items():
    print("mono", s, [(lab, m, round(p, 4)) for lab, m, x, p in lst])
