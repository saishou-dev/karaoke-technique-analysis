# -*- coding: utf-8 -*-
"""正解v2（技法マーク出現時刻）の作成と効果の仮判定（2026-10-08・E3）。
入力: marks_<session>.csv（detect_marks → rematch → refine_onsets 済み。mark_time_sec＝ゲート通過の精緻化時刻）
      dataset/G1_events_all.csv（正解v1＝カウンター増分時刻）
      dataset/ledgers_20261008/検出照合台帳_test<session>.csv（学習済みCRNNの予測時刻。再予測はしない）
      g3v2_20260928/index.json（offset＝録音WAV時刻−録画時刻。読むだけ）
出力（marks_20260928/）:
  marks_table_<session>.csv        … video, technique, mark_time_sec（＋対応した G1 の count_value・T・lead）
  mark_lead_stats_<session>.csv/.md … 技法別の対応率・差（T−mark）の中央値・四分位・p10/p90・期待範囲内率
  G1_events_v2_<session>.csv       … G1 の元の列＋mark_time_sec＋mark_dt_sec＋mark_status（mark/none/skipped_multi）
                                      ＋truth_v2_video_time_sec／truth_v2_audio_time_sec（v2 の正解時刻＝mark があればそれ、無ければ元の時刻）
  v2_eval_f1_<session>.csv/.md     … 正解 v1→v2 置換時の技法別 F1（±1s・±0.5s）。照合は build_ledgers_20261008.py と同じ貪欲1対1
使い方: python build_v2_and_eval.py --session 20260924 [--no_eval]"""
import os, sys, json, argparse, collections
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

BASE = r"G:\研究アーカイブ\成果物\動画解析"
OUT_DIR = os.path.join(BASE, "marks_20260928")
G1_CSV = os.path.join(BASE, "dataset", "G1_events_all.csv")
INDEX = os.path.join(BASE, "g3v2_20260928", "index.json")
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
JA = {"shakuri": "しゃくり", "kobushi": "こぶし", "vibrato": "ビブラート", "fall": "フォール", "accent": "アクセント", "hammering": "ハンマリング", "ALL": "全体"}
EXPECT = (0.2, 1.5)   # 期待＝マークはカウンター増分の数百ms〜1.5s前

def q(s, p):
    s = s.dropna(); return round(float(s.quantile(p)), 3) if len(s) else np.nan

def match(pred, truth, tol):
    """貪欲1対1照合（build_ledgers_20261008.py / render_pitchbar.py と同じ）"""
    used = set(); pairs = []
    for pi in sorted(range(len(pred)), key=lambda i: pred[i]):
        best = None
        for k, t in enumerate(truth):
            if k in used or abs(t - pred[pi]) > tol: continue
            if best is None or abs(t - pred[pi]) < abs(truth[best] - pred[pi]): best = k
        if best is not None: used.add(best); pairs.append((pi, best))
    return pairs

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

def build_v2(session):
    marks = pd.read_csv(os.path.join(OUT_DIR, f"marks_{session}.csv"))
    g1 = pd.read_csv(G1_CSV)
    g = g1[g1.session == int(session)].copy()
    m = marks[["video", "counter", "count_value", "mark_time_sec", "lead_sec", "conf", "method", "note", "mark_x", "mark_y"]].copy()
    m = m.rename(columns={"conf": "mark_conf", "method": "mark_method", "note": "mark_note"})
    v2 = g.merge(m, on=["video", "counter", "count_value"], how="left")
    v2["mark_dt_sec"] = (v2.video_time_sec - v2.mark_time_sec).round(3)       # 正＝マークがカウンター増分より前
    v2["mark_status"] = np.where(v2.mark_time_sec.notna(), "mark", np.where(v2.multi_increment == 1, "skipped_multi", "none"))
    off = (v2.audio_time_sec - v2.video_time_sec)
    v2["truth_v2_video_time_sec"] = np.where(v2.mark_status == "mark", v2.mark_time_sec, v2.video_time_sec).round(3)
    v2["truth_v2_audio_time_sec"] = (v2.truth_v2_video_time_sec + off).round(3)
    v2["mark_method"] = v2.mark_method.where(v2.mark_status == "mark", "")
    v2["mark_note"] = v2.mark_note.fillna("")
    v2 = v2.sort_values(["video", "video_time_sec", "counter", "count_value"]).reset_index(drop=True)
    v2.to_csv(os.path.join(OUT_DIR, f"G1_events_v2_{session}.csv"), index=False, encoding="utf-8-sig")
    # マーク表
    tb = v2[v2.mark_status == "mark"][["video", "counter", "mark_time_sec", "count_value", "video_time_sec", "mark_dt_sec", "mark_conf", "mark_x", "mark_y", "mark_method"]].rename(
        columns={"counter": "technique", "count_value": "g1_count_value", "video_time_sec": "g1_time_sec", "mark_dt_sec": "lead_sec"})
    tb.sort_values(["video", "mark_time_sec"]).to_csv(os.path.join(OUT_DIR, f"marks_table_{session}.csv"), index=False, encoding="utf-8-sig")
    # 差の分布
    ev = v2[v2.multi_increment == 0]
    rows = []
    for c, gg in list(ev.groupby("counter")) + [("ALL", ev)]:
        ok = gg[gg.mark_status == "mark"]; d = ok.mark_dt_sec
        rows.append(dict(technique=c, technique_ja=JA[c], n_events=len(gg), n_mark=len(ok), rate=round(len(ok) / len(gg), 3) if len(gg) else np.nan,
                         rate_none=round(1 - len(ok) / len(gg), 3) if len(gg) else np.nan,
                         dt_median=q(d, .5), dt_q1=q(d, .25), dt_q3=q(d, .75), dt_p10=q(d, .1), dt_p90=q(d, .9), dt_mean=round(float(d.mean()), 3) if len(d) else np.nan,
                         in_expect=round(float(((d >= EXPECT[0]) & (d <= EXPECT[1])).mean()), 3) if len(d) else np.nan,
                         n_negative=int((d < 0).sum()), n_invalidated=int(gg.mark_note.str.contains("無効化").sum())))
    st = pd.DataFrame(rows); st.to_csv(os.path.join(OUT_DIR, f"mark_lead_stats_{session}.csv"), index=False, encoding="utf-8-sig")
    L = [f"### 差の分布（{session}・mark_dt = T(カウンター増分) − mark 出現、秒。multi_increment=0 の {len(ev)} 件）", "",
         "| 技法 | 件数 | 対応 | 対応率 | 未対応率 | 中央値 | Q1 | Q3 | p10 | p90 | 平均 | 0.2〜1.5s 内 | 負(マークが後) | ゲート無効化 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['technique_ja']} | {r['n_events']} | {r['n_mark']} | {r['rate']:.1%} | {r['rate_none']:.1%} | {r['dt_median']} | {r['dt_q1']} | {r['dt_q3']} | {r['dt_p10']} | {r['dt_p90']} | {r['dt_mean']} | {r['in_expect']:.1%} | {r['n_negative']} | {r['n_invalidated']} |")
    bv = ev.groupby("video").apply(lambda x: pd.Series(dict(n=len(x), n_mark=int((x.mark_status == "mark").sum()), dt_median=q(x.mark_dt_sec, .5)))).reset_index()
    bv["rate"] = (bv.n_mark / bv.n).round(3)
    L += ["", "### 曲別", "", "| 曲 | 件数 | 対応 | 対応率 | 差の中央値 |", "|---|---|---|---|---|"]
    for _, r in bv.iterrows(): L.append(f"| {r.video} | {int(r.n)} | {int(r.n_mark)} | {r.rate:.1%} | {r.dt_median} |")
    open(os.path.join(OUT_DIR, f"mark_lead_stats_{session}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    return v2, st

def evaluate(session, v2, st):
    """検出照合台帳の予測（再予測なし）に対し、正解を v1（カウンター）→v2（マーク）に置き換えて F1 を比べる。"""
    lp = os.path.join(BASE, "dataset", "ledgers_20261008", f"検出照合台帳_test{session}.csv")
    if not os.path.exists(lp): print("台帳なし、評価スキップ", lp); return
    led = pd.read_csv(lp)
    idx = {d["video"]: d for d in json.load(open(INDEX, encoding="utf-8")) if d["session"] == str(session)}
    med = {r.technique: r.dt_median for r in st.itertuples()}    # v2b の補完値（技法別中央値）
    ev = v2[v2.multi_increment == 0].copy()
    key = {(r.video, r.counter, round(r.video_time_sec, 3)): r for r in ev.itertuples()}
    variants = ["v1", "v2", "v2b", "v2_markonly"]
    tols = [1.0, 0.5]
    lags = [round(x, 1) for x in np.arange(-1.5, 0.61, 0.1)]      # 診断用：予測時刻の一律シフト（テストで最良を選ぶ＝楽観値）
    agg = {(var, tol): collections.Counter() for var in variants for tol in tols}
    aggc = {(var, tol): collections.defaultdict(collections.Counter) for var in variants for tol in tols}
    difftp = {(var, tol): collections.defaultdict(list) for var in variants for tol in tols}
    lagagg = {(var, lag): collections.defaultdict(collections.Counter) for var in ("v1", "v2") for lag in lags}
    n_truth_lookup_fail = 0; n_mark_in_eval = 0; n_truth_total = 0
    sub = {(var, lag, tol): collections.defaultdict(collections.Counter) for var in ("v1", "v2") for lag in lags for tol in tols}   # マーク対応イベントだけの再現率
    for (video, tech), gl in led.groupby(["video", "technique"]):
        off = float(idx[video]["offset"]) if video in idx else float(gl.truth_time_sec.dropna().iloc[0] - gl.truth_video_time_sec.dropna().iloc[0]) if gl.truth_time_sec.notna().any() else 0.0
        pred = sorted(float(x) for x in gl.pred_time_sec.dropna())
        tv = sorted(float(x) for x in gl.truth_video_time_sec.dropna())
        t1, t2, t2b, tmo = [], [], [], []
        for t in tv:
            n_truth_total += 1
            r = key.get((video, tech, round(t, 3)))
            if r is None:
                cand = [k for k in key if k[0] == video and k[1] == tech and abs(k[2] - t) < 0.01]
                r = key[cand[0]] if cand else None
            if r is None: n_truth_lookup_fail += 1; t1.append(t + off); t2.append(t + off); t2b.append(t + off); continue
            t1.append(r.video_time_sec + off)
            if r.mark_status == "mark":
                n_mark_in_eval += 1; t2.append(r.mark_time_sec + off); t2b.append(r.mark_time_sec + off); tmo.append(r.mark_time_sec + off)
            else:
                t2.append(r.video_time_sec + off); t2b.append(r.video_time_sec - (med.get(tech) if med.get(tech) == med.get(tech) else 0.0) + off)
        truths = {"v1": t1, "v2": t2, "v2b": t2b, "v2_markonly": tmo}
        # マーク対応イベントだけ：同じイベントの時刻を v1（カウンター）／v2（マーク）で与え、予測を一律シフトして再現率を比べる（ばらつきの小さい方が高く出る）
        t1m = [t1[i] for i in range(len(t1)) if t2[i] != t1[i]]; t2m = [t2[i] for i in range(len(t1)) if t2[i] != t1[i]]
        for lag in lags:
            pr = [p + lag for p in pred]
            for tol in tols:
                for var, tr in (("v1", t1m), ("v2", t2m)):
                    tp = len(match(pr, tr, tol)); sub[(var, lag, tol)][tech].update(collections.Counter(tp=tp, fn=len(tr) - tp))
        for var in variants:
            tr = truths[var]
            for tol in tols:
                pairs = match(pred, tr, tol); tp = len(pairs)
                c = collections.Counter(tp=tp, fp=len(pred) - tp, fn=len(tr) - tp)
                agg[(var, tol)].update(c); aggc[(var, tol)][tech].update(c)
                difftp[(var, tol)][tech] += [pred[i] - tr[k] for i, k in pairs]
        for var in ("v1", "v2"):
            for lag in lags:
                pr = [p + lag for p in pred]; pairs = match(pr, truths[var], 1.0); tp = len(pairs)
                lagagg[(var, lag)][tech].update(collections.Counter(tp=tp, fp=len(pred) - tp, fn=len(truths[var]) - tp))
    rows = []
    for var in variants:
        for tol in tols:
            for tech in TYPES + ["ALL"]:
                c = agg[(var, tol)] if tech == "ALL" else aggc[(var, tol)][tech]
                p, r, f = prf(c["tp"], c["fp"], c["fn"])
                d = [x for t_ in (TYPES if tech == "ALL" else [tech]) for x in difftp[(var, tol)][t_]]
                rows.append(dict(variant=var, tol=tol, technique=tech, technique_ja=JA[tech], n_truth=c["tp"] + c["fn"], n_pred=c["tp"] + c["fp"], TP=c["tp"], FP=c["fp"], FN=c["fn"],
                                 precision=round(p, 3), recall=round(r, 3), f1=round(f, 3), tp_diff_median=round(float(np.median(d)), 3) if d else np.nan))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT_DIR, f"v2_eval_f1_{session}.csv"), index=False, encoding="utf-8-sig")
    # 最良ラグ（診断）
    lrows = []
    for var in ("v1", "v2"):
        for tech in TYPES + ["ALL"]:
            best = None
            for lag in lags:
                c = collections.Counter()
                for t_ in (TYPES if tech == "ALL" else [tech]): c.update(lagagg[(var, lag)][t_])
                f = prf(c["tp"], c["fp"], c["fn"])[2]
                if best is None or f > best[0]: best = (f, lag)
            lrows.append(dict(variant=var, technique=tech, best_lag=best[1], f1_best=round(best[0], 3)))
    ldf = pd.DataFrame(lrows); ldf.to_csv(os.path.join(OUT_DIR, f"v2_eval_bestlag_{session}.csv"), index=False, encoding="utf-8-sig")
    srows = []
    for tol in tols:
        for tech in TYPES + ["ALL"]:
            for var in ("v1", "v2"):
                best = None; at0 = np.nan
                for lag in lags:
                    c = collections.Counter()
                    for t_ in (TYPES if tech == "ALL" else [tech]): c.update(sub[(var, lag, tol)][t_])
                    rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else np.nan
                    if abs(lag) < 1e-6: at0 = rec
                    if best is None or (rec == rec and rec > best[0]): best = (rec, lag, c["tp"] + c["fn"])
                srows.append(dict(tol=tol, technique=tech, variant=var, n=best[2], recall_lag0=round(at0, 3) if at0 == at0 else np.nan, best_lag=best[1], recall_best=round(best[0], 3) if best[0] == best[0] else np.nan))
    sdf = pd.DataFrame(srows); sdf.to_csv(os.path.join(OUT_DIR, f"v2_eval_subset_recall_{session}.csv"), index=False, encoding="utf-8-sig")
    # md
    def tbl(tol):
        L = [f"#### ±{tol}s", "", "| 技法 | 正解数 | 予測数 | F1 v1(カウンター) | F1 v2(マーク・未対応は元時刻) | F1 v2b(未対応は T−技法中央値) | F1 マーク対応分のみ(正解数) | TP差の中央値 v1 | v2 |", "|---|---|---|---|---|---|---|---|---|"]
        for tech in TYPES + ["ALL"]:
            g = {v: df[(df.variant == v) & (df.tol == tol) & (df.technique == tech)].iloc[0] for v in variants}
            L.append(f"| {JA[tech]} | {g['v1'].n_truth} | {g['v1'].n_pred} | {g['v1'].f1:.3f} | {g['v2'].f1:.3f} | {g['v2b'].f1:.3f} | {g['v2_markonly'].f1:.3f} ({g['v2_markonly'].n_truth}) | {g['v1'].tp_diff_median} | {g['v2'].tp_diff_median} |")
        return L
    L = [f"### 正解 v1→v2 置換時の F1（{session}・予測＝検出照合台帳 test{session} の CRNN アンサンブル予測をそのまま使用・再学習なし・貪欲1対1照合）", "",
         f"- 評価対象の正解＝台帳の truth（採点区間内）{n_truth_total} 件、うちマーク対応 {n_mark_in_eval} 件（{n_mark_in_eval / max(1, n_truth_total):.1%}）。G1 と結合できなかった正解 {n_truth_lookup_fail} 件は元時刻のまま。",
         "- 予測時刻には学習時の lag +0.3s（v1 基準で dev 9/11 に合わせた値）がそのまま入っている。v2 は正解が平均 0.8s ほど早まるので、同じ予測のままでは一致が減るのが自然。下の「最良ラグ」はその補正量の目安（テストで選ぶため楽観値）。", ""]
    L += tbl(1.0) + [""] + tbl(0.5)
    L += ["", "#### 予測時刻の一律シフト（診断・±1s・テストで最良＝楽観値）", "", "| 技法 | v1 最良ラグ | v1 F1 | v2 最良ラグ | v2 F1 |", "|---|---|---|---|---|"]
    for tech in TYPES + ["ALL"]:
        a = ldf[(ldf.variant == "v1") & (ldf.technique == tech)].iloc[0]; b = ldf[(ldf.variant == "v2") & (ldf.technique == tech)].iloc[0]
        L.append(f"| {JA[tech]} | {a.best_lag:+.1f} | {a.f1_best:.3f} | {b.best_lag:+.1f} | {b.f1_best:.3f} |")
    L += ["", "#### マーク対応イベントだけの再現率（同じイベント・同じ予測。正解時刻だけ v1↔v2 を入れ替え、予測を一律シフトして最良を取る＝正解のばらつきが小さい方が高く出る）", "",
          "| 許容 | 技法 | 件数 | v1 再現率(シフト0) | v1 最良シフト | v1 最良再現率 | v2 再現率(シフト0) | v2 最良シフト | v2 最良再現率 |", "|---|---|---|---|---|---|---|---|---|"]
    for tol in tols:
        for tech in TYPES + ["ALL"]:
            a = sdf[(sdf.tol == tol) & (sdf.technique == tech) & (sdf.variant == "v1")].iloc[0]; b = sdf[(sdf.tol == tol) & (sdf.technique == tech) & (sdf.variant == "v2")].iloc[0]
            L.append(f"| ±{tol}s | {JA[tech]} | {int(a.n)} | {a.recall_lag0} | {a.best_lag:+.1f} | {a.recall_best} | {b.recall_lag0} | {b.best_lag:+.1f} | {b.recall_best} |")
    open(os.path.join(OUT_DIR, f"v2_eval_f1_{session}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--session", default="20260924"); ap.add_argument("--no_eval", action="store_true")
    a = ap.parse_args()
    v2, st = build_v2(a.session)
    if not a.no_eval: evaluate(a.session, v2, st)
