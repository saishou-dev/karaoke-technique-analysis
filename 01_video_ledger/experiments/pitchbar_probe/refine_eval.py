# -*- coding: utf-8 -*-
"""5段目：方式Aの精緻化（逆探索）の結果を 9/24 と同条件で並べ、正解データの時刻精度を数値にする。

精緻化＝`refine_onsets.py`（既存・無変更で呼ぶだけ）。
印の位置を中心に ±160px の小領域を [印 −2.6 秒, 印 +0.35 秒] で復号し、印自身をテンプレートにして
過去へ辿り、最も早い「印がある」コマを真の出現時刻とする。加点演出のキラキラで一度隠れた分を跨ぐ。

これで 4段目の頭打ち（基本検出の窓は ビブラート 3.0 秒・他 2.2 秒）が外れ、
最大で 2.2+2.6＝4.8 秒（ビブラートは 3.0+2.6＝5.6 秒）前まで遡れる。

測るもの:
  ① 表示先行時間 lead_sec ＝ カウンター増分の時刻 − 印の出現時刻［秒］
     技法別に 中央値・第3四分位・90%点・最大・2秒超の割合 を出し、9/24 と並べる（同じ定義・同じ列）。
  ② 正解データの時刻精度＝①の分布そのもの。
     方式Aで時刻を置き換えると、正解データの時刻は「印の出現時刻」になる。
     カウンター増分との差がその誤差の形なので、中央値で補正したときの残差も出す。

出力:
  <OUT_DIR>/refine_lead_by_counter.csv   技法別（10/14）
  <OUT_DIR>/refine_compare_sessions.csv  10/14 と 9/24 を並べた表
  <OUT_DIR>/refine_accuracy.csv          時刻精度（中央値補正後の残差）
  <OUT_DIR>/refine_outliers.csv          外れ値の一覧（根拠画像の所在つき）
"""
import os, sys, glob, shutil, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P

MARKS_DIR = r"G:\研究アーカイブ\成果物\動画解析\marks_20260928"
CSV_1014 = os.path.join(MARKS_DIR, "marks_20261014.csv")
CSV_0924 = os.path.join(MARKS_DIR, "marks_20260924.csv")
PRE_1014 = os.path.join(MARKS_DIR, "marks_20261014.pre_refine_20261022.bak.csv")
JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "fall": "フォール",
      "vibrato": "ビブラート", "accent": "アクセント", "hammering": "ハンマリング"}
ORDER = ["vibrato", "shakuri", "kobushi", "accent", "hammering", "fall"]

def load(path):
    df = pd.read_csv(path)
    for c in ("mark_time_sec", "lead_sec", "video_time_sec", "mark_time_raw", "mark_time_refined", "back_frames"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def dist(lead):
    lead = lead.dropna()
    if len(lead) == 0:
        return dict(件数=0, 中央値=np.nan, 第3四分位=np.nan, p90=np.nan, 最大=np.nan, 超2秒の割合=np.nan)
    return dict(件数=len(lead), 中央値=round(float(lead.median()), 3),
                第3四分位=round(float(lead.quantile(.75)), 3),
                p90=round(float(lead.quantile(.9)), 3),
                最大=round(float(lead.max()), 3),
                超2秒の割合=round(float((lead > 2.0).mean()), 3))

def by_counter(df, label):
    rows = []
    for c in ORDER:
        g = df[df.counter == c]
        if len(g) == 0: continue
        d = dist(g.lead_sec)
        rows.append(dict(セッション=label, 技法=JP.get(c, c), 処理=len(g),
                         抽出=int(g.mark_time_sec.notna().sum()),
                         抽出率=round(float(g.mark_time_sec.notna().mean()), 3), **d))
    return pd.DataFrame(rows)

def accuracy(df):
    """方式Aで正解データの時刻を置き換えたときの誤差の形。
    技法ごとに中央値で補正した残差（＝印の出現時刻＋中央値 と カウンター増分の差）を出す。"""
    rows = []
    for c in ORDER:
        g = df[(df.counter == c) & df.lead_sec.notna()]
        if len(g) == 0: continue
        med = float(g.lead_sec.median())
        res = (g.lead_sec - med).abs()
        rows.append(dict(技法=JP.get(c, c), 件数=len(g), 補正値=round(med, 3),
                         残差の中央値=round(float(res.median()), 3),
                         残差p90=round(float(res.quantile(.9)), 3),
                         残差最大=round(float(res.max()), 3),
                         残差05秒以内=round(float((res <= 0.5).mean()), 3),
                         残差10秒以内=round(float((res <= 1.0).mean()), 3)))
    allg = df[df.lead_sec.notna()]
    med = float(allg.lead_sec.median()); res = (allg.lead_sec - med).abs()
    rows.append(dict(技法="全体（技法を分けない）", 件数=len(allg), 補正値=round(med, 3),
                     残差の中央値=round(float(res.median()), 3),
                     残差p90=round(float(res.quantile(.9)), 3),
                     残差最大=round(float(res.max()), 3),
                     残差05秒以内=round(float((res <= 0.5).mean()), 3),
                     残差10秒以内=round(float((res <= 1.0).mean()), 3)))
    return pd.DataFrame(rows)

def outliers(df, n=3, thr=2.0):
    g = df[df.lead_sec > thr].sort_values("lead_sec", ascending=False)
    rows = []
    for _, r in g.iterrows():
        rows.append(dict(video=r.video, 技法=JP.get(r.counter, r.counter), count_value=int(r.count_value),
                         カウンター増分=round(float(r.video_time_sec), 3),
                         印の出現=round(float(r.mark_time_sec), 3),
                         遅れ=round(float(r.lead_sec), 3),
                         精緻化前=round(float(r.mark_time_raw), 3) if pd.notna(r.mark_time_raw) else np.nan,
                         遡り=int(r.back_frames) if pd.notna(r.back_frames) else -1,
                         根拠画像=f"evidence/refined/{r.video}/{r.counter}_{int(r.count_value):03d}_T{float(r.video_time_sec):.3f}.png"))
    return pd.DataFrame(rows)

def copy_evidence(out_rows, n=3):
    """外れ値の根拠画像を作業場所へ写す（精緻化が自前で書いたものを使う）"""
    dst = os.path.join(P.OUT_DIR, "refine_evidence")
    os.makedirs(dst, exist_ok=True)
    got = []
    for _, r in out_rows.head(n).iterrows():
        src = os.path.join(MARKS_DIR, r["根拠画像"].replace("/", os.sep))
        if os.path.exists(src):
            fn = f"outlier_{r['技法']}_{r.video}_{int(r.count_value):03d}_lead{r['遅れ']:.3f}.png"
            shutil.copyfile(src, os.path.join(dst, fn)); got.append(fn)
    return got

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n_out", type=int, default=3)
    a = ap.parse_args()
    new = load(CSV_1014)
    print(f"10/14 行数 {len(new)}  抽出 {int(new.mark_time_sec.notna().sum())}  抽出率 {new.mark_time_sec.notna().mean():.3f}")
    if os.path.exists(PRE_1014):
        pre = load(PRE_1014)
        print(f"精緻化前 抽出 {int(pre.mark_time_sec.notna().sum())}  抽出率 {pre.mark_time_sec.notna().mean():.3f}")
    print()
    t_new = by_counter(new, "10/14（精緻化あり）")
    t_old = by_counter(load(CSV_0924), "9/24（精緻化あり）")
    both = pd.concat([t_new, t_old], ignore_index=True)
    both.to_csv(os.path.join(P.OUT_DIR, "refine_compare_sessions.csv"), index=False, encoding="utf-8-sig")
    t_new.to_csv(os.path.join(P.OUT_DIR, "refine_lead_by_counter.csv"), index=False, encoding="utf-8-sig")
    print("=== 技法別の表示先行時間（2 セッションを並べる） ===")
    print(both.to_string(index=False))
    acc = accuracy(new)
    acc.to_csv(os.path.join(P.OUT_DIR, "refine_accuracy.csv"), index=False, encoding="utf-8-sig")
    print()
    print("=== 正解データの時刻精度（技法ごとに中央値で補正したときの残差・10/14） ===")
    print(acc.to_string(index=False))
    out = outliers(new)
    out.to_csv(os.path.join(P.OUT_DIR, "refine_outliers.csv"), index=False, encoding="utf-8-sig")
    got = copy_evidence(out, a.n_out)
    print()
    print(f"=== 2 秒を超えた件（{len(out)} 件・上位 {len(got)} 件の根拠画像を写した） ===")
    if len(out): print(out.head(8).to_string(index=False))
    for g in got: print("  根拠画像", g)
