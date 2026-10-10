# -*- coding: utf-8 -*-
"""marks_20260924.csv を集計し、技法別・曲別の抽出率、lead_sec の分布、外れ値一覧を出す。
出力: summary_by_counter.csv / summary_by_video.csv / outliers.csv / summary.md（報告書に貼る表）"""
import os, sys, numpy as np, pandas as pd

OUT_DIR = r"G:\研究アーカイブ\成果物\動画解析\marks_20260928"
AUG = {"shakuri": 0.30, "kobushi": 0.47, "vibrato": 0.90, "fall": 0.51}   # 8月の中央値

def main():
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--session", default="20260924"); a = ap.parse_args()
    sfx = "" if a.session == "20260924" else f"_{a.session}"
    df = pd.read_csv(os.path.join(OUT_DIR, f"marks_{a.session}.csv"))
    df["ok"] = df.mark_time_sec.notna()
    rows = []
    for c, g in df.groupby("counter"):
        ok = g[g.ok]
        q = ok.lead_sec.quantile([0.1, 0.5, 0.9]) if len(ok) else pd.Series([np.nan]*3, index=[0.1, 0.5, 0.9])
        rows.append(dict(counter=c, n=len(g), n_ok=len(ok), rate=round(len(ok)/len(g), 3),
                         lead_p10=round(q[0.1], 3), lead_med=round(q[0.5], 3), lead_p90=round(q[0.9], 3),
                         lead_mean=round(ok.lead_sec.mean(), 3) if len(ok) else np.nan,
                         aug_median=AUG.get(c, np.nan),
                         n_after_flash=int((ok.after_flash == 1).sum()) if len(ok) else 0))
    ok = df[df.ok]
    q = ok.lead_sec.quantile([0.1, 0.5, 0.9])
    rows.append(dict(counter="ALL", n=len(df), n_ok=len(ok), rate=round(len(ok)/len(df), 3),
                     lead_p10=round(q[0.1], 3), lead_med=round(q[0.5], 3), lead_p90=round(q[0.9], 3),
                     lead_mean=round(ok.lead_sec.mean(), 3), aug_median=np.nan,
                     n_after_flash=int((ok.after_flash == 1).sum())))
    bc = pd.DataFrame(rows); bc.to_csv(os.path.join(OUT_DIR, f"summary_by_counter{sfx}.csv"), index=False, encoding="utf-8-sig")
    bv = df.groupby("video").agg(n=("ok", "size"), n_ok=("ok", "sum")).reset_index()
    bv["rate"] = (bv.n_ok / bv.n).round(3)
    bv.to_csv(os.path.join(OUT_DIR, f"summary_by_video{sfx}.csv"), index=False, encoding="utf-8-sig")
    # 外れ値：技法別中央値から ±0.6s 以上、または lead<0（マークが T より後）
    med = ok.groupby("counter").lead_sec.median()
    ok2 = ok.copy(); ok2["dev"] = ok2.lead_sec - ok2.counter.map(med)
    out = ok2[(ok2.dev.abs() > 0.6) | (ok2.lead_sec < 0)].sort_values("dev")
    out.to_csv(os.path.join(OUT_DIR, f"outliers{sfx}.csv"), index=False, encoding="utf-8-sig")
    # 失敗理由の集計
    fail = df[~df.ok].copy()
    fail["reason"] = fail.note.fillna("").str.replace(r"（.*?）", "", regex=True).str.split("；").str[0]
    fr = fail.reason.value_counts()
    with open(os.path.join(OUT_DIR, f"summary{sfx}.md"), "w", encoding="utf-8") as f:
        f.write("## 技法別\n\n| 技法 | 件数 | 抽出 | 抽出率 | lead p10 | 中央値 | p90 | 平均 | 8月中央値 | 演出直後 |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['counter']} | {r['n']} | {r['n_ok']} | {r['rate']:.1%} | {r['lead_p10']} | {r['lead_med']} | {r['lead_p90']} | {r['lead_mean']} | {r['aug_median']} | {r['n_after_flash']} |\n")
        f.write("\n## 曲別\n\n| 曲 | 件数 | 抽出 | 抽出率 |\n|---|---|---|---|\n")
        for _, r in bv.iterrows():
            f.write(f"| {r.video} | {r.n} | {int(r.n_ok)} | {r.rate:.1%} |\n")
        f.write("\n## 失敗理由\n\n| 理由 | 件数 |\n|---|---|\n")
        for k, v in fr.items(): f.write(f"| {k} | {v} |\n")
        f.write(f"\n## 外れ値（技法別中央値から±0.6s超 または lead<0）: {len(out)}件\n\n")
        f.write(out[["video", "counter", "count_value", "video_time_sec", "mark_time_sec", "lead_sec", "conf", "note"]].head(40).to_markdown(index=False))
        f.write("\n")
    print(bc.to_string()); print(bv.to_string()); print(fr.to_string()); print("outliers", len(out))

if __name__ == "__main__":
    main()
