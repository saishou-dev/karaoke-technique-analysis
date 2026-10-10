# -*- coding: utf-8 -*-
"""複数セッションの marks_<session>.csv をまとめて集計する。
出力: summary_sessions.csv（セッション×技法の抽出率・lead 中央値/p10/p90）, summary_sessions.md（報告書用の表）"""
import os, sys, numpy as np, pandas as pd
OUT_DIR = r"G:\研究アーカイブ\成果物\動画解析\marks_20260928"
SESSIONS = ["20260812", "20260818", "20260902", "20260911", "20260924"]

def q(s):
    s = s.dropna()
    return (np.nan, np.nan, np.nan) if len(s) == 0 else tuple(round(float(s.quantile(p)), 2) for p in (0.5, 0.1, 0.9))

def main():
    rows = []
    for ses in SESSIONS:
        p = os.path.join(OUT_DIR, f"marks_{ses}.csv")
        if not os.path.exists(p): continue
        df = pd.read_csv(p); df["ok"] = df.mark_time_sec.notna()
        for c, g in list(df.groupby("counter")) + [("ALL", df)]:
            m, p10, p90 = q(g.lead_sec[g.ok])
            inval = int(g.note.fillna("").str.contains("無効化").sum())
            rows.append(dict(session=ses, counter=c, n=len(g), n_ok=int(g.ok.sum()), rate=round(g.ok.mean(), 3), lead_med=m, lead_p10=p10, lead_p90=p90, n_invalidated=inval))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT_DIR, "summary_sessions.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT_DIR, "summary_sessions.md"), "w", encoding="utf-8") as f:
        f.write("| セッション | 技法 | 件数 | 抽出 | 抽出率 | lead 中央値 | p10 | p90 | ゲート無効化 |\n|---|---|---|---|---|---|---|---|---|\n")
        for _, r in t.iterrows():
            f.write(f"| {r.session} | {r.counter} | {r.n} | {r.n_ok} | {r.rate:.1%} | {r.lead_med} | {r.lead_p10} | {r.lead_p90} | {r.n_invalidated} |\n")
    piv = t[t.counter != "ALL"].pivot(index="counter", columns="session", values="rate")
    print(piv.to_string()); print(t[t.counter == "ALL"].to_string())

if __name__ == "__main__":
    main()
