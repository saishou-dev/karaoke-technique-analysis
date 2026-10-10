# -*- coding: utf-8 -*-
"""方式B（演出検出）の集計と、方式A（色付き印）との比較表、目視サンプル 10 件のシートを作る。
出力: summary_flash.md（報告書に差し込む）, summary_flash_by_counter.csv, evidence/flash/sheet_flash_samples.png"""
import os, glob, cv2, numpy as np, pandas as pd
OUT_DIR = r"G:\研究アーカイブ\成果物\動画解析\marks_20260928"
EV = os.path.join(OUT_DIR, "evidence", "flash")

def q(s):
    s = s.dropna()
    if len(s) == 0: return (np.nan, np.nan, np.nan)
    v = s.quantile([0.1, 0.5, 0.9]); return (round(v[0.5], 3), round(v[0.1], 3), round(v[0.9], 3))

def main():
    df = pd.read_csv(os.path.join(OUT_DIR, "marks_20260924.csv"))
    df["okA"] = df.mark_time_sec.notna(); df["okB"] = df.flash_time_sec.notna()
    df["diffAB"] = (df.mark_time_sec - df.flash_time_sec).round(3)     # 印 − 演出（正＝印が後）
    rows = []
    for c, g in list(df.groupby("counter")) + [("ALL", df)]:
        mA, pA10, pA90 = q(g.lead_sec); mB, pB10, pB90 = q(g.lead_flash_sec)
        both = g[g.okA & g.okB]
        dm, d10, d90 = q(both.diffAB)
        rows.append(dict(counter=c, n=len(g), rateA=round(g.okA.mean(), 3), rateB=round(g.okB.mean(), 3),
                         leadA_med=mA, leadA_p10=pA10, leadA_p90=pA90, leadB_med=mB, leadB_p10=pB10, leadB_p90=pB90,
                         n_both=len(both), diff_med=dm, diff_p10=d10, diff_p90=d90,
                         n_within_0p2=int((both.diffAB.abs() <= 0.2).sum()),
                         n_matchnote=int((g.flash_note == "色付き印と一致").sum()), n_multi=int((g.flash_n_in_win >= 2).sum())))
    bc = pd.DataFrame(rows); bc.to_csv(os.path.join(OUT_DIR, "summary_flash_by_counter.csv"), index=False, encoding="utf-8-sig")
    notes = df.flash_note.fillna("").value_counts()
    with open(os.path.join(OUT_DIR, "summary_flash.md"), "w", encoding="utf-8") as f:
        f.write("### 方式B 技法別（lead = T − 演出時刻、diff = 方式Aの印時刻 − 演出時刻）\n\n")
        f.write("| 技法 | 件数 | 抽出率A | 抽出率B | leadA 中央値 [p10,p90] | leadB 中央値 [p10,p90] | 両方あり | diff 中央値 [p10,p90] | \\|diff\\|≤0.2s | 印と一致で選択 | 窓内に演出2つ以上 |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['counter']} | {r['n']} | {r['rateA']:.1%} | {r['rateB']:.1%} | {r['leadA_med']} [{r['leadA_p10']}, {r['leadA_p90']}] | {r['leadB_med']} [{r['leadB_p10']}, {r['leadB_p90']}] | {r['n_both']} | {r['diff_med']} [{r['diff_p10']}, {r['diff_p90']}] | {r['n_within_0p2']} | {r['n_matchnote']} | {r['n_multi']} |\n")
        f.write("\n### 方式B 選択の内訳\n\n| flash_note | 件数 |\n|---|---|\n")
        for k, v in notes.items(): f.write(f"| {k} | {v} |\n")
        f.write("\n### 曲別（方式B 抽出率）\n\n| 曲 | 件数 | 抽出B |\n|---|---|---|\n")
        for v, g in df.groupby("video"): f.write(f"| {v} | {len(g)} | {g.okB.mean():.1%} |\n")
    # 目視サンプル 10 件（技法をまたいで等間隔・両方式がある行を優先）
    ok = df[df.okB].sort_values(["counter", "video", "video_time_sec"])
    pref = ok[ok.okA]; pool = pref if len(pref) >= 10 else ok
    idx = np.linspace(0, len(pool) - 1, min(10, len(pool))).round().astype(int)
    tiles = []
    for _, r in pool.iloc[idx].iterrows():
        p = os.path.join(EV, r.video, f"{r.counter}_{int(r.count_value):03d}_T{r.video_time_sec:.3f}.png")
        if not os.path.exists(p): continue
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        bar = np.zeros((22, img.shape[1], 3), np.uint8)
        txt = f"{r.video[:16]} {r.counter}#{int(r.count_value)} T={r.video_time_sec:.2f} flash={r.flash_time_sec:.2f} (leadB {r.lead_flash_sec:+.2f}) markA={r.mark_time_sec if pd.notna(r.mark_time_sec) else float('nan'):.2f} conf={r.flash_conf:.2f} {r.flash_note}"
        cv2.putText(bar, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(np.vstack([bar, img]))
    if tiles:
        W = max(t.shape[1] for t in tiles)
        sheet = np.vstack([np.pad(t, ((0, 0), (0, W - t.shape[1]), (0, 0))) for t in tiles])
        hdr = np.zeros((30, W, 3), np.uint8); cv2.putText(hdr, "method B (hit-effect) samples: flash-4f | FLASH (yellow box = cursor region) | +4f | +12f", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        open(os.path.join(EV, "sheet_flash_samples.png"), "wb").write(cv2.imencode(".png", np.vstack([hdr, sheet]))[1].tobytes())
    print(bc.to_string()); print(notes.to_string()); print("samples", len(tiles))

if __name__ == "__main__":
    main()
