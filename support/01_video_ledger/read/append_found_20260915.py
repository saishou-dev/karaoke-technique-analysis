# -*- coding: utf-8 -*-
"""9/15 再走査版 G1（dataset/G1_events_20260915_rescan.csv）に、目視で増加を確認した欠落候補を追記する（2026-09-15 23:2x・Windows python）。
   判定＝rescan_missing_20260915.py の FOUND 12件＋目視で見つけた1件（天国 hammering 23）。増加フレームは found_refine／found_dense の
   タイル（dataset/rescan_20260915/evidence/refined_*.png・dense_*.png）で目視して確定（±3フレーム）。
   退避＝G1_events_20260915_rescan_pre_found.bak.csv（初回のみ）。追記行は note 列＝found_visual_20260915。冪等（note 列が既にあれば何もしない）。"""
import os, csv, shutil, pandas as pd
ROOT = r"G:\研究アーカイブ\成果物\動画解析"; G1 = os.path.join(ROOT, "dataset", "G1_events_20260915_rescan.csv")
BAK = G1[:-4] + "_pre_found.bak.csv"; PTS = os.path.join(ROOT, "dataset", "pts_bands_20260915")
# (video, counter, value, event_frame, 根拠)
ADD = [("ニュー・マイ・ノーマル_20260915_111844", "hammering", 6, 14028, "refined f14028: 5→6 を +0/+1 で確認"),
       ("天国_20260915_113749", "hammering", 22, 13593, "refined f13593: 21→22 を +0 で確認"),
       ("天国_20260915_113749", "hammering", 23, 13650, "refined f13650: 22→23 を +0 で確認（FOUND 外・montage +3 で発見）"),
       ("地球儀 with Vaundy_20260915_120243", "shakuri", 9, 11805, "refined f11805: 8→9 を +0/+1 で確認"),
       ("おもかげ (produced by Vaundy)_20260915_120810", "shakuri", 3, 4197, "dense: f4195=2 → f4198=3"),
       ("おもかげ (produced by Vaundy)_20260915_120810", "shakuri", 4, 5073, "dense: f5071=3 → f5074=4"),
       ("おもかげ (produced by Vaundy)_20260915_120810", "shakuri", 5, 5449, "dense: f5447=4 → f5450=5"),
       ("Mela!(生音)_20260915_125548", "hammering", 6, 12823, "dense: f12820=5 → f12825=6"),
       ("Mela!(生音)_20260915_131253", "shakuri", 11, 11929, "refined f11929: 10→11 を +0/+1 で確認（9・10 は未検出＝delta 3）"),
       ("Mela!(生音)_20260915_131253", "accent", 10, 11518, "dense: f11515=9 → f11520=10")]
G = pd.read_csv(G1, encoding="utf-8-sig")
if "note" in G.columns and (G["note"] == "found_visual_20260915").any(): print("already appended"); raise SystemExit(0)
if not os.path.exists(BAK): shutil.copy(G1, BAK)
rows = []
for v, c, val, f, why in ADD:
    pts = [float(r["pts_sec"]) for r in csv.DictReader(open(os.path.join(PTS, v + ".csv"), encoding="utf-8"))]
    ce = pd.read_csv(os.path.join(ROOT, "results_wao_20260915_rescan", v, "counter_events.csv"))  # 読取器の最終値（初期値 delta=0 行も含む）を前値にする
    prev = int(ce[ce["counter"] == c]["value"].max()) if (ce["counter"] == c).any() else 0
    prev = max(prev, max([r["count_value"] for r in rows if r["video"] == v and r["counter"] == c] or [0]))
    d = val - prev
    rows.append(dict(video=v, counter=c, count_value=val, delta=d, frame=f, event_frame_est=f, video_time_sec=round(pts[f], 3), stable_read_time_sec=round(pts[f], 3),
                     wav=v + ".wav", audio_time_sec=round(pts[f], 3), multi_increment=int(d > 1), pts_source="pts_bands_20260915", note="found_visual_20260915"))
    print(f"{v[:24]:26s} {c:10s} {val:3d} delta={d} f={f} t={pts[f]:.3f}  {why}")
A = pd.DataFrame(rows); G["note"] = ""; out = pd.concat([G, A], ignore_index=True)
out = out.sort_values(["video", "video_time_sec", "counter"], kind="stable").reset_index(drop=True)
out.to_csv(G1, index=False, encoding="utf-8-sig"); print("WROTE", G1, len(G), "->", len(out), "backup", os.path.basename(BAK))
