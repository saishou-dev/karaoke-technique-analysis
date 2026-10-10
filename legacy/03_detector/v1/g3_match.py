# -*- coding: utf-8 -*-
"""ブラインド実験の照合（2026-09-10）：予測（WAV時間軸）と正解G1（動画時間軸・pts補正済み）を同期オフセットで揃えて、種類別に再現率・適合率・F1。
   使い方: python g3_match.py --pred dataset/blind_20260911/<stem>_pred.csv --video <video_id> --offset <wav_time - video_time 秒> [--tol 1.0]
   正解＝dataset/G1_events_all.csv（video一致・multi_increment=0）。予測時刻 time_sec - offset = 動画時刻。
   許容窓＝学習と同じ非対称窓（予測アンカーの -0.3〜+0.9s／ビブラートは -0.2〜+1.0s）を tol に応じて広げる。
"""
import argparse, csv, json, os
import pandas as pd
ROOT = r"G:\研究アーカイブ\成果物\動画解析"
WIN = {"shakuri": (-0.3, 0.9), "kobushi": (-0.3, 0.9), "fall": (-0.3, 0.9), "vibrato": (-0.2, 1.0)}
ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--video", required=True); ap.add_argument("--offset", type=float, required=True); ap.add_argument("--tol", type=float, default=1.0)
a = ap.parse_args()
pred = pd.read_csv(a.pred, encoding="utf-8-sig"); pred["vt"] = pred["time_sec"] - a.offset
g1 = pd.read_csv(os.path.join(ROOT, "dataset", "G1_events_all.csv"), encoding="utf-8-sig")
g1 = g1[(g1["video"] == a.video) & (g1["multi_increment"] == 0)]
out = []
for typ, (lo, hi) in WIN.items():
    ex = a.tol - 0.5; P = pred[pred["type"] == typ]; T = g1[g1["counter"] == typ]
    hit_t = sum(1 for _, e in T.iterrows() if any((e["video_time_sec"] - p) >= lo - ex and (e["video_time_sec"] - p) <= hi + ex for p in P["vt"]))
    hit_p = sum(1 for p in P["vt"] if any((e - p) >= lo - ex and (e - p) <= hi + ex for e in T["video_time_sec"]))
    rec = hit_t / len(T) if len(T) else float("nan"); prec = hit_p / len(P) if len(P) else float("nan")
    f1 = 2 * rec * prec / (rec + prec) if len(T) and len(P) and (rec + prec) > 0 else float("nan")
    out.append(dict(type=typ, true=len(T), pred=len(P), hit_true=hit_t, hit_pred=hit_p, recall=round(rec, 3), precision=round(prec, 3), f1=round(f1, 3) if f1 == f1 else ""))
df = pd.DataFrame(out); print(df.to_string(index=False))
df.to_csv(a.pred.replace("_pred.csv", f"_match_tol{a.tol}.csv"), index=False, encoding="utf-8-sig")
