# -*- coding: utf-8 -*-
"""G1生成（帯pts＋同期オフセット付与）と、ブラインド予測との照合 DAM Ai 版（2026-09-15・g1_and_match_20260915.py と同一方式・設定だけ CfgAi）。
   (1) 時刻は <pts_bands_ai_<session>>/<video>.csv（帯クロップ動画自身のpts）だけで引く。元動画の pts 表（dataset/pts/）は参照しない。
   (2) ゲート：読取結果のある動画に帯pts表が無い／表の行数より大きい frame 番号がある／表が _gate.csv で FAIL → G1を書かずに停止（終了コード2）。
   (3) 同期表が無ければ audio_time_sec 空欄で G1 を書き WARN。 (4) 照合は blind_<session>/<T>_pred.csv があるときだけ。
   G1行: video,counter,count_value,delta,frame,event_frame_est,video_time_sec,stable_read_time_sec,wav,audio_time_sec,multi_increment,pts_source(=pts_bands_ai_<S>)
   使い方: python scripts/g1_and_match_20260915ai.py [session]（KV_AI_BASE でセルフテスト側へ・KV_G1_OUT/KV_RESULTS_DIR/KV_PTS_BAND_DIR も可）
"""
import os, sys, csv
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915ai import CfgAi

C = CfgAi(); S = C.S
if os.environ.get("KV_G1_OUT"): C.g1_csv = os.environ["KV_G1_OUT"]
if os.environ.get("KV_RESULTS_DIR"): C.results_dir = os.environ["KV_RESULTS_DIR"]
if os.environ.get("KV_PTS_BAND_DIR"): C.pts_band_dir = os.environ["KV_PTS_BAND_DIR"]; C.gate_csv = os.path.join(C.pts_band_dir, "_gate.csv")

def stop(reason):
    C.log("G1 STOP: " + reason); C.log("G1 STOP: " + reason, C.gate_log)
    print("G1 STOP:", reason, flush=True); raise SystemExit(2)

if not os.path.isdir(C.results_dir): stop(f"読取結果ディレクトリが無い {C.results_dir}")
videos = sorted(d for d in os.listdir(C.results_dir) if os.path.exists(os.path.join(C.results_dir, d, "counter_events.csv")) and ".mp4.part" not in d)
if not videos: stop("counter_events.csv が1本もない")
failed = {r["video"] for r in C.gate_failures()}

pts = {}; problems = []
for name in videos:
    p = os.path.join(C.pts_band_dir, name + ".csv")
    if name in failed: problems.append(f"{name}: 帯ptsゲート FAIL（_gate.csv）"); continue
    if not os.path.exists(p): problems.append(f"{name}: 帯pts表が無い {p}"); continue
    pts[name] = C.read_pts_csv(p)
    ev = pd.read_csv(os.path.join(C.results_dir, name, "counter_events.csv"))
    if len(ev):
        mxf = int(max(ev["frame"].max(), ev["event_frame_est"].max()))
        if mxf >= len(pts[name]): problems.append(f"{name}: frame番号 {mxf} が帯pts表の行数 {len(pts[name])} を超える（表と動画の不一致）")
if problems: stop("；".join(problems))
C.log(f"G1(ai) gate OK videos={len(videos)} (帯pts表 {len(pts)} 本・元動画ptsは不使用)")

if os.path.exists(C.sync_csv):
    sync = pd.read_csv(C.sync_csv, encoding="utf-8-sig")
else:
    sync = pd.DataFrame(columns=["video", "wav", "offset_sec"]); C.log(f"G1 WARN: 同期表なし {C.sync_csv} → audio_time_sec は空欄（sync後に再実行）")

g1 = []
for name in videos:
    ev = pd.read_csv(os.path.join(C.results_dir, name, "counter_events.csv")); ev = ev[ev["delta"] > 0]; p = pts[name]
    srow = sync[sync["video"] == name]
    for _, e in ev.iterrows():
        fi = int(e["event_frame_est"]); si = int(e["frame"]); vt = p[fi]; st = p[si]
        for _, s in (srow.iterrows() if len(srow) else [(0, None)]):
            g1.append(dict(video=name, counter=e["counter"], count_value=int(e["value"]), delta=int(e["delta"]), frame=si, event_frame_est=fi,
                           video_time_sec=round(vt, 3), stable_read_time_sec=round(st, 3),
                           wav=(s["wav"] if s is not None else ""), audio_time_sec=(round(vt + float(s["offset_sec"]), 3) if s is not None else None),
                           multi_increment=int(e["delta"] > 1), pts_source=C.pts_source))
G = pd.DataFrame(g1); os.makedirs(os.path.dirname(C.g1_csv), exist_ok=True)
G.to_csv(C.g1_csv, index=False, encoding="utf-8-sig")
C.log(f"G1 WROTE {C.g1_csv} rows={len(G)} {G['counter'].value_counts().to_dict() if len(G) else {}}")

WIN = {"shakuri": (-0.3, 0.9), "kobushi": (-0.3, 0.9), "fall": (-0.3, 0.9), "vibrato": (-0.2, 1.0)}
out = []
for wav in sorted(set(G["wav"])) if len(G) else []:
    if not wav: continue
    stem = wav[:-4]; pf = os.path.join(C.blind_dir, stem + "_pred.csv")
    if not os.path.exists(pf): continue
    P = pd.read_csv(pf, encoding="utf-8-sig"); T = G[(G["wav"] == wav) & (G["multi_increment"] == 0)]
    for typ, (lo, hi) in WIN.items():
        for tol in (1.0, 0.5):
            ex = tol - 0.5; Pt = P[P["type"] == typ]["time_sec"].values; Tt = T[T["counter"] == typ]["audio_time_sec"].values
            hit_t = sum(1 for t in Tt if any(lo - ex <= t - q <= hi + ex for q in Pt)); hit_p = sum(1 for q in Pt if any(lo - ex <= t - q <= hi + ex for t in Tt))
            out.append(dict(wav=stem, type=typ, tol=tol, true=len(Tt), pred=len(Pt), hit_true=hit_t, hit_pred=hit_p,
                            recall=(hit_t / len(Tt) if len(Tt) else np.nan), precision=(hit_p / len(Pt) if len(Pt) else np.nan)))
if out:
    M = pd.DataFrame(out); M.to_csv(os.path.join(C.blind_dir, f"match_{S}.csv"), index=False, encoding="utf-8-sig")
    Sm = M.groupby(["type", "tol"]).agg(true=("true", "sum"), pred=("pred", "sum"), hit_true=("hit_true", "sum"), hit_pred=("hit_pred", "sum")).reset_index()
    Sm["recall"] = (Sm["hit_true"] / Sm["true"]).round(3); Sm["precision"] = (Sm["hit_pred"] / Sm["pred"]).round(3)
    Sm["f1"] = (2 * Sm["recall"] * Sm["precision"] / (Sm["recall"] + Sm["precision"])).round(3)
    Sm.to_csv(os.path.join(C.blind_dir, f"match_summary_{S}.csv"), index=False, encoding="utf-8-sig")
    C.log("MATCH\n" + Sm.to_string(index=False))
else:
    C.log(f"MATCH skipped（{C.blind_dir} に *_pred.csv が無い、または同期表なし）")
print("G1 DONE", flush=True)
