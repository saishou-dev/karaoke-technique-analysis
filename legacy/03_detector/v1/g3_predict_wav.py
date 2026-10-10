# -*- coding: utf-8 -*-
"""ブラインド実験用：WAVだけから加点箇所を予測する（2026-09-10）。
   使い方: python g3_predict_wav.py <wav1> [<wav2> ...] --out dataset/blind_20260911
   出力: <out>/<stem>_f0.csv（pYIN 10ms）、<out>/<stem>_pred.csv（type, time_sec, prob, features）、<out>/summary.csv（曲ごとの予測回数）
   モデル: models/g3_v1/<type>.joblib + thresholds.json（学習スクリプトと同じ特徴量関数を使う）
"""
import os, sys, csv, json, argparse, time
import numpy as np, pandas as pd, librosa, joblib
sys.path.insert(0, os.path.dirname(__file__))
import g3_features as F
ROOT = r"G:\研究アーカイブ\成果物\動画解析"; MDIR = os.path.join(ROOT, "models", "g3_v1")
ap = argparse.ArgumentParser(); ap.add_argument("wavs", nargs="+"); ap.add_argument("--out", default=os.path.join(ROOT, "dataset", "blind_" + time.strftime("%Y%m%d")))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
th = json.load(open(os.path.join(MDIR, "thresholds.json"))); models = {t: joblib.load(os.path.join(MDIR, t + ".joblib")) for t in th}
summary = []
for wav in a.wavs:
    stem = os.path.splitext(os.path.basename(wav))[0]; t0 = time.time()
    y, sr = librosa.load(wav, sr=22050, mono=True)
    f0, vflag, vprob = librosa.pyin(y, fmin=70, fmax=800, sr=sr, frame_length=2048, hop_length=220)
    f0 = np.nan_to_num(f0, nan=0.0); conf = np.nan_to_num(vprob, nan=0.0); t = np.arange(len(f0)) * 220 / sr
    with open(os.path.join(a.out, stem + "_f0.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["audio_time_sec", "f0_hz", "conf", "voiced"])
        for i in range(len(f0)): w.writerow([f"{t[i]:.3f}", f"{f0[i]:.2f}", f"{conf[i]:.3f}", int(f0[i] > 0)])
    st, v = F.to_semitone(f0, conf); sg = F.segments(t, st, v)
    preds = []
    if sg:
        segs, sm = sg; cands = F.candidates(t, sm, segs)
        for typ, rows in cands.items():
            if typ not in models or not rows: continue
            X = np.array([r[1] for r in rows], dtype=float); p = models[typ].predict_proba(X)[:, 1]
            keep = [(rows[i][0], float(p[i]), rows[i][1]) for i in range(len(rows)) if p[i] >= th[typ]]
            keep.sort(); merged = []
            for anc, pr, fe in keep:  # 0.5s以内は最大確率の1件に
                if merged and anc - merged[-1][0] <= 0.5:
                    if pr > merged[-1][1]: merged[-1] = (anc, pr, fe)
                else: merged.append((anc, pr, fe))
            preds += [dict(type=typ, time_sec=round(anc, 3), prob=round(pr, 3), features=json.dumps([round(float(x), 3) for x in fe])) for anc, pr, fe in merged]
    preds.sort(key=lambda r: r["time_sec"])
    pd.DataFrame(preds, columns=["type", "time_sec", "prob", "features"]).to_csv(os.path.join(a.out, stem + "_pred.csv"), index=False, encoding="utf-8-sig")
    cnt = {typ: sum(1 for r in preds if r["type"] == typ) for typ in th}
    summary.append(dict(wav=stem, duration_sec=round(len(y) / sr, 1), **cnt, generated=time.strftime("%Y-%m-%d %H:%M:%S")))
    print(stem, cnt, f"{time.time()-t0:.0f}s", flush=True)
pd.DataFrame(summary).to_csv(os.path.join(a.out, "summary.csv"), index=False, encoding="utf-8-sig")
print("DONE", a.out)
