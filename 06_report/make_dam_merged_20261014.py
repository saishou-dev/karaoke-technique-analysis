# -*- coding: utf-8 -*-
r"""song_metrics.py の --dam に渡す「当日分だけの DAM CSV（src列つき）」を作る（2026-10-15）。
   理由＝song_metrics.py は machine 列を src（ai/hearts）から引くが、DAM API の生CSV に src 列が無い。
   既存の dam_all_merged_*.csv と同じ列・同じ src 表記に揃える（既存スクリプトは無変更）。
   使い方: python make_dam_merged_20261014.py --session 20261014 \
             --hearts G:\研究アーカイブ\成果物\dam_api\dam_ai-hearts_full_2026-10-14T16-22-34.csv \
             --out <path>.csv   （--ai を渡せば Ai 機種の行も足す）"""
import argparse, sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
COLS = ['scoringDateTime', 'songName', 'artistName', 'totalScore', 'radarChartPitch', 'radarChartStability',
        'radarChartExpressive', 'radarChartVibratoLongtone', 'radarChartRhythm', 'vibratoCount']
ap = argparse.ArgumentParser(); ap.add_argument("--session", required=True)
ap.add_argument("--hearts", default=""); ap.add_argument("--ai", default=""); ap.add_argument("--out", required=True)
a = ap.parse_args()
frames = []
for path, src in ((a.hearts, "hearts"), (a.ai, "ai")):
    if not path: continue
    d = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    x = d[d["scoringDateTime"].astype(str).str[:8] == a.session][COLS].copy(); x["src"] = src
    print(src, len(x), "行"); frames.append(x)
out = pd.concat(frames) if frames else pd.DataFrame(columns=COLS + ["src"])
out = out.sort_values("scoringDateTime")
out.to_csv(a.out, index=False, encoding="utf-8-sig")
print("DONE", a.out, len(out), "行")
