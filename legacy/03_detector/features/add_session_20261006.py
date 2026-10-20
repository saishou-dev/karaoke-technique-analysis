# -*- coding: utf-8 -*-
"""10/6セッション（29曲）を index.json に追加する（2026-10-08・E1）。build_index.py と同じ規則＝
   multi_increment==0 の行だけ・時刻＝録音WAV時刻（G1 の audio_time_sec＝video_time_sec+offset）・span＝[最初−5s, 最後＋2s]。
   machine 列（ai／wao）を 10/6 のエントリだけに付ける。既存149エントリは触らない（追加前に index_pre20261006.bak.json へ退避）。pitch＝dataset/pitch_pyin_20261006/<wav stem>_f0.csv。
   使い方: python add_session_20261006.py [--dry]"""
import os, sys, json, shutil, argparse
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = "G:/研究アーカイブ/成果物/動画解析"; REC = "G:/研究アーカイブ/録音"; OUT = f"{ROOT}/g3v2_20260928"; SES = "20261006"
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
index = json.load(open(f"{OUT}/index.json", encoding="utf-8"))
have = {d["video"] for d in index}
assert not any(d["session"] == SES for d in index), "10/6 は既に index にあります"
g1 = pd.read_csv(f"{ROOT}/dataset/G1_events_all.csv", encoding="utf-8-sig", dtype={"session": str})
g1 = g1[(g1["session"] == SES)]
n_all = len(g1); g1 = g1[g1["multi_increment"] == 0]
sync = pd.read_csv(f"{ROOT}/dataset/sync/sync_offsets_{SES}.csv", encoding="utf-8-sig")
smap = {r["video"]: (f"{REC}/songs_{SES}_raw/{r['wav']}", float(r["offset_sec"])) for _, r in sync.iterrows()}
new = []; warn = []
for video, ev in g1.groupby("video"):
    if video not in smap: warn.append(f"sync なし: {video}"); continue
    wav, off = smap[video]
    if not os.path.exists(wav): warn.append(f"wav なし: {wav}"); continue
    stem = os.path.splitext(os.path.basename(wav))[0]; pf = f"{ROOT}/dataset/pitch_pyin_{SES}/{stem}_f0.csv"
    if not os.path.exists(pf): warn.append(f"pitch なし: {pf}"); pf = None
    # audio_time_sec（G1 に既にある録音WAV時刻）を使う。video_time+offset と一致することを確認
    dev = (ev["video_time_sec"] + off - ev["audio_time_sec"]).abs().max()
    if dev > 0.002: warn.append(f"audio_time_sec と video_time+offset の差 {dev:.3f}s: {video}")
    events = [(str(r.counter), float(r.audio_time_sec)) for r in ev.itertuples()]
    ts = [t for _, t in events]
    import soundfile as sf
    dur = sf.info(wav).duration
    n_out = sum(1 for t in ts if t < 0 or t > dur)
    if n_out: warn.append(f"WAV の範囲外（0〜{dur:.1f}s）のイベント {n_out}件: {video}（録音が曲の途中から／手前で終了）")
    machine = "ai" if int(stem[1:3]) <= 9 else "wao"  # mapping.csv の machine= と同じ（T01〜T09＝精密採点Ai・T10〜＝WAO）
    new.append(dict(video=video, session=SES, wav=wav, offset=off, pitch=pf, events=events, span=[min(ts) - 5.0, max(ts) + 2.0], n=len(events), machine=machine))
print(f"G1 {SES}: 全行 {n_all} → multi_increment==0 {len(g1)} 行 → {len(new)} 曲 {sum(d['n'] for d in new)} 件")
for w in warn: print("  注意:", w)
if a.dry: sys.exit(0)
bak = f"{OUT}/index_pre20261006.bak.json"
if not os.path.exists(bak): shutil.copy2(f"{OUT}/index.json", bak); print("退避:", bak)
json.dump(index + new, open(f"{OUT}/index.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(f"index.json: {len(index)} → {len(index) + len(new)} エントリ")
