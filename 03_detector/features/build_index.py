# -*- coding: utf-8 -*-
"""G3 v2: 曲ごとの index（wav・同期オフセット・イベント wav時刻・採点区間）を作る。wav_time = video_time + offset_sec

2026-10-20 改訂（10/20 午前の報告 ③摩擦1件＝本人○）：
  これまで本スクリプトが読む同期表は `sync_offsets.csv`・`sync_offsets_20260911.csv`・`sync_offsets_20260924.csv` の
  3 本だけで、10/6 の 29 曲が索引から落ちていた。そのため流すたびに `add_session_20261006.py` を足して
  流し直す必要があった（10/15・10/20 の 2 回とも実際にそうした）。本改訂で **10/6・10/14 の同期表も
  このスクリプトが読む**ようにして 1 本にまとめた。

  出力の互換性＝既定（--extra を付けない）では **従来の 2 本流し（build_index.py → add_session_20261006.py）と
  同じ index.json** を出す（md5 一致を 2026-10-20 に実測）。そのために 10/6 は従来どおり
  「基本セッションを全部並べた後ろに追加する」順序・規則（audio_time_sec をそのまま使う・machine 列を付ける）で
  書いている。
  10/14 は**既定では入れない**。索引に入れる曲を増やすのは評価の入力そのものを変えること（型を変える改善）で、
  旧挙動と md5 が変わる。入れるときは `--extra 20261006,20261014` と明示する＝本人○待ち。

使い方:
  python build_index.py                              # 従来どおり（基本セッション ＋ 10/6）
  python build_index.py --extra 20261006,20261014    # 10/14 も入れる（index.json の md5 は変わる）
  python build_index.py --extra ""                   # 基本セッションだけ（2026-10-20 改訂前の build_index.py 相当）
  python build_index.py --out <path>                 # 書き出し先を変える（既定 g3v2_20260928/index.json）
"""
import os, sys, json, csv, collections, argparse
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
ROOT="G:/研究アーカイブ/成果物/動画解析"; REC="G:/研究アーカイブ/録音"; OUT="G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"

# ---- 後ろに足すセッション（従来 add_session_20261006.py がやっていたもの）----
# audio_time_sec（G1 に既にある録音WAV時刻）をそのまま使い、machine 列を付ける。
# machine＝mapping.csv の machine= と同じ規則（T01〜T09＝精密採点Ai・T10〜＝WAO）。10/14 は全曲 WAO。
EXTRA = {
    "20261006": dict(pitch="pitch_pyin_20261006", machine="take"),
    "20261014": dict(pitch="pitch_pyin_20261014", machine="wao"),
}
ap = argparse.ArgumentParser()
ap.add_argument("--extra", default="20261006", help="後ろに足すセッション（カンマ区切り・既定 20261006＝従来の 2 本流しと同じ出力）")
ap.add_argument("--out", default=f"{OUT}/index.json")
a = ap.parse_args()
EXTRA_SES = [s for s in a.extra.split(",") if s.strip()]
for s in EXTRA_SES:
    assert s in EXTRA, f"未知のセッション: {s}（{sorted(EXTRA)} のいずれか）"

g1=pd.read_csv(f"{ROOT}/dataset/G1_events_all.csv",encoding="utf-8-sig",dtype={"session":str})
g1=g1[g1["multi_increment"]==0]
# sync tables
sync={}
s0=pd.read_csv(f"{ROOT}/dataset/sync/sync_offsets.csv",encoding="utf-8-sig",dtype={"session":str})
for _,r in s0.iterrows():
    if isinstance(r["wav_file"],str) and r["wav_file"]:
        sync[r["video_id"]]=(f"{REC}/{r['wav_file']}",float(r["offset_sec"]),r["session"])
for ses in ["20260911","20260924"]:
    d=pd.read_csv(f"{ROOT}/dataset/sync/sync_offsets_{ses}.csv",encoding="utf-8-sig")
    for _,r in d.iterrows():
        sync[r["video"]]=(f"{REC}/songs_{ses}_raw/{r['wav']}",float(r["offset_sec"]),ses)
# pitch files
pdirs=[f"{ROOT}/dataset/pitch_pyin",f"{ROOT}/dataset/pitch_pyin_20260911",f"{ROOT}/dataset/pitch_pyin_20260924"]
def find_pitch(video,wav):
    stem=os.path.splitext(os.path.basename(wav))[0]
    for d in pdirs:
        for n in (video+"_f0.csv",stem+"_f0.csv"):
            p=f"{d}/{n}"
            if os.path.exists(p): return p
    return None
index=[]; miss=[]
base=g1[~g1["session"].isin(EXTRA_SES)] if EXTRA_SES else g1
for video,ev in base.groupby("video"):
    if video not in sync: miss.append((video,ev["session"].iloc[0],len(ev))); continue
    wav,off,ses=sync[video]
    if not os.path.exists(wav): miss.append((video,ses,f"wav missing {wav}")); continue
    pf=find_pitch(video,wav)
    events=[(str(r.counter),float(r.video_time_sec)+off) for r in ev.itertuples()]
    ts=[t for _,t in events]
    index.append(dict(video=video,session=ses,wav=wav,offset=off,pitch=pf,events=events,span=[min(ts)-5.0,max(ts)+2.0],n=len(events)))
print("videos",len(index),"events",sum(d["n"] for d in index)); print("missing",len(miss));
for m in miss: print("  ",m)
print("no pitch:",[d["video"] for d in index if d["pitch"] is None])
# sanity: voiced rate at event time (offset sign check)
bys=collections.defaultdict(list)
for d in index:
    if not d["pitch"]: continue
    p=pd.read_csv(d["pitch"],encoding="utf-8-sig"); t=p["audio_time_sec"].values; v=(p["f0_hz"].values>40)
    for c,tt in d["events"]:
        if c in("shakuri","kobushi","vibrato"):
            i=np.searchsorted(t,tt-0.4); j=np.searchsorted(t,tt-0.1)
            if j>i: bys[d["session"]].append(v[i:j].mean()>0.5)
for s in sorted(bys): print("voiced@event-0.25s",s,round(float(np.mean(bys[s])),3),len(bys[s]))

# ---- 後ろに足すセッション（旧 add_session_20261006.py と同じ規則・同じ順序）----
for SES in EXTRA_SES:
    cfg = EXTRA[SES]
    gs = g1[g1["session"] == SES]
    sync2 = pd.read_csv(f"{ROOT}/dataset/sync/sync_offsets_{SES}.csv", encoding="utf-8-sig")
    smap = {r["video"]: (f"{REC}/songs_{SES}_raw/{r['wav']}", float(r["offset_sec"])) for _, r in sync2.iterrows()}
    new = []; warn = []
    for video, ev in gs.groupby("video"):
        if video not in smap: warn.append(f"sync なし: {video}"); continue
        wav, off = smap[video]
        if not os.path.exists(wav): warn.append(f"wav なし: {wav}"); continue
        stem = os.path.splitext(os.path.basename(wav))[0]; pf = f"{ROOT}/dataset/{cfg['pitch']}/{stem}_f0.csv"
        if not os.path.exists(pf): warn.append(f"pitch なし: {pf}"); pf = None
        dev = (ev["video_time_sec"] + off - ev["audio_time_sec"]).abs().max()
        if dev > 0.002: warn.append(f"audio_time_sec と video_time+offset の差 {dev:.3f}s: {video}")
        events = [(str(r.counter), float(r.audio_time_sec)) for r in ev.itertuples()]
        ts = [t for _, t in events]
        import soundfile as sf
        dur = sf.info(wav).duration
        n_out = sum(1 for t in ts if t < 0 or t > dur)
        if n_out: warn.append(f"WAV の範囲外（0〜{dur:.1f}s）のイベント {n_out}件: {video}（録音が曲の途中から／手前で終了）")
        machine = ("ai" if int(stem[1:3]) <= 9 else "wao") if cfg["machine"] == "take" else cfg["machine"]
        new.append(dict(video=video, session=SES, wav=wav, offset=off, pitch=pf, events=events, span=[min(ts) - 5.0, max(ts) + 2.0], n=len(events), machine=machine))
    print(f"[追加] {SES}: {len(gs)} 行 → {len(new)} 曲 {sum(d['n'] for d in new)} 件")
    for w in warn: print("  注意:", w)
    index = index + new

print("index 合計", len(index), "エントリ", sum(d["n"] for d in index), "件")
json.dump(index,open(a.out,"w",encoding="utf-8"),ensure_ascii=False,indent=0)
