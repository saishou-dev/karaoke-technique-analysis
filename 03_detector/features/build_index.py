# -*- coding: utf-8 -*-
"""G3 v2: 曲ごとの index（wav・同期オフセット・イベント wav時刻・採点区間）を作る。wav_time = video_time + offset_sec"""
import os, sys, json, csv, collections
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
ROOT="G:/研究アーカイブ/成果物/動画解析"; REC="G:/研究アーカイブ/録音"; OUT="G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"
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
for video,ev in g1.groupby("video"):
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
json.dump(index,open(f"{OUT}/index.json","w",encoding="utf-8"),ensure_ascii=False,indent=0)
