# -*- coding: utf-8 -*-
"""G3 v2 特徴量：log-mel(80) + RMS + pYIN(f0半音・有声) を 10ms グリッドで npz に保存。16並列。"""
import os, sys, json, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings("ignore")
ROOT="G:/研究アーカイブ/成果物/動画解析"; OUT=f"{ROOT}/g3v2_20260928"
SR=22050; HOP=220; NMEL=80
def pitch_for(d, synced_map):
    p=d["pitch"]
    if p is None:
        pf=synced_map.get(d["video"])
        if pf and os.path.exists(f"{ROOT}/dataset/pitch_pyin/{pf}"): p=f"{ROOT}/dataset/pitch_pyin/{pf}"
    return p
def work(args):
    d,pf=args
    import librosa, soundfile as sf
    out=f"{OUT}/feats/{d['video']}.npz"
    if os.path.exists(out): return d["video"],"skip"
    y,sr=sf.read(d["wav"],dtype="float32",always_2d=True); y=y.mean(1)
    if sr!=SR: y=librosa.resample(y,orig_sr=sr,target_sr=SR)
    S=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=1024,hop_length=HOP,n_mels=NMEL,fmin=50,fmax=8000,power=2.0)
    logmel=np.log(S+1e-6).astype(np.float32)
    rms=librosa.feature.rms(y=y,frame_length=1024,hop_length=HOP)[0]; rms=np.log(rms+1e-6).astype(np.float32)
    T=logmel.shape[1]; t=np.arange(T)*HOP/SR
    if pf and os.path.exists(pf):
        p=pd.read_csv(pf,encoding="utf-8-sig"); pt=p["audio_time_sec"].values.astype(float); f0=p["f0_hz"].values.astype(float); conf=p["conf"].values.astype(float)
        src="csv"
    else:
        f0,vf,vp=librosa.pyin(y,fmin=70,fmax=800,sr=SR,frame_length=2048,hop_length=HOP); f0=np.nan_to_num(f0,nan=0.0); conf=np.nan_to_num(vp,nan=0.0); pt=np.arange(len(f0))*HOP/SR; src="pyin_now"
    st=np.where(f0>40,69+12*np.log2(np.maximum(f0,1)/440.0),np.nan)
    # 10ms グリッドへ最近傍
    idx=np.clip(np.searchsorted(pt,t),0,len(pt)-1)
    st_g=st[idx]; voiced=(~np.isnan(st_g)).astype(np.float32); conf_g=conf[idx].astype(np.float32)
    st_f=np.nan_to_num(st_g,nan=0.0).astype(np.float32)
    np.savez_compressed(out,logmel=logmel.astype(np.float16),rms=rms,st=st_f,voiced=voiced,conf=conf_g,t0=0.0,hop=HOP/SR,src=src)
    return d["video"],f"ok T={T} pitch={src}"
if __name__=="__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    index=json.load(open(f"{OUT}/index.json",encoding="utf-8"))
    sy=pd.read_csv(f"{ROOT}/dataset/G1_events_synced.csv",encoding="utf-8-sig"); smap=dict(zip(sy["video"],sy["pitch_file"]))
    jobs=[(d,pitch_for(d,smap)) for d in index]
    print("no pitch file ->",[d["video"] for d,pf in jobs if not pf],flush=True)
    # 並列数は環境変数 KV_FEAT_WORKERS で変えられる（既定 14＝従来どおり）。2026-10-20 追加＝
    # 外付け HDD を読む工程は 3〜4 並列が上限という運用（作業ログ 10/20 の摩擦1件）に合わせるため。
    nw = int(os.environ.get("KV_FEAT_WORKERS", "14"))
    print("workers", nw, flush=True)
    with ProcessPoolExecutor(nw) as ex:
        for v,msg in ex.map(work,jobs): print(v,msg,flush=True)
    print("DONE",flush=True)
