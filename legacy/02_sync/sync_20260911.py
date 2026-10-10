# -*- coding: utf-8 -*-
"""録画と録音の同期 v1（2026-09-12・Windows python・librosa）。方式＝9/2のD23を再現：8kHzモノ化→RMS包絡(10ms)の全域正規化相互相関（主）＋オンセット強度の相互相関（検証）→±0.05s一致で確定、不一致は声帯域(300-3000Hz)RMSで裁定。
   対応付け＝rename_mapping.csv（動画）× mapping.csv（録音）を played_at（採点時刻）で結合。
   出力: dataset/sync/sync_offsets_20260911.csv（video, wav, offset_sec=wav_time-video_time, method, r_main, r_onset, agree, confidence）
"""
import os, csv, subprocess, numpy as np, librosa, datetime as dt
ROOT=r"G:\研究アーカイブ\成果物\動画解析"; VID=r"G:\研究アーカイブ\ビデオ素材\20260911"; WAV=r"G:\研究アーカイブ\録音\songs_20260911_raw"
FF=os.path.join(os.environ["LOCALAPPDATA"],"Microsoft","WinGet","Packages","Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe","ffmpeg-9.0.1-full_build","bin","ffmpeg.exe")
SR=8000; HOP=80  # 10ms
def env(y):
    r=librosa.feature.rms(y=y,frame_length=400,hop_length=HOP)[0]; r=(r-r.mean())/(r.std()+1e-9); return r
def onset(y):
    o=librosa.onset.onset_strength(y=y,sr=SR,hop_length=HOP,fmin=200,fmax=3400); o=(o-o.mean())/(o.std()+1e-9); return o
def xcorr(a,b):
    n=len(a)+len(b); N=1<<(n-1).bit_length(); A=np.fft.rfft(a,N); B=np.fft.rfft(b,N); c=np.fft.irfft(A*np.conj(B),N)
    c=np.concatenate([c[-(len(b)-1):],c[:len(a)]])  # lags from -(len(b)-1) .. len(a)-1
    lags=np.arange(-(len(b)-1),len(a)); k=int(np.argmax(c)); 
    # 正規化相関（ラグ位置で重なり長で割る簡易版）
    ov=min(len(a),len(b)); r=c[k]/(ov)
    srt=np.sort(c)[::-1]; ratio=srt[1]/srt[0] if srt[0]>0 else 1
    return lags[k]*HOP/SR, float(r), float(ratio)
vids={}
for r in csv.DictReader(open(os.path.join(VID,"rename_mapping.csv"),encoding="utf-8-sig")):
    if r["played_at"]: vids[r["played_at"]]=r["renamed"]
wavs={}
for r in csv.DictReader(open(os.path.join(WAV,"mapping.csv"),encoding="utf-8-sig")):
    if r.get("played_at"): wavs.setdefault(r["played_at"],[]).append(r["renamed"])
os.makedirs(os.path.join(ROOT,"dataset","sync"),exist_ok=True); tmp=os.path.join(ROOT,"dataset","sync","tmp_20260911"); os.makedirs(tmp,exist_ok=True)
rows=[]
for pa,vname in sorted(vids.items()):
    for wname in wavs.get(pa,[]):
        va=os.path.join(tmp,os.path.splitext(vname)[0]+"_8k.wav")
        if not os.path.exists(va): subprocess.run([FF,"-y","-v","error","-i",os.path.join(VID,vname),"-vn","-ac","1","-ar",str(SR),va],check=True)
        yv,_=librosa.load(va,sr=SR,mono=True); yw,_=librosa.load(os.path.join(WAV,wname),sr=SR,mono=True)
        # video_time = wav_time + lag  →  offset(wav_time - video_time) = -lag  where lag = argmax xcorr(env_v, env_w)
        lag1,r1,ratio1=xcorr(env(yv),env(yw)); lag2,r2,ratio2=xcorr(onset(yv),onset(yw))
        agree=abs(lag1-lag2)<=0.05; method="rms+onset"
        if not agree:
            yb_v=librosa.effects.preemphasis(yv); yb_w=librosa.effects.preemphasis(yw)
            import scipy.signal as ss
            b,a=ss.butter(4,[300/(SR/2),3000/(SR/2)],btype="band"); lag3,r3,_=xcorr(env(ss.lfilter(b,a,yv)),env(ss.lfilter(b,a,yw)))
            method="band-arbitrated"; lag=lag3 if abs(lag3-lag1)<=0.05 or abs(lag3-lag2)>0.05 else lag2
            if abs(lag3-lag1)>0.05 and abs(lag3-lag2)<=0.05: lag=lag2
        else: lag=lag1
        rows.append(dict(video=os.path.splitext(vname)[0],wav=wname,offset_sec=round(-lag,3),method=method,r_main=round(r1,3),r_onset=round(r2,3),lag_main=round(lag1,3),lag_onset=round(lag2,3),agree=agree,peak_ratio=round(ratio1,3),played_at=pa))
        print(rows[-1],flush=True)
p=os.path.join(ROOT,"dataset","sync","sync_offsets_20260911.csv")
with open(p,"w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE",p,len(rows))
