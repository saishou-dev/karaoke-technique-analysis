#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""帯（カウンター帯）が画面に出ているかを「同時に数字が読めたカウンター数」で測る。
   引数: <video名> [<video名> ...]  出力: dataset/fall_diag_20261015/presence_<video>.csv"""
import cv2,numpy as np,sys,os,json,csv,io
ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"
REPS=os.path.join(ROOT,"templates/wao_20260816/cluster_reps_wao_ext.npy")
LAB={int(k):v for k,v in json.load(open(os.path.splitext(REPS)[0]+"_labels.json")).items()}
THRESHOLDS=(215,200,185,228,170,150,130);NORM=(28,40);NCC_MIN=0.75
reps=np.load(REPS);rf=np.stack([(r.ravel()-r.mean())/max(1e-6,np.linalg.norm(r.ravel()-r.mean())) for r in reps])
def classify(gl):
    v=gl.ravel()-gl.mean();n=np.linalg.norm(v)
    if n<1e-6: return None
    s=rf@(v/n);bi=int(np.argmax(s))
    return LAB.get(bi) if s[bi]>=NCC_MIN else None
def norm_glyph(rg,b):
    x,y,w,h=b[:4];p=2
    y0,y1=max(0,y-p),min(rg.shape[0],y+h+p);x0,x1=max(0,x-p),min(rg.shape[1],x+w+p)
    g=rg[y0:y1,x0:x1].astype(np.float32);g=(g-g.min())/max(1.0,(g.max()-g.min()))
    return cv2.resize(g,NORM)
def boxes_at(rg):
    best1=None;best2=None
    for T in THRESHOLDS:
        m=(rg>T).astype(np.uint8);n,_,stats,_=cv2.connectedComponentsWithStats(m,8)
        bs=[];junk=0
        for i in range(1,n):
            x,y,w,h,ar=stats[i]
            if 28<=h<=58 and 3<=w<=44 and ar>=50: bs.append((x,y,w,h,ar))
            elif ar>30: junk+=ar
        if not(1<=len(bs)<=2 and junk<=1200): continue
        bs.sort(key=lambda b:b[0])
        if len(bs)==2:
            gap=bs[1][0]-(bs[0][0]+bs[0][2])
            if gap>14 or gap<-4 or abs(bs[0][3]-bs[1][3])>8 or abs(bs[0][1]-bs[1][1])>6: continue
            if best2 is None: best2=bs
        elif best1 is None: best1=bs
    return best2 if best2 is not None else best1
OUT=os.path.join(ROOT,"dataset","fall_diag_20261015")
for name in sys.argv[1:]:
    roi=json.load(open(os.path.join(ROOT,"results_wao_20261014",name,"roi.json")))
    cap=cv2.VideoCapture(os.path.join(ROOT,"bands_wao",name+".mp4"));fi=-1
    w=csv.writer(io.open(os.path.join(OUT,"presence_%s.csv"%name),"w",encoding="utf-8",newline=""))
    w.writerow(["frame","n_ok","vals"])
    while True:
        ok,img=cap.read()
        if not ok: break
        fi+=1
        g=img.max(axis=2);n_ok=0;vals=[]
        for c,(x1,y1,x2,y2) in roi.items():
            bs=boxes_at(g[y1:y2,x1:x2])
            if not bs: vals.append("");continue
            ds=[classify(norm_glyph(g[y1:y2,x1:x2],b)) for b in bs]
            if any(d is None for d in ds): vals.append("");continue
            n_ok+=1;vals.append(str(ds[0] if len(ds)==1 else ds[0]*10+ds[1]))
        w.writerow([fi,n_ok,"|".join(vals)])
    cap.release();print("done",name,flush=True)
print("WROTE")
