#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2桁採用フレームの箱の幾何（gap・w0・w1・重なり）を全曲で集計し、
   「真の2桁」と「1つのグリフが2つに割れた偽の2桁」を分離できる指標を探す。
   引数: <session> <results_dir> <bands_dir> [--videos a,b,c] [--step N]"""
import cv2,numpy as np,sys,os,json,csv,io
ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"
REPS=os.path.join(ROOT,"templates/wao_20260816/cluster_reps_wao_ext.npy")
LAB={int(k):v for k,v in json.load(open(os.path.splitext(REPS)[0]+"_labels.json")).items()}
THRESHOLDS=(215,200,185,228,170,150,130);NORM=(28,40);NCC_MIN=0.75
reps=np.load(REPS);rf=np.stack([ (r.ravel()-r.mean())/max(1e-6,np.linalg.norm(r.ravel()-r.mean())) for r in reps])
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
            x,y,w,h,a=stats[i]
            if 28<=h<=58 and 3<=w<=44 and a>=50: bs.append((x,y,w,h,a))
            elif a>30: junk+=a
        if not(1<=len(bs)<=2 and junk<=1200): continue
        bs.sort(key=lambda b:b[0])
        if len(bs)==2:
            gap=bs[1][0]-(bs[0][0]+bs[0][2]);hd=abs(bs[0][3]-bs[1][3]);yd=abs(bs[0][1]-bs[1][1])
            if gap>14 or gap<-4 or hd>8 or yd>6: continue
            if best2 is None: best2=bs
        elif best1 is None: best1=bs
    return best2 if best2 is not None else best1
session=sys.argv[1];step=int(sys.argv[2]) if len(sys.argv)>2 else 3
vids=sys.argv[3].split("|") if len(sys.argv)>3 and sys.argv[3] else None
RES=os.path.join(ROOT,"results_wao_"+session)
out=io.open(os.path.join(ROOT,"dataset","fall_diag_20261015","geom_%s.csv"%session),"w",encoding="utf-8",newline="")
w=csv.writer(out);w.writerow(["video","counter","frame","nbox","val","d0","d1","x0","w0","x1","w1","gap","hdiff","ydiff","a0","a1"])
names=vids if vids else sorted(d for d in os.listdir(RES) if os.path.isdir(os.path.join(RES,d)))
for name in names:
    rj=os.path.join(RES,name,"roi.json")
    band=os.path.join(ROOT,"bands_wao",name+".mp4")
    if not(os.path.exists(rj) and os.path.exists(band)): print("skip",name);continue
    roi=json.load(open(rj));cap=cv2.VideoCapture(band);fi=-1;n2=0
    while True:
        ok,img=cap.read()
        if not ok: break
        fi+=1
        if fi%step: continue
        g=img.max(axis=2)
        for c,(x1,y1,x2,y2) in roi.items():
            rg=g[y1:y2,x1:x2];bs=boxes_at(rg)
            if not bs or len(bs)!=2: continue
            ds=[classify(norm_glyph(rg,b)) for b in bs]
            if any(d is None for d in ds): continue
            gap=bs[1][0]-(bs[0][0]+bs[0][2])
            w.writerow([name,c,fi,2,ds[0]*10+ds[1],ds[0],ds[1],bs[0][0],bs[0][2],bs[1][0],bs[1][2],gap,
                        abs(bs[0][3]-bs[1][3]),abs(bs[0][1]-bs[1][1]),bs[0][4],bs[1][4]])
            n2+=1
    cap.release();print("%s 2box=%d"%(name,n2),flush=True)
out.close();print("WROTE")
