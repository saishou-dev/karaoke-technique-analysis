#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fall過剰読みの診断：帯動画の指定カウンターROIを1フレームずつ読み、
   安定ラン（>=STABLE_MIN）とWLIS採用パスを全部印字する。読取器 v0.8d と同一のロジック。"""
import cv2, numpy as np, sys, os, json
VIDEO=sys.argv[1]; REPS=sys.argv[2]; ROIJ=sys.argv[3]; TARGETS=sys.argv[4].split(",")
_lab=os.path.splitext(REPS)[0]+"_labels.json"
LABELS={int(k):v for k,v in json.load(open(_lab)).items()}
THRESHOLDS=(215,200,185,228,170,150,130); NORM=(28,40); NCC_MIN=0.75; STABLE_MIN=6
ROI=json.load(open(ROIJ))
def glyph_boxes(rg):
    best1=None; best2=None
    for T in THRESHOLDS:
        m=(rg>T).astype(np.uint8); n,lab,stats,_=cv2.connectedComponentsWithStats(m,8)
        boxes=[]; junk=0
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            if 28<=h<=58 and 3<=w<=44 and area>=50: boxes.append((x,y,w,h))
            elif area>30: junk+=area
        if not (1<=len(boxes)<=2 and junk<=1200): continue
        boxes.sort(key=lambda b:b[0])
        if len(boxes)==2:
            gap=boxes[1][0]-(boxes[0][0]+boxes[0][2]); hd=abs(boxes[0][3]-boxes[1][3]); yd=abs(boxes[0][1]-boxes[1][1])
            if gap>14 or gap<-4 or hd>8 or yd>6: continue
            if best2 is None: best2=boxes
        elif best1 is None: best1=boxes
    return best2 if best2 is not None else best1
def norm_glyph(rg,box):
    x,y,w,h=box; p=2
    y0,y1=max(0,y-p),min(rg.shape[0],y+h+p); x0,x1=max(0,x-p),min(rg.shape[1],x+w+p)
    g=rg[y0:y1,x0:x1].astype(np.float32); g=(g-g.min())/max(1.0,(g.max()-g.min()))
    return cv2.resize(g,NORM)
reps=np.load(REPS); rf=[]
for r in reps:
    v=r.ravel()-r.mean(); rf.append(v/max(1e-6,np.linalg.norm(v)))
rf=np.stack(rf)
def classify(gl):
    v=gl.ravel()-gl.mean(); n=np.linalg.norm(v)
    if n<1e-6: return None,0.0
    s=rf@(v/n); bi=int(np.argmax(s))
    if s[bi]<NCC_MIN: return None,float(s[bi])
    return LABELS.get(bi),float(s[bi])
cap=cv2.VideoCapture(VIDEO); fps=cap.get(cv2.CAP_PROP_FPS); nf=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("video=%s fps=%.3f frames=%d"%(os.path.basename(VIDEO),fps,nf))
vals={k:[] for k in TARGETS}; nbox={k:{1:0,2:0} for k in TARGETS}
fi=-1
while True:
    ok,img=cap.read()
    if not ok: break
    fi+=1
    g=img.max(axis=2)
    for name in TARGETS:
        x1,y1,x2,y2=ROI[name]; rg=g[y1:y2,x1:x2]
        bx=glyph_boxes(rg)
        if bx is None: continue
        ds=[]; sc=[]
        for b in bx:
            d,s=classify(norm_glyph(rg,b))
            if d is None: ds=None; break
            ds.append(d); sc.append(s)
        if ds is None: continue
        nbox[name][len(bx)]+=1
        val=ds[0] if len(ds)==1 else ds[0]*10+ds[1]
        vals[name].append((fi,val,len(bx),round(min(sc),3)))
cap.release()
for name in TARGETS:
    seq=vals[name]
    print("\n== %s : readable %d/%d (%.1f%%)  1箱=%d 2箱=%d"%(name,len(seq),nf,100*len(seq)/nf,nbox[name][1],nbox[name][2]))
    runs=[]; rv=None
    for fr,v,nb,s in seq:
        if rv is not None and v==rv: runs[-1][2]+=1; runs[-1][4]=min(runs[-1][4],s); runs[-1][5].add(nb)
        else: runs.append([fr,v,1,fr,s,{nb}]); rv=v
    kept=[r for r in runs if r[2]>=STABLE_MIN]
    print("  全ラン=%d  安定ラン(>=6)=%d"%(len(runs),len(kept)))
    print("  | # | 開始frame | 時刻s | 値 | 滞在frames | 箱数 | NCC最小 |")
    for i,r in enumerate(kept):
        print("  | %d | %d | %.2f | %d | %d | %s | %.3f |"%(i,r[0],r[0]/fps,r[1],r[2],",".join(map(str,sorted(r[5]))),r[4]))
    n=len(kept)
    if n:
        dp=[0.0]*n; prev=[-1]*n
        for i in range(n):
            dp[i]=kept[i][2]
            for j in range(i):
                if kept[j][1]<=kept[i][1] and dp[j]+kept[i][2]>dp[i]: dp[i]=dp[j]+kept[i][2]; prev[i]=j
        b=int(np.argmax(dp)); path=[]
        while b!=-1: path.append(b); b=prev[b]
        path.reverse()
        print("  WLIS採用=%s  (重み=%.0f)"%([ (kept[i][0],kept[i][1],kept[i][2]) for i in path ], max(dp)))
        print("  不採用ラン=%s"%[ (kept[i][0],kept[i][1],kept[i][2]) for i in range(n) if i not in path ])
