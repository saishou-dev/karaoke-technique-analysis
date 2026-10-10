# -*- coding: utf-8 -*-
"""WAO帯の位置・スケール・傾き推定 v1（2026-09-12・WSL cv2）。基準帯＝templates/wao_20260911/ref_band_from_0812.png（2480x120・正規化済み）
   各動画で数フレーム（曲中盤）を取り、多段スケールのテンプレート照合で帯左上(x,y)とscaleを推定→左右端の部分照合でdyL/dyRから傾き角を推定。
   出力: dataset/band_probe_20260911.csv（video,xL,yL,w,h,scale,angle_deg,score,n_frames,note）
   使い方: python wao_probe_20260911.py <video1> [<video2> ...]  (パスは/mnt/g/...)
"""
import sys, os, csv, math, numpy as np, cv2
ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"
REF=cv2.imread(os.path.join(ROOT,"templates/wao_20260911/ref_band_from_0812.png"),cv2.IMREAD_GRAYSCALE)
RW,RH=REF.shape[1],REF.shape[0]
def prep(g): 
    g=cv2.GaussianBlur(g,(3,3),0); return cv2.Laplacian(g,cv2.CV_32F,ksize=3)
def probe_frame(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); H,W=g.shape
    y0,y1=int(H*0.20),int(H*0.55)   # 帯は画面上部〜中央
    sub=g[y0:y1]; d=0.5; subs=cv2.resize(sub,None,fx=d,fy=d); P=prep(subs)
    best=None
    for s in np.arange(0.78,1.08,0.01):
        tw,th=int(RW*s*d),int(RH*s*d)
        if tw>=subs.shape[1] or th>=subs.shape[0]: continue
        T=prep(cv2.resize(REF,(tw,th)))
        r=cv2.matchTemplate(P,T,cv2.TM_CCOEFF_NORMED); _,mx,_,loc=cv2.minMaxLoc(r)
        if best is None or mx>best[0]: best=(mx,loc[0]/d,loc[1]/d+y0,s)
    sc,x,y,s=best; w,h=RW*s,RH*s
    # 傾き: 左右300pxのサブ帯を±40pxの縦窓で照合
    def part(x0ref,x1ref):
        T=prep(cv2.resize(REF[:,x0ref:x1ref],(int((x1ref-x0ref)*s),int(RH*s))))
        px=int(x+x0ref*s); py=int(y); m=40
        reg=prep(g[max(0,py-m):py+int(RH*s)+m, max(0,px-30):px+T.shape[1]+30].astype(np.uint8) if False else g[max(0,py-m):py+int(RH*s)+m, max(0,px-30):px+T.shape[1]+30])
        if reg.shape[0]<T.shape[0] or reg.shape[1]<T.shape[1]: return None
        r=cv2.matchTemplate(reg,T,cv2.TM_CCOEFF_NORMED); _,mx,_,loc=cv2.minMaxLoc(r)
        return (loc[1]-m+ (0 if py-m>=0 else (m-py)) , mx, loc[0]-30+px+ (0 if px-30>=0 else (30-px)))
    L=part(0,300); R=part(RW-300,RW)
    ang=None; dyL=dyR=None
    if L and R:
        dyL,dyR=L[0],R[0]; xl=L[2]; xr=R[2]+300*s
        ang=math.degrees(math.atan2(dyR-dyL, xr-xl))
    return dict(score=sc,x=x,y=y,scale=s,w=w,h=h,angle=ang,dyL=dyL,dyR=dyR,scL=L[1] if L else None,scR=R[1] if R else None)
out=[]
for v in sys.argv[1:]:
    cap=cv2.VideoCapture(v); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS) or 59.94
    res=[]
    for frac in (0.25,0.45,0.65,0.85):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(n*frac)); ok,img=cap.read()
        if ok: res.append(probe_frame(img))
    cap.release()
    good=[r for r in res if r["score"]>0.15]
    if not good: good=res
    med=lambda k: float(np.median([r[k] for r in good if r[k] is not None])) if any(r[k] is not None for r in good) else float("nan")
    row=dict(video=os.path.splitext(os.path.basename(v))[0],xL=round(med("x")),yL=round(med("y")),w=round(med("w")),h=round(med("h")),scale=round(med("scale"),4),angle_deg=round(med("angle"),3),score=round(med("score"),3),n_frames=len(good),
             note="frames:"+";".join(f"{r['score']:.2f}/{r['x']:.0f},{r['y']:.0f}/s{r['scale']:.2f}/a{(r['angle'] if r['angle'] is not None else float('nan')):.2f}" for r in res))
    out.append(row); print(row,flush=True)
p=os.path.join(ROOT,"dataset","band_probe_20260911.csv"); ex=os.path.exists(p)
with open(p,"a",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0].keys()));
    if not ex: w.writeheader()
    w.writerows(out)
print("WROTE",p)
