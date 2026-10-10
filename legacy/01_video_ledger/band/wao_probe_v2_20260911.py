# -*- coding: utf-8 -*-
"""WAO帯プローブ v2（2026-09-12）：v1の粗探索（多段スケール全帯照合）の後、左端400px・右端400pxの部分帯をそれぞれ照合して
   実スパンから scale を再推定（右側ROIのずれ対策）＋ dyL/dyR から傾き。出力: dataset/band_probe_v2_20260911.csv"""
import sys, os, csv, math, numpy as np, cv2
ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"
REF=cv2.imread(os.path.join(ROOT,"templates/wao_20260911/ref_band_from_0812.png"),cv2.IMREAD_GRAYSCALE); RW,RH=REF.shape[1],REF.shape[0]
PW=400
def prep(g): return cv2.Laplacian(cv2.GaussianBlur(g,(3,3),0),cv2.CV_32F,ksize=3)
def coarse(g):
    H,W=g.shape; y0,y1=int(H*0.20),int(H*0.55); d=0.5; subs=cv2.resize(g[y0:y1],None,fx=d,fy=d); P=prep(subs); best=None
    for s in np.arange(0.78,1.08,0.01):
        tw,th=int(RW*s*d),int(RH*s*d)
        if tw>=subs.shape[1] or th>=subs.shape[0]: continue
        r=cv2.matchTemplate(P,prep(cv2.resize(REF,(tw,th))),cv2.TM_CCOEFF_NORMED); _,mx,_,loc=cv2.minMaxLoc(r)
        if best is None or mx>best[0]: best=(mx,loc[0]/d,loc[1]/d+y0,s)
    return best
def part(g,x,y,s,x0ref,x1ref,mx=60,my=40):
    best=None
    for ds in (-0.03,-0.015,0.0,0.015,0.03):
        ss=s+ds; T=prep(cv2.resize(REF[:,x0ref:x1ref],(int((x1ref-x0ref)*ss),int(RH*ss))))
        px=int(x+x0ref*s); py=int(y); X0=max(0,px-mx); Y0=max(0,py-my)
        reg=prep(g[Y0:py+int(RH*ss)+my, X0:px+T.shape[1]+mx])
        if reg.shape[0]<T.shape[0] or reg.shape[1]<T.shape[1]: continue
        r=cv2.matchTemplate(reg,T,cv2.TM_CCOEFF_NORMED); _,m,_,loc=cv2.minMaxLoc(r)
        if best is None or m>best[0]: best=(m,X0+loc[0],Y0+loc[1],ss)
    return best
def probe(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); sc,x,y,s=coarse(g)
    L=part(g,x,y,s,0,PW); R=part(g,x,y,s,RW-PW,RW)
    if not L or not R: return dict(score=sc,x=x,y=y,scale=s,w=RW*s,h=RH*s,angle=0.0,scL=None,scR=None,refined=0)
    xl,yl,sl=L[1],L[2],L[3]; xr,yr,sr=R[1],R[2],R[3]
    span=(xr+PW*sr)-xl; scale=span/RW; ang=math.degrees(math.atan2(yr-yl,(xr+PW*sr/2)-(xl+PW*sl/2)))
    return dict(score=sc,x=xl,y=yl,scale=scale,w=RW*scale,h=RH*scale,angle=ang,scL=L[0],scR=R[0],refined=1)
out=[]
for v in sys.argv[1:]:
    cap=cv2.VideoCapture(v); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); res=[]
    for frac in (0.25,0.45,0.65,0.85):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(n*frac)); ok,img=cap.read()
        if ok: res.append(probe(img))
    cap.release(); good=[r for r in res if r["refined"] and r["scL"] is not None and r["scL"]>0.1 and r["scR"]>0.1] or res
    med=lambda k: float(np.median([r[k] for r in good]))
    row=dict(video=os.path.splitext(os.path.basename(v))[0],xL=round(med("x")),yL=round(med("y")),w=round(med("w")),h=round(med("h")),scale=round(med("scale"),4),angle_deg=round(med("angle"),3),score=round(med("score"),3),n_frames=len(good),
             note=";".join(f"L{(r['scL'] or 0):.2f}/R{(r['scR'] or 0):.2f}/s{r['scale']:.3f}/a{r['angle']:.2f}" for r in res))
    out.append(row); print(row,flush=True)
p=os.path.join(ROOT,"dataset","band_probe_v2_20260911.csv"); ex=os.path.exists(p)
with open(p,"a",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0].keys()))
    if not ex: w.writeheader()
    w.writerows(out)
print("WROTE",p)
