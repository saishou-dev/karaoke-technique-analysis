# cv2の連番インデックスと実時刻の対応を実測する（Universe 0807・こぶし1が上がる付近）
import cv2, sys
V="/mnt/g/研究アーカイブ/ビデオ素材/20260807/Universe(ドラえもんアニメバージョン)_20260807_133455.mp4"
OUT="/mnt/g/研究アーカイブ/成果物/動画解析/dataset/pts/check/"
cap=cv2.VideoCapture(V); fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("fps",fps,"frames",n, "backend", cap.getBackendName(), flush=True)
want=set(range(2800,3160,20)); fidx=-1; log=[]
while True:
    ok,img=cap.read()
    if not ok: break
    fidx+=1
    if fidx>3160: break
    if fidx in want:
        ms=cap.get(cv2.CAP_PROP_POS_MSEC)
        h,w=img.shape[:2]; band=img[int(h*0.315):int(h*0.425), int(w*0.02):int(w*0.52)]
        band=cv2.resize(band,(960,int(960*band.shape[0]/band.shape[1])))
        cv2.putText(band,f"idx={fidx} idx/fps={fidx/fps:.2f}s posmsec={ms/1000:.2f}s",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
        cv2.imwrite(OUT+f"f{fidx}.png",band); log.append((fidx,round(fidx/fps,3),round(ms/1000,3)))
print(log, flush=True)
