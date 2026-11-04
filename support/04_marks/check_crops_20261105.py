# -*- coding: utf-8 -*-
"""案D 目視点検用＝各技法2件について、D の時刻と出現+0.3秒の 4K フレームを mark_x 中心に切り出し、mark_x に縦線を引いて並べる・2026-11-05 Opus
出力＝out/mark_pos_time_20261105/check_mark_vs_cursor_20261105.png（左＝D の時刻＝カーソルが mark_x にあるはず・右＝マーク出現後）"""
import sys, os, subprocess, numpy as np, pandas as pd, cv2
sys.path.insert(0, r"G:\研究アーカイブ\成果物\動画解析\marks_20260928\scripts")
import detect_marks as dm
OUT=r"G:\研究アーカイブ\成果物\動画解析\marks_20260928\out\mark_pos_time_20261105"
E=pd.read_csv(os.path.join(OUT,"D_times_20261105.csv")); E=E[E.D.notna()]
pick=pd.concat([E[E.counter==t].head(2) for t in ["vibrato","shakuri","kobushi","accent","hammering","fall"]])
tiles=[]
for _,r in pick.iterrows():
    dm.configure(str(r.session)); cx,cy,cw,ch=dm.roi_for(r.video,pd.read_csv(dm.BAND_CSV))
    tiles_r=[]
    for t in (r.D, r.mark_time_sec+0.3):
        cmd=["ffmpeg","-v","error","-ss",f"{t:.3f}","-i",os.path.join(dm.VID_DIR,r.video+".mp4"),"-frames:v","1","-f","rawvideo","-pix_fmt","bgr24","pipe:1"]
        raw=subprocess.run(cmd,stdout=subprocess.PIPE,stdin=subprocess.DEVNULL).stdout
        f=np.frombuffer(raw,np.uint8)[:3840*2160*3].reshape(2160,3840,3).copy()
        x=int(r.mark_x); y=int(r.mark_y); x0=max(0,x-500); y0=max(0,y-200)
        c=f[y0:y0+400, x0:x0+1000].copy(); cv2.line(c,(x-x0,0),(x-x0,399),(0,255,255),2)
        cv2.putText(c,f"{r.counter} {'D' if t==r.D else 'mark+0.3'} t={t:.2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        tiles_r.append(cv2.resize(c,(500,200)))
    tiles.append(np.hstack(tiles_r))
cv2.imencode(".png",np.vstack(tiles))[1].tofile(os.path.join(OUT,"check_mark_vs_cursor_20261105.png"))
print(pick[["video","counter","mark_time_sec","D","mark_x"]].to_string())
