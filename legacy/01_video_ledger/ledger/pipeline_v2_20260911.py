# -*- coding: utf-8 -*-
"""9/11 回収パイプライン v2 監督（2026-09-12・Windows python・別プロセス）
 1) プローブv2（band_probe_v2_20260911.csv・30本）の完了を待つ
 2) 旧帯（v1パラメータ）を bands_wao_v1_20260911/ へ退避、旧結果を results_wao_20260911_v1/ へ退避
 3) v2パラメータで30本を再クロップ（3並列・傾き補正・2480x120）
 4) WSLの読取ランナー×4（v0.8d・setsid nohup）を起動し、30本の完了を待つ
 5) verify → G1 → 予測照合 を実行し PIPELINE_V2_DONE.txt に書く
"""
import os, csv, math, time, glob, shutil, subprocess, datetime, concurrent.futures as cf
ROOT=r"G:\研究アーカイブ\成果物\動画解析"; VID=r"G:\研究アーカイブ\ビデオ素材\20260911"
FF=os.path.join(os.environ["LOCALAPPDATA"],"Microsoft","WinGet","Packages","Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe","ffmpeg-9.0.1-full_build","bin","ffmpeg.exe")
OUT=os.path.join(ROOT,"dataset","blind_20260911"); LOG=open(os.path.join(OUT,"pipeline_v2.log"),"a",encoding="utf-8")
def log(m): LOG.write(f"{datetime.datetime.now():%H:%M:%S} {m}\n"); LOG.flush()
def wsl(cmd): return subprocess.run(["wsl","-d","Ubuntu-26.04","-u","toshiaki","--","bash","-lc",cmd],capture_output=True,text=True,encoding="utf-8",errors="replace")
probe=os.path.join(ROOT,"dataset","band_probe_v2_20260911.csv"); plog=os.path.join(ROOT,"dataset","band_probe_v2_20260911.log")
t0=time.time()
while time.time()-t0<3600:
    if os.path.exists(plog) and "WROTE" in open(plog,encoding="utf-8",errors="replace").read(): break
    time.sleep(20)
rows=list(csv.DictReader(open(probe,encoding="utf-8")))
log(f"probe v2 rows={len(rows)}")
if len(rows)<30: log("probe rows <30 -> abort"); raise SystemExit(1)
# 2) 退避
b1=os.path.join(ROOT,"bands_wao_v1_20260911"); os.makedirs(b1,exist_ok=True)
for f in glob.glob(os.path.join(ROOT,"bands_wao","*_20260911_*.mp4")):
    shutil.move(f,os.path.join(b1,os.path.basename(f)))
r1=os.path.join(ROOT,"results_wao_20260911_v1")
if os.path.exists(os.path.join(ROOT,"results_wao_20260911")) and not os.path.exists(r1):
    shutil.move(os.path.join(ROOT,"results_wao_20260911"),r1)
os.makedirs(os.path.join(ROOT,"results_wao_20260911"),exist_ok=True)
log("moved v1 bands/results")
# 3) crop
def job(r):
    name=r["video"]; out=os.path.join(ROOT,"bands_wao",name+".mp4")
    xL,yL,w,h=int(r["xL"]),int(r["yL"]),int(r["w"]),int(r["h"]); ang=float(r["angle_deg"]); mx,my=20,48
    vf=f"crop={w+2*mx}:{h+2*my}:{xL-mx}:{yL-my},rotate={-math.radians(ang):.6f}:c=black,crop={w}:{h}:{mx}:{my},scale=2480:120"
    t=time.time(); p=subprocess.run([FF,"-y","-v","error","-threads","4","-i",os.path.join(VID,name+".mp4"),"-vf",vf,"-c:v","libx264","-crf","15","-preset","fast","-an",out+".part.mp4"],capture_output=True,text=True)
    if p.returncode==0: os.replace(out+".part.mp4",out); return f"ok {name} {time.time()-t:.0f}s"
    return f"err {name} {p.stderr[-150:]}"
with cf.ThreadPoolExecutor(max_workers=3) as ex:
    for s in ex.map(job,rows): log(s)
log("crop done")
# 4) readers
wsl("rm -rf /tmp/claims_20260911; cp '/mnt/g/研究アーカイブ/成果物/動画解析/scripts/read_counters_wao.py' ~/karaoke-vision/; bash '/mnt/g/研究アーカイブ/成果物/動画解析/scripts/launch_read_20260911.sh'")
log("readers launched")
def n_done(): return len([p for p in glob.glob(os.path.join(ROOT,"results_wao_20260911","*","counter_events.csv")) if ".mp4.part" not in p])
t1=time.time()
while time.time()-t1<5*3600:
    n=n_done(); log(f"read done={n}")
    if n>=30: break
    time.sleep(120)
# 5) verify / G1 / match
res=[]
for sc in ("verify_20260911.py","g1_and_match_20260911.py"):
    p=subprocess.run(["python",os.path.join(ROOT,"scripts",sc)],capture_output=True,text=True,encoding="utf-8",errors="replace",env=dict(os.environ,PYTHONIOENCODING="utf-8"))
    res.append(f"### {sc} rc={p.returncode}\n{p.stdout[-6000:]}\n{p.stderr[-800:]}")
open(os.path.join(OUT,"PIPELINE_V2_DONE.txt"),"w",encoding="utf-8").write(f"finished {datetime.datetime.now():%Y-%m-%d %H:%M:%S} done={n_done()}\n\n"+"\n\n".join(res))
log("ALL DONE")
