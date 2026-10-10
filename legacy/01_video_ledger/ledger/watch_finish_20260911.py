# -*- coding: utf-8 -*-
"""読取完了の見張り（2026-09-12・Windows python・別プロセス）。results_wao_20260911 の counter_events.csv が30本揃ったら verify→G1→照合を実行し、PIPELINE_DONE.txt に結果を書く。最長5時間。"""
import os, time, subprocess, glob, datetime
ROOT=r"G:\研究アーカイブ\成果物\動画解析"; RES=os.path.join(ROOT,"results_wao_20260911"); OUT=os.path.join(ROOT,"dataset","blind_20260911")
def n_done(): return len([p for p in glob.glob(os.path.join(RES,"*","counter_events.csv")) if ".mp4.part" not in p])
t0=time.time(); log=open(os.path.join(OUT,"watch_finish.log"),"a",encoding="utf-8")
while time.time()-t0<5*3600:
    n=n_done(); log.write(f"{datetime.datetime.now():%H:%M:%S} done={n}\n"); log.flush()
    if n>=30: break
    time.sleep(120)
res=[]
for sc in ("verify_20260911.py","g1_and_match_20260911.py"):
    p=subprocess.run(["python",os.path.join(ROOT,"scripts",sc)],capture_output=True,text=True,encoding="utf-8",errors="replace",env=dict(os.environ,PYTHONIOENCODING="utf-8"))
    res.append(f"### {sc} rc={p.returncode}\n{p.stdout[-6000:]}\n{p.stderr[-1500:]}")
open(os.path.join(OUT,"PIPELINE_DONE.txt"),"w",encoding="utf-8").write(f"finished {datetime.datetime.now():%Y-%m-%d %H:%M:%S} done={n_done()}\n\n"+"\n\n".join(res))
