# -*- coding: utf-8 -*-
"""pipeline_v2 の再開（読取ランナー起動→30本待ち→verify→G1→照合）。PCスリープ等で監督が止まった時に使う。"""
import os, time, glob, subprocess, datetime
ROOT=r"G:\研究アーカイブ\成果物\動画解析"; OUT=os.path.join(ROOT,"dataset","blind_20260911")
LOG=open(os.path.join(OUT,"pipeline_v2.log"),"a",encoding="utf-8")
def log(m): LOG.write(f"{datetime.datetime.now():%H:%M:%S} [resume] {m}\n"); LOG.flush()
def wsl(cmd): return subprocess.run(["wsl","-d","Ubuntu-26.04","-u","toshiaki","--","bash","-lc",cmd],capture_output=True,text=True,encoding="utf-8",errors="replace")
def n_done(): return len([p for p in glob.glob(os.path.join(ROOT,"results_wao_20260911","*","counter_events.csv")) if ".mp4.part" not in p])
# 未完了ディレクトリ（counter_events.csv 無し）は消してランナーにやり直させる
for d in glob.glob(os.path.join(ROOT,"results_wao_20260911","*")):
    if os.path.isdir(d) and not os.path.exists(os.path.join(d,"counter_events.csv")):
        for f in glob.glob(os.path.join(d,"*")): os.remove(f)
        os.rmdir(d); log(f"removed partial {os.path.basename(d)}")
r=wsl("rm -rf /tmp/claims_20260911; pkill -f 'run_read_2026091[1]' ; pkill -f 'read_counters_wa[o]'; sleep 1; bash '/mnt/g/研究アーカイブ/成果物/動画解析/scripts/launch_read_20260911.sh'")
log("readers relaunched: "+r.stdout.strip().replace("\n"," | ")[:300])
t1=time.time()
while time.time()-t1<5*3600:
    n=n_done(); log(f"read done={n}")
    if n>=30: break
    time.sleep(60)
res=[]
for sc in ("verify_20260911.py","g1_and_match_20260911.py"):
    p=subprocess.run(["python",os.path.join(ROOT,"scripts",sc)],capture_output=True,text=True,encoding="utf-8",errors="replace",env=dict(os.environ,PYTHONIOENCODING="utf-8"))
    res.append(f"### {sc} rc={p.returncode}\n{p.stdout[-6000:]}\n{p.stderr[-800:]}")
open(os.path.join(OUT,"PIPELINE_V2_DONE.txt"),"w",encoding="utf-8").write(f"finished {datetime.datetime.now():%Y-%m-%d %H:%M:%S} done={n_done()}\n\n"+"\n\n".join(res))
log("ALL DONE")
