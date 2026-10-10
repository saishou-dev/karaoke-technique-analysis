# -*- coding: utf-8 -*-
"""pipeline_v2 の待機専用監督（ランナーは再起動しない）：30本揃うまで待ち、verify→G1→照合を実行して PIPELINE_V2_DONE.txt に書く。
   途中でランナーが全滅していたら（60秒×5回 進捗なし＆WSLプロセス0）ランナーだけ再起動する。"""
import os, time, glob, subprocess, datetime
ROOT=r"G:\研究アーカイブ\成果物\動画解析"; OUT=os.path.join(ROOT,"dataset","blind_20260911")
LOG=open(os.path.join(OUT,"pipeline_v2.log"),"a",encoding="utf-8")
def log(m): LOG.write(f"{datetime.datetime.now():%H:%M:%S} [wait] {m}\n"); LOG.flush()
def wsl(cmd): return subprocess.run(["wsl","-d","Ubuntu-26.04","-u","toshiaki","--","bash","-lc",cmd],capture_output=True,text=True,encoding="utf-8",errors="replace")
def n_done(): return len([p for p in glob.glob(os.path.join(ROOT,"results_wao_20260911","*","counter_events.csv")) if ".mp4.part" not in p])
def readers():
    r=wsl("pgrep -fa 'read_counters_wa[o]' | wc -l");
    try: return int(r.stdout.strip() or 0)
    except: return -1
log("start pid=%d"%os.getpid()); t1=time.time(); last=n_done(); stall=0
while time.time()-t1<5*3600:
    n=n_done(); rd=readers(); log(f"read done={n} readers={rd}")
    if n>=30: break
    stall = stall+1 if (n==last and rd==0) else 0; last=n
    if stall>=3:
        for d in glob.glob(os.path.join(ROOT,"results_wao_20260911","*")):
            if os.path.isdir(d) and not os.path.exists(os.path.join(d,"counter_events.csv")):
                for f in glob.glob(os.path.join(d,"*")): os.remove(f)
                os.rmdir(d)
        r=wsl("rm -rf /tmp/claims_20260911; bash '/mnt/g/研究アーカイブ/成果物/動画解析/scripts/launch_read_20260911.sh'"); log("relaunched readers: "+r.stdout.strip().replace("\n"," | ")[:200]); stall=0
    time.sleep(60)
res=[]
for sc in ("verify_20260911.py","g1_and_match_20260911.py"):
    p=subprocess.run([os.environ.get("PYEXE","python"),os.path.join(ROOT,"scripts",sc)],capture_output=True,text=True,encoding="utf-8",errors="replace",env=dict(os.environ,PYTHONIOENCODING="utf-8"))
    res.append(f"### {sc} rc={p.returncode}\n{p.stdout[-6000:]}\n{p.stderr[-800:]}")
open(os.path.join(OUT,"PIPELINE_V2_DONE.txt"),"w",encoding="utf-8").write(f"finished {datetime.datetime.now():%Y-%m-%d %H:%M:%S} done={n_done()}\n\n"+"\n\n".join(res))
log("ALL DONE")
