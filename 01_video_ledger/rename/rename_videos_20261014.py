# -*- coding: utf-8 -*-
"""録画 VID_<S>_HHMMSS.mp4 を DAM★とも CSV と時刻窓で突合し 曲名_<S>_HHMMSS.mp4 に改名（2026-10-14 版・rename_videos_20260915.py の派生）。
   10/6 の特殊事情＝1セッションで機種が途中で変わった。**Ai CSV（前半）と Hearts CSV（後半）の当日行を1つの採点表に統合**して突合し、
   rename_mapping.csv に `machine`（ai|wao）列を足す。機種の時刻窓（MACHINE_SPLIT 10:55 より前＝ai／以後＝wao）に反する採点行があれば STOP。
   規則（9/15 版と同じ）：動画開始時刻 < 採点時刻 ≦ 次の動画の開始時刻（最後は開始＋KV_LAST_WINDOW_MIN 分＝既定10分）。
        1動画に採点行2つ以上＝AMBIGUOUS（改名しない・報告に残す）。0＝NO_MATCH（原名のまま）。採点行に動画が無いものは UNUSED_ROW として表示。
        曲名の正規化：先頭「[生音]」→末尾「(生音)」・Windows禁止文字 \\/:*?"<>| は「_」。
   **既定は dry-run**・`--apply` で改名＋rename_mapping.csv 作成。
   使い方: python scripts/rename_videos_20261014.py [20261014] [--apply]
"""
import os, sys, csv, re, datetime
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20261014 import Cfg1014, HEARTS_CSV, AI_CSV, MACHINE_SPLIT

C = Cfg1014(); S = C.S; APPLY = "--apply" in sys.argv
day = datetime.datetime.strptime(S, "%Y%m%d"); split = datetime.datetime.strptime(S + MACHINE_SPLIT, "%Y%m%d%H:%M:%S")

def load(path, machine):
    if not os.path.exists(path): print(f"STOP: CSV が無い {path}"); raise SystemExit(2)
    d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    d["dt"] = pd.to_datetime(d["scoringDateTime"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    d = d[(d["dt"] >= day) & (d["dt"] < day + datetime.timedelta(days=1))][["dt", "songName", "totalScore"]].copy()
    d["machine"] = machine; return d

ai = load(AI_CSV, "ai"); he = load(HEARTS_CSV, "wao")
bad = pd.concat([ai[ai["dt"] >= split], he[he["dt"] < split]])
if len(bad):
    print("STOP: 機種の時刻窓（split=%s）に反する採点行:" % MACHINE_SPLIT)
    for x in bad.itertuples(): print(f"  {x.machine} {x.dt:%H:%M:%S} {x.songName}")
    raise SystemExit(2)
d = pd.concat([ai, he]).sort_values("dt").reset_index(drop=True)

allv = sorted((datetime.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S"), f) for f in os.listdir(C.vid_dir) for m in [re.search(rf"_({S}_\d{{6}})\.mp4$", f)] if m)
vids = [f for _, f in allv if f.startswith("VID_")]
if not vids: print(f"改名対象の VID_{S}_*.mp4 が無い（既に改名済み？）"); raise SystemExit(0)
LASTW = datetime.timedelta(minutes=float(os.environ.get("KV_LAST_WINDOW_MIN", "10")))
win = {f: (st, (allv[i + 1][0] if i + 1 < len(allv) else st + LASTW)) for i, (st, f) in enumerate(allv)}

def norm(song):
    s = str(song).strip(); m = re.fullmatch(r"\[生音\](.+)", s)
    if m: s = m.group(1).strip() + "(生音)"
    return re.sub(r'[\\/:*?"<>|]', "_", s)

rows = []; used = set()
for v in vids:
    st, en = win[v]
    hit = d[(d["dt"] > st) & (d["dt"] <= en)]
    if len(hit) == 1:
        r = hit.iloc[0]; song = norm(r["songName"]); new = f"{song}_{S}_{v[13:19]}.mp4"; used.add(int(hit.index[0]))
        rows.append(dict(original=v, renamed=new, song=song, played_at=r["dt"].strftime("%Y-%m-%d %H:%M:%S"), how="TIME_WINDOW", note="", machine=r["machine"]))
    elif len(hit) == 0:
        rows.append(dict(original=v, renamed=v, song="", played_at="", how="NO_MATCH", note="採点行なし（ウォームアップ・途中終了等）", machine=""))
    else:
        rows.append(dict(original=v, renamed=v, song="", played_at="", how="AMBIGUOUS", machine="/".join(sorted(set(hit["machine"]))),
                         note="採点行が複数: " + " / ".join(f"{norm(x.songName)}@{x.dt:%H:%M:%S}({x.machine})" for x in hit.itertuples())))
print(f"ai_csv={os.path.basename(AI_CSV)} 当日行={len(ai)}  hearts={os.path.basename(HEARTS_CSV)} 当日行={len(he)}  統合={len(d)}  動画(VID_)={len(vids)}")
for r in rows: print(f"{r['how']:12s} {r['machine']:4s} {r['original']} -> {r['renamed']}  {r['played_at']} {r['note']}")
unused = d[~d.index.isin(used)]
for x in unused.itertuples(): print(f"UNUSED_ROW   {x.machine:4s} {x.dt:%H:%M:%S} {norm(x.songName)} {x.totalScore}（動画が対応しない採点行）")
n_ok = sum(1 for r in rows if r["how"] == "TIME_WINDOW")
print(f"TIME_WINDOW={n_ok}（ai={sum(1 for r in rows if r['how']=='TIME_WINDOW' and r['machine']=='ai')} wao={sum(1 for r in rows if r['how']=='TIME_WINDOW' and r['machine']=='wao')}） "
      f"NO_MATCH={sum(r['how']=='NO_MATCH' for r in rows)} AMBIGUOUS={sum(r['how']=='AMBIGUOUS' for r in rows)} UNUSED_ROW={len(unused)}")
if not APPLY: print("dry-run（--apply で改名＋rename_mapping.csv 作成）"); raise SystemExit(0)
names = [r["renamed"] for r in rows if r["how"] == "TIME_WINDOW"]
if len(names) != len(set(names)): print("STOP: 改名後の名前が重複"); raise SystemExit(2)
for r in rows:
    if r["how"] == "TIME_WINDOW":
        os.rename(os.path.join(C.vid_dir, r["original"]), os.path.join(C.vid_dir, r["renamed"]))
mp = C.rename_csv; old = []
if os.path.exists(mp): old = list(csv.DictReader(open(mp, encoding="utf-8-sig")))
with open(mp, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["original", "renamed", "song", "played_at", "how", "note", "machine"]); w.writeheader()
    for r in old: w.writerow({k: r.get(k, "") for k in w.fieldnames})
    for r in rows: w.writerow(r)
print(f"RENAMED {n_ok} → {mp}")
