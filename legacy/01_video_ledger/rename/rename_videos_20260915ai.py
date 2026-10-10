# -*- coding: utf-8 -*-
"""録画 VID_<session>_HHMMSS.mp4 を DAM★とも **Ai** CSV と時刻窓で突合し 曲名_<session>_HHMMSS.mp4 に改名（2026-09-15・rename_videos_20260915.py の Ai 派生）。
   **既定は dry-run（表示のみ）**・`--apply` で改名＋rename_mapping.csv 作成。
   規則：動画開始時刻 < 採点時刻(scoringDateTime) ≦ 次の動画の開始時刻（最後の動画は開始＋KV_LAST_WINDOW_MIN 分＝既定10分）。
        1動画に採点行が2つ以上＝AMBIGUOUS（改名しない・手で裁定）。0＝NO_MATCH（原名のまま＝ウォームアップ等）。
        曲名の正規化：先頭「[生音]」→末尾「(生音)」（Ai は元々「(生音)」表記・Hearts 由来の規約を念のため適用）・Windows禁止文字 \\/:*?"<>| は「_」。
   Ai CSV＝dam_api/ の最新 dam_ai_full_*.csv（KV_DAM_CSV で明示可）。
   使い方: python scripts/rename_videos_20260915ai.py [session] [--apply]
"""
import os, sys, csv, re, datetime
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915ai import CfgAi

C = CfgAi(); S = C.S; APPLY = "--apply" in sys.argv
dam = C.latest_dam_csv()
if not dam: print(f"STOP: Ai CSV が無い（dam_api/{C.dam_glob}）"); raise SystemExit(2)
d = pd.read_csv(dam, encoding="utf-8-sig", low_memory=False); d["dt"] = pd.to_datetime(d["scoringDateTime"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
day = datetime.datetime.strptime(S, "%Y%m%d"); d = d[(d["dt"] >= day) & (d["dt"] < day + datetime.timedelta(days=1))].sort_values("dt")
allv = sorted((datetime.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S"), f) for f in os.listdir(C.vid_dir) for m in [re.search(rf"_({S}_\d{{6}})\.mp4$", f)] if m)
vids = [f for _, f in allv if f.startswith("VID_")]
if not vids: print(f"改名対象の VID_{S}_*.mp4 が無い（既に改名済み？）"); raise SystemExit(0)
LASTW = datetime.timedelta(minutes=float(os.environ.get("KV_LAST_WINDOW_MIN", "10")))
win = {f: (st, (allv[i + 1][0] if i + 1 < len(allv) else st + LASTW)) for i, (st, f) in enumerate(allv)}
def norm(song):
    s = str(song).strip(); m = re.fullmatch(r"\[生音\](.+)", s)
    if m: s = m.group(1).strip() + "(生音)"
    return re.sub(r'[\\/:*?"<>|]', "_", s)
rows = []
for v in vids:
    st, en = win[v]
    hit = d[(d["dt"] > st) & (d["dt"] <= en)]
    if len(hit) == 1:
        r = hit.iloc[0]; song = norm(r["songName"]); new = f"{song}_{S}_{v[13:19]}.mp4"
        rows.append(dict(original=v, renamed=new, song=song, played_at=r["dt"].strftime("%Y-%m-%d %H:%M:%S"), how="TIME_WINDOW", note=""))
    elif len(hit) == 0: rows.append(dict(original=v, renamed=v, song="", played_at="", how="NO_MATCH", note="採点行なし（ウォームアップ等）"))
    else: rows.append(dict(original=v, renamed=v, song="", played_at="", how="AMBIGUOUS", note="採点行が複数: " + " / ".join(f"{norm(x.songName)}@{x.dt:%H:%M:%S}" for x in hit.itertuples())))
print(f"ai_csv={os.path.basename(dam)} 当日行={len(d)} 動画={len(vids)}")
for r in rows: print(f"{r['how']:12s} {r['original']} -> {r['renamed']}  {r['played_at']} {r['note']}")
n_ok = sum(1 for r in rows if r["how"] == "TIME_WINDOW"); print(f"TIME_WINDOW={n_ok} NO_MATCH={sum(r['how']=='NO_MATCH' for r in rows)} AMBIGUOUS={sum(r['how']=='AMBIGUOUS' for r in rows)}")
if not APPLY: print("dry-run（--apply で改名＋rename_mapping.csv 作成）"); raise SystemExit(0)
names = [r["renamed"] for r in rows if r["how"] == "TIME_WINDOW"]
if len(names) != len(set(names)): print("STOP: 改名後の名前が重複"); raise SystemExit(2)
for r in rows:
    if r["how"] == "TIME_WINDOW":
        os.rename(os.path.join(C.vid_dir, r["original"]), os.path.join(C.vid_dir, r["renamed"]))
mp = C.rename_csv; old = []
if os.path.exists(mp): old = list(csv.DictReader(open(mp, encoding="utf-8-sig")))
with open(mp, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["original", "renamed", "song", "played_at", "how", "note"]); w.writeheader()
    for r in old: w.writerow({k: r.get(k, "") for k in w.fieldnames})
    for r in rows: w.writerow(r)
print(f"RENAMED {n_ok} → {mp}")
