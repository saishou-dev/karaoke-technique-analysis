# -*- coding: utf-8 -*-
"""録音 mapping.csv に sync が読む列（original／renamed／song／played_at／status）を補う（2026-10-14 版・新規）。

背景＝ノートPCから受け取った `録音\\songs_<S>_raw\\mapping.csv` は take／src／end_time／bytes／md5 だけで song・played_at が空。
sync_20260915.py は `played_at`（Hearts の採点時刻と同じ文字列 "YYYY-MM-DD HH:MM:SS"）と `renamed`（WAVファイル名）で
rename_mapping.csv と結合するため、先に埋める必要がある。

突合方式＝**窓方式（10/6 と同じ）**：WAVファイル名の時刻＝録音終了時刻、長さ＝wave ヘッダの nframes/framerate から開始時刻を出し、
「開始−15秒 〜 終了+15秒」に採点時刻が入る行を候補にする。候補が1件のときだけ確定（0件・2件以上は status に残して空のまま）。
根拠＝WAO は曲末に採点されるため採点時刻は録音終了の 10〜30 秒前に入る（10/14 実測＝終了 −10〜−24 秒）。
9/24 の「最近傍 ±120 秒」ではなく窓方式を使う理由＝10/6 に Ai（曲頭採点）が外れた実績。

原本は mapping_pre_fill_<S>.bak.csv に退避（冪等＝再実行しても同じ結果・既に埋まっていれば触らない）。
使い方: python scripts/fill_wav_mapping_20261014.py [20261014] [--apply]
"""
import os, re, csv, sys, wave, shutil, datetime
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20261014 import Cfg1014

C = Cfg1014(); S = C.S; APPLY = "--apply" in sys.argv
MAP = os.path.join(C.wav_dir, "mapping.csv")
if not os.path.exists(MAP): print(f"STOP: 録音 mapping.csv が無い {MAP}"); raise SystemExit(2)
if not os.path.exists(C.dam_csv): print(f"STOP: Hearts CSV が無い {C.dam_csv}"); raise SystemExit(2)

day = datetime.datetime.strptime(S, "%Y%m%d")
d = pd.read_csv(C.dam_csv, encoding="utf-8-sig", low_memory=False)
d["dt"] = pd.to_datetime(d["scoringDateTime"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
d = d[(d["dt"] >= day) & (d["dt"] < day + datetime.timedelta(days=1))][["dt", "songName"]].sort_values("dt").reset_index(drop=True)


def norm(song):
    s = str(song).strip(); m = re.fullmatch(r"\[生音\](.+)", s)
    if m: s = m.group(1).strip() + "(生音)"
    return re.sub(r'[\\/:*?"<>|]', "_", s)


src_rows = list(csv.DictReader(open(MAP, encoding="utf-8-sig")))
W = datetime.timedelta(seconds=15)
out = []
for r in src_rows:
    take = (r.get("take") or r.get("renamed") or "").strip()
    m = re.fullmatch(rf"(T\d+)_{S}_(\d{{6}})\.wav", take)
    path = os.path.join(C.wav_dir, take)
    row = dict(original=(r.get("src") or r.get("original") or "").strip(), renamed=take, song=r.get("song") or "",
               played_at=r.get("played_at") or "", status="", **{k: r.get(k, "") for k in ("take", "src", "end_time", "bytes", "md5")})
    if not m or not os.path.exists(path):
        row["status"] = "SKIP 名前がTnn_<S>_HHMMSS.wav でない／実体が無い"; out.append(row); continue
    if row["played_at"]:
        row["status"] = "KEEP 既に played_at あり"; out.append(row); continue
    end = datetime.datetime.strptime(S + m.group(2), "%Y%m%d%H%M%S")
    try:
        with wave.open(path) as w: dur = w.getnframes() / float(w.getframerate())
    except Exception as e:
        row["status"] = f"SKIP wave ヘッダが読めない {e}"; out.append(row); continue
    start = end - datetime.timedelta(seconds=dur)
    hit = d[(d["dt"] >= start - W) & (d["dt"] <= end + W)]
    if len(hit) == 1:
        t = hit.iloc[0]
        row["song"] = norm(t["songName"]); row["played_at"] = t["dt"].strftime("%Y-%m-%d %H:%M:%S")
        row["status"] = (f"OK song=WINDOW(start-15s..end+15s) machine=wao dur={dur:.2f}s "
                         f"diff_end={(t['dt'] - end).total_seconds():+.0f}s n_cand=1")
    else:
        row["status"] = f"AMBIGUOUS n_cand={len(hit)} 窓={start:%H:%M:%S}-{end:%H:%M:%S}（手で確定）"
    out.append(row)

n_ok = sum(1 for r in out if r["status"].startswith("OK"))
n_keep = sum(1 for r in out if r["status"].startswith("KEEP"))
for r in out: print(f"{r['renamed']:28s} {r['played_at']:19s} {r['song'][:28]:28s} {r['status']}")
pa = [r["played_at"] for r in out if r["played_at"]]
print(f"\nWAV={len(out)} 充填={n_ok} 既存維持={n_keep} 未確定={len(out) - n_ok - n_keep} played_at重複={len(pa) - len(set(pa))}")
if len(pa) != len(set(pa)): print("STOP: played_at が重複している（1採点行に2本のWAV）"); raise SystemExit(2)
if not APPLY: print("dry-run（--apply で mapping.csv を書き換え・原本は mapping_pre_fill_<S>.bak.csv へ）"); raise SystemExit(0)
bak = os.path.join(C.wav_dir, f"mapping_pre_fill_{S}.bak.csv")
if not os.path.exists(bak): shutil.copy2(MAP, bak); print("BACKUP", bak)
fields = ["original", "renamed", "song", "played_at", "status", "take", "src", "end_time", "bytes", "md5"]
with open(MAP, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in out: w.writerow({k: r.get(k, "") for k in fields})
print(f"WROTE {MAP} rows={len(out)} filled={n_ok}")
