# -*- coding: utf-8 -*-
"""10/6 版 G1（dataset/G1_events_20261006_wao.csv 986行＋G1_events_20261006_ai.csv 315行・目視追記なし）を統合台帳 dataset/G1_events_all.csv に結合する（2026-10-08・本人「なんで統合できないの？」＝裁定待ちを置かない規則に従い実行）。
   merge_g1_all_20260924.py のコピー。列対応は同じ（session=20261006／take＝wav の stem／idx列＝同値／audio_time_sec＝同期済み）。Ai 9曲と WAO 23曲は同じ session に入れる（6/28・8/7 の Ai 回と同じ扱い＝技法セットは counter 列で区別できる）。
   退避＝G1_events_all_pre20261006.bak.csv（初回のみ）。冪等。"""
import os, shutil, pandas as pd
ROOT = r"G:\研究アーカイブ\成果物\動画解析"; ALL = os.path.join(ROOT, "dataset", "G1_events_all.csv"); BAK = ALL[:-4] + "_pre20261006.bak.csv"
SRCS = [os.path.join(ROOT, "dataset", "G1_events_20261006_wao.csv"), os.path.join(ROOT, "dataset", "G1_events_20261006_ai.csv")]
A = pd.read_csv(ALL, encoding="utf-8-sig"); n0 = len(A)
if (A["session"] == 20261006).any(): print("already merged:", (A["session"] == 20261006).sum()); raise SystemExit(0)
G = pd.concat([pd.read_csv(s, encoding="utf-8-sig") for s in SRCS], ignore_index=True)
assert G["pts_source"].isin(["pts_bands_20261006","pts_bands_ai_20261006"]).all(), "pts_source が揃っていない: " + str(G["pts_source"].unique())
assert G["wav"].notna().all() and G["audio_time_sec"].notna().all(), "wav/audio_time_sec に欠損"
N = pd.DataFrame(dict(session=20261006, video=G["video"], counter=G["counter"], count_value=G["count_value"].astype(int), delta=G["delta"].astype(int),
                      video_time_sec=G["video_time_sec"], stable_read_time_sec=G["stable_read_time_sec"], take=G["wav"].str[:-4],
                      audio_time_sec=G["audio_time_sec"], video_time_idx_sec=G["video_time_sec"], stable_read_time_idx_sec=G["stable_read_time_sec"],
                      multi_increment=G["multi_increment"].astype(int)))[list(A.columns)]
assert N.isna().sum().sum() == 0, "新規行に欠損"
if not os.path.exists(BAK): shutil.copy(ALL, BAK)
out = pd.concat([A, N], ignore_index=True); out.to_csv(ALL, index=False, encoding="utf-8-sig")
chk = pd.read_csv(ALL, encoding="utf-8-sig")
print(f"rows {n0} -> {len(chk)} (+{len(N)})"); print(chk.groupby("session").size().to_string())
new = chk[chk.session == 20261006]
print("dup(session,take,counter,count_value) new =", new.duplicated(["session", "take", "counter", "count_value"]).sum(), " all =", chk.duplicated(["session", "take", "counter", "count_value"]).sum())
print("dup(video,counter,count_value) all =", chk.duplicated(["video", "counter", "count_value"]).sum())
print("na new =", new.isna().sum().sum())
print("10/6 videos =", N["video"].nunique(), " takes =", N["take"].nunique(), " counters =", N["counter"].value_counts().to_dict())
print("backup", BAK, os.path.getsize(BAK))
