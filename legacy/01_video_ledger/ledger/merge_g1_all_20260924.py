# -*- coding: utf-8 -*-
"""9/24 版 G1（dataset/G1_events_20260924.csv・1,230行・目視追記なし）を統合台帳 dataset/G1_events_all.csv に結合する（2026-09-25 12:2x・本人裁定）。
   merge_g1_all_20260915.py のコピー。列対応（9/12 の 9/11 結合＝8/12→9/12 型と同じ規約）：session=20260924／take＝wav の stem（SSL2 録音 T01..・9/11 型）／
   video_time_idx_sec・stable_read_time_idx_sec＝video_time_sec・stable_read_time_sec と同値（帯pts時刻・index 換算の旧値は無い）／
   audio_time_sec＝G1 の audio_time_sec（録音WAV同期済み・sync_offsets_20260924.csv）。台帳に audio_source 列は無い（9/15 の phone 注記は受信箱・04_research 側）。
   退避＝G1_events_all_pre20260924.bak.csv（初回のみ）。冪等（session 20260924 が既にあれば何もしない）。
   検証＝行数・session 別件数・重複ゼロ（session+take+counter+count_value・session+take+counter+video_time_sec）・新規行の欠損ゼロ。"""
import os, shutil, pandas as pd
ROOT = r"G:\研究アーカイブ\成果物\動画解析"; ALL = os.path.join(ROOT, "dataset", "G1_events_all.csv"); BAK = ALL[:-4] + "_pre20260924.bak.csv"
SRC = os.path.join(ROOT, "dataset", "G1_events_20260924.csv")
A = pd.read_csv(ALL, encoding="utf-8-sig"); n0 = len(A)
if (A["session"] == 20260924).any(): print("already merged:", (A["session"] == 20260924).sum()); raise SystemExit(0)
G = pd.read_csv(SRC, encoding="utf-8-sig")
assert (G["pts_source"] == "pts_bands_20260924").all() and G["wav"].notna().all() and G["audio_time_sec"].notna().all(), "前提列が揃っていない"
assert "audio_source" not in G.columns or (G["audio_source"] != "phone").all(), "audio_source が phone＝9/24 は SSL2 録音のはず"
N = pd.DataFrame(dict(session=20260924, video=G["video"], counter=G["counter"], count_value=G["count_value"].astype(int), delta=G["delta"].astype(int),
                      video_time_sec=G["video_time_sec"], stable_read_time_sec=G["stable_read_time_sec"], take=G["wav"].str[:-4],
                      audio_time_sec=G["audio_time_sec"], video_time_idx_sec=G["video_time_sec"], stable_read_time_idx_sec=G["stable_read_time_sec"],
                      multi_increment=G["multi_increment"].astype(int)))[list(A.columns)]
assert N.isna().sum().sum() == 0, "新規行に欠損"
if not os.path.exists(BAK): shutil.copy(ALL, BAK)
out = pd.concat([A, N], ignore_index=True); out.to_csv(ALL, index=False, encoding="utf-8-sig")
chk = pd.read_csv(ALL, encoding="utf-8-sig")
print(f"rows {n0} -> {len(chk)} (+{len(N)})"); print(chk.groupby("session").size().to_string())
new = chk[chk.session == 20260924]
print("dup(session,take,counter,count_value) new =", new.duplicated(["session", "take", "counter", "count_value"]).sum(), " all =", chk.duplicated(["session", "take", "counter", "count_value"]).sum())
print("dup(session,take,counter,video_time_sec) new =", new.duplicated(["session", "take", "counter", "video_time_sec"]).sum(), " all =", chk.duplicated(["session", "take", "counter", "video_time_sec"]).sum())
print("dup(video,counter,count_value) all =", chk.duplicated(["video", "counter", "count_value"]).sum())
print("na new =", new.isna().sum().sum(), " na all (take/audio_time_sec 旧セッション由来) =", chk.isna().sum().to_dict())
print("9/24 videos =", N["video"].nunique(), " takes =", N["take"].nunique(), " multi_increment =", int(N["multi_increment"].sum()), " counters =", N["counter"].value_counts().to_dict())
print("backup", BAK, os.path.getsize(BAK))
